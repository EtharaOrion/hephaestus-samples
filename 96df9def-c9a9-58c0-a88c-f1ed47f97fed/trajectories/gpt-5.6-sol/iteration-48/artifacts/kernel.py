import torch
import triton
import triton.language as tl

_BT = 16
_FBV = 32
_BBV = 16


@triton.jit
def _inverse_lower(x, BT: tl.constexpr):
    ident = tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]
    r = ident.to(tl.float32)
    p = x
    for _ in tl.static_range(0, 4):
        r += tl.dot(p, r, input_precision="ieee")
        p = tl.dot(p, p, input_precision="ieee")
    return r


@triton.jit
def _forward_kernel(q, k, v, a, b, g, out,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    oi = tl.arange(0, BT)
    state = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / tl.sqrt(float(K))
    for ic in range(NC):
        tt = ic * BT + oi
        mt = tt < T
        basek = (ib * T * H + tt * H + ih) * K
        basev = (ib * T * H + tt * H + ih) * V
        mk = mt[:, None] & (ok[None, :] < K)
        mv = mt[:, None] & (ov[None, :] < V)
        q0 = tl.load(q + basek[:, None] + ok[None, :], mask=mk, other=0.0).to(tl.float32) * scale
        k0 = tl.load(k + basek[:, None] + ok[None, :], mask=mk, other=0.0).to(tl.float32)
        a0 = tl.load(a + basek[:, None] + ok[None, :], mask=mk, other=0.0).to(tl.float32)
        b0 = tl.load(b + basek[:, None] + ok[None, :], mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(g + basek[:, None] + ok[None, :], mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(v + basev[:, None] + ov[None, :], mask=mv, other=0.0).to(tl.float32)
        c = tl.cumsum(gg, axis=0)
        causal = oi[:, None] >= oi[None, :]
        strict = oi[:, None] > oi[None, :]
        d = tl.exp(c[:, None, :] - c[None, :, :])
        dp = tl.exp(c[:, None, :] - gg[:, None, :] - c[None, :, :])
        aqk = tl.sum(q0[:, None, :] * k0[None, :, :] * d, axis=2)
        aqb = tl.sum(q0[:, None, :] * b0[None, :, :] * d, axis=2)
        aak = tl.sum(a0[:, None, :] * k0[None, :, :] * dp, axis=2)
        aab = tl.sum(a0[:, None, :] * b0[None, :, :] * dp, axis=2)
        aqk = tl.where(causal & mt[:, None] & mt[None, :], aqk, 0.0)
        aqb = tl.where(causal & mt[:, None] & mt[None, :], aqb, 0.0)
        aak = tl.where(strict & mt[:, None] & mt[None, :], aak, 0.0)
        aab = tl.where(strict & mt[:, None] & mt[None, :], aab, 0.0)
        inv = _inverse_lower(aab, BT)
        ag = a0 * tl.exp(c - gg)
        u = tl.dot(inv, tl.dot(aak, vv, input_precision="ieee"), input_precision="ieee")
        w = tl.dot(inv, ag, input_precision="ieee")
        vn = u + tl.dot(w, state, input_precision="ieee")
        oo = tl.dot(aqk, vv, input_precision="ieee") + tl.dot(aqb, vn, input_precision="ieee")
        oo += tl.dot(q0 * tl.exp(c), state, input_precision="ieee")
        tl.store(out + basev[:, None] + ov[None, :], oo, mask=mv)
        end = tl.sum(tl.where(oi[:, None] == BT - 1, c, 0.0), axis=0)
        if ic == NC - 1:
            last = (T - 1) % BT
            end = tl.sum(tl.where(oi[:, None] == last, c, 0.0), axis=0)
        rem = tl.exp(end[None, :] - c)
        state = state * tl.exp(end)[:, None]
        state += tl.dot(tl.trans(k0 * rem), vv, input_precision="ieee")
        state += tl.dot(tl.trans(b0 * rem), vn, input_precision="ieee")


@triton.jit
def _state_replay_kernel(k, v, a, b, g, states,
                         T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                         V: tl.constexpr, NC: tl.constexpr,
                         BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    s = tl.zeros((BK, BV), tl.float32)
    for ic in range(NC):
        for j in tl.static_range(0, BT):
            t = ic * BT + j
            valid = t < T
            pk = (ib * T * H + t * H + ih) * K + ok
            pv = (ib * T * H + t * H + ih) * V + ov
            kk = tl.load(k + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            aa = tl.load(a + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            bb = tl.load(b + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            gg = tl.load(g + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            vv = tl.load(v + pv, mask=valid & (ov < V), other=0.0).to(tl.float32)
            read = tl.sum(aa[:, None] * s, axis=0)
            s = s * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
            ps = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
            tl.store(states + ps, s, mask=valid & (ok[:, None] < K) & (ov[None, :] < V))


@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, states, partial, dv,
                     B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, NV: tl.constexpr,
                     BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    ds = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / tl.sqrt(float(K))
    for ic0 in range(tl.cdiv(T, BT)):
        ic = tl.cdiv(T, BT) - 1 - ic0
        for j0 in tl.static_range(0, BT):
            j = BT - 1 - j0
            t = ic * BT + j
            valid = t < T
            pk = (ib * T * H + t * H + ih) * K + ok
            pv = (ib * T * H + t * H + ih) * V + ov
            q0 = tl.load(q + pk, mask=valid & (ok < K), other=0.0).to(tl.float32) * scale
            k0 = tl.load(k + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            a0 = tl.load(a + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            b0 = tl.load(b + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            gg = tl.load(g + pk, mask=valid & (ok < K), other=0.0).to(tl.float32)
            vv = tl.load(v + pv, mask=valid & (ov < V), other=0.0).to(tl.float32)
            dz = tl.load(do + pv, mask=valid & (ov < V), other=0.0).to(tl.float32)
            psn = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
            sn = tl.load(states + psn, mask=valid & (ok[:, None] < K) & (ov[None, :] < V), other=0.0)
            sp = tl.zeros((BK, BV), tl.float32)
            if t > 0:
                psp = ((((ib * T + t - 1) * H + ih) * K + ok[:, None]) * V + ov[None, :])
                sp = tl.load(states + psp, mask=(ok[:, None] < K) & (ov[None, :] < V), other=0.0)
            ds += q0[:, None] * dz[None, :]
            read = tl.sum(a0[:, None] * sp, axis=0)
            dr = tl.sum(b0[:, None] * ds, axis=0)
            dq = tl.sum(sn * dz[None, :], axis=1) * scale
            dk = tl.sum(ds * vv[None, :], axis=1)
            db = tl.sum(ds * read[None, :], axis=1)
            da = tl.sum(sp * dr[None, :], axis=1)
            dg = tl.sum(ds * (sp * tl.exp(gg)[:, None]), axis=1)
            dv0 = tl.sum(k0[:, None] * ds, axis=0)
            tl.store(dv + pv, dv0, mask=valid & (ov < V))
            plane = B * T * H * K
            base = iv * 5 * plane + ((ib * T + t) * H + ih) * K
            tl.store(partial + base + 0 * plane + ok, dq, mask=valid & (ok < K))
            tl.store(partial + base + 1 * plane + ok, dk, mask=valid & (ok < K))
            tl.store(partial + base + 2 * plane + ok, da, mask=valid & (ok < K))
            tl.store(partial + base + 3 * plane + ok, db, mask=valid & (ok < K))
            tl.store(partial + base + 4 * plane + ok, dg, mask=valid & (ok < K))
            ds = ds * tl.exp(gg)[:, None] + a0[:, None] * dr[None, :]


@triton.jit
def _reduce_kernel(partial, grads, N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    stride = N
    for iv in range(NV):
        acc += tl.load(partial + iv * 5 * stride + x, mask=x < 5 * N, other=0.0)
    tl.store(grads + x, acc, mask=x < 5 * N)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _FBV)
        out = torch.empty_like(v)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out,
                                     T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_FBV, BT=_BT,
                                     num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta, gk = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        bv = _BBV if V == triton.next_power_of_2(V) else max(16, triton.next_power_of_2(V))
        nv = triton.cdiv(V, bv)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _state_replay_kernel[(nv, B * H)](k, v, alpha, beta, gk, states,
                                          T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv, BT=_BT,
                                          num_warps=4, num_stages=2)
        partial = torch.zeros((nv, 5, B, T, H, K), device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        _backward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, do.contiguous(), states, partial, dv,
                                      B=B, T=T, H=H, K=K, V=V, NV=nv, BK=bk, BV=bv, BT=_BT,
                                      num_warps=4, num_stages=2)
        grads = torch.empty((5, B, T, H, K), device=q.device, dtype=torch.float32)
        n = B * T * H * K
        _reduce_kernel[(triton.cdiv(5 * n, 256),)](partial, grads, N=n, NV=nv, BLOCK=256)
        dq, dk, da, db, dg = (grads[i].to(x.dtype) for i, x in enumerate((q, k, alpha, beta, gk)))
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
