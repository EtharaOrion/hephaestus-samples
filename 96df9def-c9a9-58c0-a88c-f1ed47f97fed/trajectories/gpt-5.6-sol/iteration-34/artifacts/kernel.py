import torch
import triton
import triton.language as tl

_BT = 16
_FBV = 32
_BBV = 16


@triton.jit
def _fwd_kernel(q, k, v, a, b, g, out, h, vn, T: tl.constexpr,
                H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                BK: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib, ih = bh // H, bh % H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    ri = tl.arange(0, 16)
    state = tl.zeros((BK, BV), tl.float32)
    scale: tl.constexpr = SCALE
    nc: tl.constexpr = tl.cdiv(T, 16)
    for c in range(nc):
        t = c * 16 + ri
        mk = (t[:, None] < T) & (rk[None, :] < K)
        mv = (t[:, None] < T) & (rv[None, :] < V)
        xbase = ((ib * T + t[:, None]) * H + ih) * K + rk[None, :]
        ybase = ((ib * T + t[:, None]) * H + ih) * V + rv[None, :]
        q0 = tl.load(q + xbase, mask=mk, other=0.0).to(tl.float32) * scale
        k0 = tl.load(k + xbase, mask=mk, other=0.0).to(tl.float32)
        a0 = tl.load(a + xbase, mask=mk, other=0.0).to(tl.float32)
        b0 = tl.load(b + xbase, mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(g + xbase, mask=mk, other=0.0).to(tl.float32)
        v0 = tl.load(v + ybase, mask=mv, other=0.0).to(tl.float32)
        gc = tl.cumsum(gg, axis=0)
        ge = tl.sum(gg, axis=0)
        ep = tl.exp(tl.maximum(gc, -60.0))
        en = tl.exp(tl.minimum(-gc, 60.0))
        qg = q0 * ep
        ki = k0 * en
        bi = b0 * en
        apre = a0 * tl.exp(tl.maximum(gc - gg, -60.0))
        aqk = tl.dot(qg, tl.trans(ki), input_precision="ieee")
        aqb = tl.dot(qg, tl.trans(bi), input_precision="ieee")
        lab = tl.dot(apre, tl.trans(bi), input_precision="ieee")
        lak = tl.dot(apre, tl.trans(ki), input_precision="ieee")
        lower = ri[:, None] > ri[None, :]
        incl = ri[:, None] >= ri[None, :]
        valid2 = (t[:, None] < T) & (t[None, :] < T)
        lab = tl.where(lower & valid2, lab, 0.0)
        lak = tl.where(lower & valid2, lak, 0.0)
        aqk = tl.where(incl & valid2, aqk, 0.0)
        aqb = tl.where(incl & valid2, aqb, 0.0)
        ident = (ri[:, None] == ri[None, :]).to(tl.float32)
        inv = ident + lab
        power = lab
        for _ in tl.static_range(0, 3):
            power = tl.dot(power, power, input_precision="ieee")
            inv += tl.dot(power, inv, input_precision="ieee")
        u = tl.dot(inv, tl.dot(lak, v0, input_precision="ieee"), input_precision="ieee")
        w = tl.dot(inv, apre, input_precision="ieee")
        hp = (((ib * nc + c) * H + ih) * K + rk[:, None]) * V + rv[None, :]
        tl.store(h + hp, state, mask=(rk[:, None] < K) & (rv[None, :] < V))
        vnew = u + tl.dot(w, state, input_precision="ieee")
        tl.store(vn + ybase, vnew, mask=mv)
        oo = tl.dot(qg, state, input_precision="ieee")
        oo += tl.dot(aqk, v0, input_precision="ieee")
        oo += tl.dot(aqb, vnew, input_precision="ieee")
        tl.store(out + ybase, oo, mask=mv)
        kg = k0 * tl.exp(tl.maximum(ge[None, :] - gc, -60.0))
        bg = b0 * tl.exp(tl.maximum(ge[None, :] - gc, -60.0))
        state *= tl.exp(tl.maximum(ge, -60.0))[:, None]
        state += tl.dot(tl.trans(kg), v0, input_precision="ieee")
        state += tl.dot(tl.trans(bg), vnew, input_precision="ieee")



@triton.jit
def _checkpoint_kernel(k, v, a, b, g, h, states, T: tl.constexpr,
                       H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                       BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib, ih = bh // H, bh % H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mask = (rk[:, None] < K) & (rv[None, :] < V)
    state = tl.zeros((BK, BV), tl.float32)
    nc: tl.constexpr = tl.cdiv(T, 16)
    for c in range(nc):
        hp = (((ib * nc + c) * H + ih) * K + rk[:, None]) * V + rv[None, :]
        tl.store(h + hp, state, mask=mask)
        for j in tl.static_range(0, 16):
            t = c * 16 + j
            valid = t < T
            xb = ((ib * T + t) * H + ih) * K + rk
            yb = ((ib * T + t) * H + ih) * V + rv
            mk = valid & (rk < K)
            mv = valid & (rv < V)
            kk = tl.load(k + xb, mask=mk, other=0.0).to(tl.float32)
            aa = tl.load(a + xb, mask=mk, other=0.0).to(tl.float32)
            bb = tl.load(b + xb, mask=mk, other=0.0).to(tl.float32)
            gg = tl.load(g + xb, mask=mk, other=0.0).to(tl.float32)
            vv = tl.load(v + yb, mask=mv, other=0.0).to(tl.float32)
            read = tl.reshape(tl.dot(aa[None, :], state, input_precision="ieee"), (BV,))
            updated = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
            state = tl.where(valid, updated, state)
            sp = (((ib * T + t) * H + ih) * K + rk[:, None]) * V + rv[None, :]
            tl.store(states + sp, state, mask=valid & mask)

@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, h, states, dv, partial,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                     NV: tl.constexpr, B: tl.constexpr, SCALE: tl.constexpr,
                     PN: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib, ih = bh // H, bh % H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    dstate = tl.zeros((BK, BV), tl.float32)
    scale: tl.constexpr = SCALE
    nc: tl.constexpr = tl.cdiv(T, 16)
    for cc in range(nc):
        c = nc - 1 - cc
        for jj in tl.static_range(0, 16):
            j = 15 - jj
            t = c * 16 + j
            valid = t < T
            xb = ((ib * T + t) * H + ih) * K + rk
            yb = ((ib * T + t) * H + ih) * V + rv
            mk = valid & (rk < K)
            mv = valid & (rv < V)
            q0 = tl.load(q + xb, mask=mk, other=0.0).to(tl.float32)
            k0 = tl.load(k + xb, mask=mk, other=0.0).to(tl.float32)
            a0 = tl.load(a + xb, mask=mk, other=0.0).to(tl.float32)
            b0 = tl.load(b + xb, mask=mk, other=0.0).to(tl.float32)
            gg = tl.load(g + xb, mask=mk, other=0.0).to(tl.float32)
            v0 = tl.load(v + yb, mask=mv, other=0.0).to(tl.float32)
            d0 = tl.load(do + yb, mask=mv, other=0.0).to(tl.float32)
            sp = (((ib * T + t) * H + ih) * K + rk[:, None]) * V + rv[None, :]
            st = tl.load(states + sp, mask=mk[:, None] & (rv[None, :] < V), other=0.0)
            if j == 0:
                hp = (((ib * nc + c) * H + ih) * K + rk[:, None]) * V + rv[None, :]
                spre = tl.load(h + hp, mask=(rk[:, None] < K) & (rv[None, :] < V), other=0.0)
            else:
                spp = (((ib * T + t - 1) * H + ih) * K + rk[:, None]) * V + rv[None, :]
                spre = tl.load(states + spp, mask=mk[:, None] & (rv[None, :] < V), other=0.0)
            dstate += q0[:, None] * (d0[None, :] * scale)
            dq = tl.sum(st * d0[None, :], axis=1) * scale
            dv0 = tl.sum(k0[:, None] * dstate, axis=0)
            dk = tl.sum(dstate * v0[None, :], axis=1)
            r = tl.reshape(tl.dot(a0[None, :], spre, input_precision="ieee"), (BV,))
            db = tl.sum(dstate * r[None, :], axis=1)
            dr = tl.reshape(tl.dot(b0[None, :], dstate, input_precision="ieee"), (BV,))
            da = tl.sum(spre * dr[None, :], axis=1)
            eg = tl.exp(gg)
            dg = tl.sum(dstate * spre, axis=1) * eg
            tl.store(dv + yb, dv0, mask=mv)
            p0 = (((iv * 5 * B + ib) * T + t) * H + ih) * K + rk
            tl.store(partial + p0, dq, mask=mk)
            tl.store(partial + p0 + PN, dk, mask=mk)
            tl.store(partial + p0 + 2 * PN, da, mask=mk)
            tl.store(partial + p0 + 3 * PN, db, mask=mk)
            tl.store(partial + p0 + 4 * PN, dg, mask=mk)
            dstate = dstate * eg[:, None] + a0[:, None] * dr[None, :]


@triton.jit
def _reduce_kernel(partial, dq, dk, da, db, dg, N: tl.constexpr,
                   NV: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = x < N
    s0 = tl.zeros((BLOCK,), tl.float32)
    s1 = tl.zeros((BLOCK,), tl.float32)
    s2 = tl.zeros((BLOCK,), tl.float32)
    s3 = tl.zeros((BLOCK,), tl.float32)
    s4 = tl.zeros((BLOCK,), tl.float32)
    stride = N
    for iv in range(NV):
        base = iv * 5 * N + x
        s0 += tl.load(partial + base, mask=m, other=0.0)
        s1 += tl.load(partial + base + stride, mask=m, other=0.0)
        s2 += tl.load(partial + base + 2 * stride, mask=m, other=0.0)
        s3 += tl.load(partial + base + 3 * stride, mask=m, other=0.0)
        s4 += tl.load(partial + base + 4 * stride, mask=m, other=0.0)
    tl.store(dq + x, s0, mask=m)
    tl.store(dk + x, s1, mask=m)
    tl.store(da + x, s2, mask=m)
    tl.store(db + x, s3, mask=m)
    tl.store(dg + x, s4, mask=m)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _FBV)
        nc = triton.cdiv(T, _BT)
        out = torch.empty_like(v)
        h = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        vn = torch.empty_like(v)
        _fwd_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, h, vn,
                                          T=T, H=H, K=K, V=V, BK=bk, BV=_FBV,
                                          SCALE=K ** -0.5, num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, h, vn)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, a, b, g, h, vn = ctx.saved_tensors
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        bv = _BBV
        nv = triton.cdiv(V, bv)
        nc = triton.cdiv(T, _BT)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _checkpoint_kernel[(nv, B * H)](k, v, a, b, g, h, states,
            T=T, H=H, K=K, V=V, BK=bk, BV=bv, num_warps=4, num_stages=2)
        dv = torch.empty_like(v)
        partial = torch.zeros((nv, 5, B, T, H, K), device=q.device, dtype=torch.float32)
        _backward_kernel[(nv, B * H)](q, k, v, a, b, g, do.contiguous(), h, states,
            dv, partial, T=T, H=H, K=K, V=V, BK=bk, BV=bv, NV=nv,
            B=B, SCALE=K ** -0.5, PN=B * T * H * K,
            num_warps=8, num_stages=2)
        dq, dk, da, db = (torch.empty_like(q) for _ in range(4))
        dg = torch.empty_like(g)
        n = B * T * H * K
        _reduce_kernel[(triton.cdiv(n, 256),)](partial, dq, dk, da, db, dg,
                                               N=n, NV=nv, BLOCK=256,
                                               num_warps=4)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
