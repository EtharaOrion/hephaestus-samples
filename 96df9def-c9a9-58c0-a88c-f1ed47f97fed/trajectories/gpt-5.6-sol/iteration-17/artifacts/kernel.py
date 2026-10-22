import torch
import triton
import triton.language as tl

_BT = 16
_BV = 32
_BWD_BV = 16

@triton.jit
def _forward_kernel(q, k, v, a, b, g, o, hbuf, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    rk, rv, rt = tl.arange(0, BK), iv * BV + tl.arange(0, BV), tl.arange(0, 16)
    state = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / (K ** 0.5)
    for ic in range(NC):
        t = ic * 16 + rt
        mt = t < T
        xoff = (ib * T + t[:, None]) * H * K + ih * K + rk[None, :]
        voff = (ib * T + t[:, None]) * H * V + ih * V + rv[None, :]
        mtk, mtv = mt[:, None] & (rk[None, :] < K), mt[:, None] & (rv[None, :] < V)
        qm = tl.load(q + xoff, mask=mtk, other=0.0).to(tl.float32) * scale
        km = tl.load(k + xoff, mask=mtk, other=0.0).to(tl.float32)
        am = tl.load(a + xoff, mask=mtk, other=0.0).to(tl.float32)
        bm = tl.load(b + xoff, mask=mtk, other=0.0).to(tl.float32)
        gm = tl.load(g + xoff, mask=mtk, other=0.0).to(tl.float32)
        vm = tl.load(v + voff, mask=mtv, other=0.0).to(tl.float32)
        gc = tl.cumsum(gm, axis=0)
        ri, rj = rt[:, None], rt[None, :]
        causal = (rj <= ri) & mt[:, None] & mt[None, :]
        strict = (rj < ri) & mt[:, None] & mt[None, :]
        dqgate = tl.exp(tl.maximum(gc[:, None, :] - gc[None, :, :], -80.0))
        dagate = tl.exp(tl.maximum(gc[:, None, :] - gm[:, None, :] - gc[None, :, :], -80.0))
        Aqk = tl.where(causal, tl.sum(qm[:, None, :] * km[None, :, :] * dqgate, axis=2), 0.0)
        Aqb = tl.where(causal, tl.sum(qm[:, None, :] * bm[None, :, :] * dqgate, axis=2), 0.0)
        Aak = tl.where(strict, tl.sum(am[:, None, :] * km[None, :, :] * dagate, axis=2), 0.0)
        lab = tl.where(strict, tl.sum(am[:, None, :] * bm[None, :, :] * dagate, axis=2), 0.0)
        inv = tl.where(ri == rj, 1.0, 0.0)
        for row in tl.static_range(1, 16):
            coeff = tl.sum(tl.where(ri == row, lab, 0.0), axis=0)
            solved = tl.sum(coeff[:, None] * inv, axis=0)
            inv = tl.where(ri == row, inv + solved[None, :], inv)
        u = tl.dot(inv, tl.dot(Aak, vm, input_precision="ieee"), input_precision="ieee")
        w = tl.dot(inv, am * tl.exp(gc - gm), input_precision="ieee")
        vnew = u + tl.dot(w, state, input_precision="ieee")
        hoff = (((ib * H + ih) * NC + ic) * K + rk[:, None]) * V + rv[None, :]
        tl.store(hbuf + hoff, state, mask=(rk[:, None] < K) & (rv[None, :] < V))
        out = tl.dot(qm * tl.exp(gc), state, input_precision="ieee")
        out += tl.dot(Aqk, vm, input_precision="ieee") + tl.dot(Aqb, vnew, input_precision="ieee")
        tl.store(o + voff, out, mask=mtv)
        last = tl.minimum(15, T - ic * 16 - 1)
        gend = tl.sum(tl.where(rt[:, None] == last, gc, 0.0), axis=0)
        decay = tl.exp(tl.maximum(gend[None, :] - gc, -80.0))
        state = state * tl.exp(gend)[:, None]
        state += tl.dot(tl.trans(km * decay), vm, input_precision="ieee")
        state += tl.dot(tl.trans(bm * decay), vnew, input_precision="ieee")

@triton.jit
def _expand_states_kernel(k, v, a, b, g, states, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    rk, rv = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    state = tl.zeros((BK, BV), tl.float32)
    for t in range(T):
        xoff = (ib * T + t) * H * K + ih * K + rk
        voff = (ib * T + t) * H * V + ih * V + rv
        soff = ((ib * T + t) * H * K + ih * K + rk[:, None]) * V + rv[None, :]
        tl.store(states + soff, state, mask=(rk[:, None] < K) & (rv[None, :] < V))
        kv = tl.load(k + xoff, mask=rk < K, other=0.0).to(tl.float32)
        av = tl.load(a + xoff, mask=rk < K, other=0.0).to(tl.float32)
        bv = tl.load(b + xoff, mask=rk < K, other=0.0).to(tl.float32)
        gv = tl.load(g + xoff, mask=rk < K, other=0.0).to(tl.float32)
        vv = tl.load(v + voff, mask=rv < V, other=0.0).to(tl.float32)
        read = tl.sum(state * av[:, None], axis=0)
        state = state * tl.exp(gv)[:, None] + kv[:, None] * vv[None, :] + bv[:, None] * read[None, :]

@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, states, dv, partial, T: tl.constexpr, B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    rk, rv = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    ds = tl.zeros((BK, BV), tl.float32)
    scale = 1.0 / (K ** 0.5)
    for z in range(T):
        t = T - 1 - z
        xoff = (ib * T + t) * H * K + ih * K + rk
        voff = (ib * T + t) * H * V + ih * V + rv
        soff = ((ib * T + t) * H * K + ih * K + rk[:, None]) * V + rv[None, :]
        mk, mv = rk < K, rv < V
        prev = tl.load(states + soff, mask=mk[:, None] & mv[None, :], other=0.0)
        qv = tl.load(q + xoff, mask=mk, other=0.0).to(tl.float32)
        kv = tl.load(k + xoff, mask=mk, other=0.0).to(tl.float32)
        av = tl.load(a + xoff, mask=mk, other=0.0).to(tl.float32)
        bv = tl.load(b + xoff, mask=mk, other=0.0).to(tl.float32)
        gv = tl.load(g + xoff, mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(v + voff, mask=mv, other=0.0).to(tl.float32)
        dov = tl.load(do + voff, mask=mv, other=0.0).to(tl.float32)
        eg = tl.exp(gv)
        read = tl.sum(prev * av[:, None], axis=0)
        cur = prev * eg[:, None] + kv[:, None] * vv[None, :] + bv[:, None] * read[None, :]
        ds += qv[:, None] * (scale * dov[None, :])
        dqv = tl.sum(cur * dov[None, :], axis=1) * scale
        dkv = tl.sum(ds * vv[None, :], axis=1)
        dbv = tl.sum(ds * read[None, :], axis=1)
        drv = tl.sum(ds * bv[:, None], axis=0)
        dav = tl.sum(prev * drv[None, :], axis=1)
        dgv = tl.sum(ds * (eg[:, None] * prev), axis=1)
        tl.store(dv + voff, tl.sum(ds * kv[:, None], axis=0), mask=mv)
        poff = (((iv * B + ib) * T + t) * H + ih) * K + rk
        stride = NV * B * T * H * K
        tl.store(partial + poff + 0 * stride, dqv, mask=mk)
        tl.store(partial + poff + 1 * stride, dkv, mask=mk)
        tl.store(partial + poff + 2 * stride, dav, mask=mk)
        tl.store(partial + poff + 3 * stride, dbv, mask=mk)
        tl.store(partial + poff + 4 * stride, dgv, mask=mk)
        ds = ds * eg[:, None] + av[:, None] * drv[None, :]

@triton.jit
def _reduce_kernel(partial, dq, dk, da, db, dg, N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = x < N
    s0 = tl.zeros((BLOCK,), tl.float32); s1 = tl.zeros((BLOCK,), tl.float32)
    s2 = tl.zeros((BLOCK,), tl.float32); s3 = tl.zeros((BLOCK,), tl.float32); s4 = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        off = iv * N + x
        s0 += tl.load(partial + off + 0 * NV * N, mask=mask, other=0.0)
        s1 += tl.load(partial + off + 1 * NV * N, mask=mask, other=0.0)
        s2 += tl.load(partial + off + 2 * NV * N, mask=mask, other=0.0)
        s3 += tl.load(partial + off + 3 * NV * N, mask=mask, other=0.0)
        s4 += tl.load(partial + off + 4 * NV * N, mask=mask, other=0.0)
    tl.store(dq + x, s0, mask=mask); tl.store(dk + x, s1, mask=mask)
    tl.store(da + x, s2, mask=mask); tl.store(db + x, s3, mask=mask); tl.store(dg + x, s4, mask=mask)

class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v, alpha, beta, gk = [x.contiguous() for x in (q, k, v, alpha, beta, gk)]
        B, T, H, K = q.shape; V = v.shape[-1]
        bk, nc, nv = max(16, triton.next_power_of_2(K)), triton.cdiv(T, _BT), triton.cdiv(V, _BV)
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, H, nc, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, checkpoints, T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BV, num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, a, b, g, checkpoints = ctx.saved_tensors
        do = do.contiguous(); B, T, H, K = q.shape; V = v.shape[-1]
        bv = _BWD_BV if V == triton.next_power_of_2(V) else triton.next_power_of_2(V)
        bk, nv = max(16, triton.next_power_of_2(K)), triton.cdiv(V, bv)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, B * H)](k, v, a, b, g, states, T=T, H=H, K=K, V=V, BK=bk, BV=bv, num_warps=4, num_stages=2)
        dv = torch.empty_like(v)
        partial = torch.zeros((5, nv, B, T, H, K), device=q.device, dtype=torch.float32)
        _backward_kernel[(nv, B * H)](q, k, v, a, b, g, do, states, dv, partial, T=T, B=B, H=H, K=K, V=V, BK=bk, BV=bv, NV=nv, num_warps=4, num_stages=2)
        dq, dk, da, db = (torch.empty_like(q) for _ in range(4)); dg = torch.empty_like(g)
        n = B * T * H * K
        _reduce_kernel[(triton.cdiv(n, 256),)](partial, dq, dk, da, db, dg, N=n, NV=nv, BLOCK=256, num_warps=8, num_stages=2)
        return dq, dk, dv, da, db, dg

def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
