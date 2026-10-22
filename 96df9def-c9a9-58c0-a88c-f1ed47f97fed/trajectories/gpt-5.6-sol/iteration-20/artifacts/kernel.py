import torch
import triton
import triton.language as tl

BT = 16
BV = 32
BWD_BV = 16


@triton.jit
def _forward_kernel(q, k, v, a, b, g, o, h, T: tl.constexpr, H: tl.constexpr,
                    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                    BLOCK_V: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    rt, rk = tl.arange(0, 16), tl.arange(0, BK)
    rv = iv * BLOCK_V + tl.arange(0, BLOCK_V)
    mk, mv = rk < K, rv < V
    state = tl.zeros((BK, BLOCK_V), tl.float32)
    NC: tl.constexpr = tl.cdiv(T, 16)
    scale: tl.constexpr = K ** -0.5
    for ic in range(NC):
        tt = ic * 16 + rt
        mt = tt < T
        pk = ((ib * T + tt[:, None]) * H + ih) * K + rk[None, :]
        pv = ((ib * T + tt[:, None]) * H + ih) * V + rv[None, :]
        ph = ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        tl.store(h + ph, state, mask=mk[:, None] & mv[None, :])
        q0 = tl.load(q + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32) * scale
        k0 = tl.load(k + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
        a0 = tl.load(a + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
        b0 = tl.load(b + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
        g0 = tl.load(g + pk, mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
        v0 = tl.load(v + pv, mask=mt[:, None] & mv[None, :], other=0.).to(tl.float32)
        gi = tl.cumsum(g0, axis=0)
        ge = gi - g0
        # Pairwise causal decay arguments are non-positive on lanes that matter.
        eold = tl.exp(tl.minimum(ge[:, None, :] - gi[None, :, :], 0.))
        enow = tl.exp(tl.minimum(gi[:, None, :] - gi[None, :, :], 0.))
        lower = (rt[:, None] > rt[None, :]) & mt[:, None] & mt[None, :]
        causal = (rt[:, None] >= rt[None, :]) & mt[:, None] & mt[None, :]
        Aab = tl.sum(a0[:, None, :] * b0[None, :, :] * eold, axis=2)
        Aak = tl.sum(a0[:, None, :] * k0[None, :, :] * eold, axis=2)
        Aqk = tl.sum(q0[:, None, :] * k0[None, :, :] * enow, axis=2)
        Aqb = tl.sum(q0[:, None, :] * b0[None, :, :] * enow, axis=2)
        Aab = tl.where(lower, Aab, 0.)
        Aak = tl.where(lower, Aak, 0.)
        Aqk = tl.where(causal, Aqk, 0.)
        Aqb = tl.where(causal, Aqb, 0.)
        ag = a0 * tl.exp(tl.minimum(ge, 0.))
        x = tl.dot(ag, state, input_precision='ieee') + tl.dot(Aak, v0, input_precision='ieee')
        for j in tl.static_range(1, 16):
            row = tl.sum(tl.where(rt[:, None] == j, Aab, 0.), axis=0)
            corr = tl.sum(row[:, None] * x, axis=0)
            x = tl.where((rt == j)[:, None], x + corr[None, :], x)
        qg = q0 * tl.exp(tl.minimum(gi, 0.))
        out = tl.dot(qg, state, input_precision='ieee')
        out += tl.dot(Aqk, v0, input_precision='ieee')
        out += tl.dot(Aqb, x, input_precision='ieee')
        tl.store(o + pv, out, mask=mt[:, None] & mv[None, :])
        last = tl.minimum(T - ic * 16, 16) - 1
        glast = tl.sum(tl.where(rt[:, None] == last, gi, 0.), axis=0)
        ratio = tl.exp(tl.minimum(glast[None, :] - gi, 0.))
        state *= tl.exp(tl.minimum(glast, 0.))[:, None]
        state += tl.dot(tl.trans(k0 * ratio), v0, input_precision='ieee')
        state += tl.dot(tl.trans(b0 * ratio), x, input_precision='ieee')


@triton.jit
def _expand_states_kernel(k, v, a, b, g, h, states, T: tl.constexpr,
                          H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                          BK: tl.constexpr, BLOCK_V: tl.constexpr):
    iv, ic, ibh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = ibh // H, ibh % H
    rk = tl.arange(0, BK)
    rv = iv * BLOCK_V + tl.arange(0, BLOCK_V)
    mk, mv = rk < K, rv < V
    NC: tl.constexpr = tl.cdiv(T, 16)
    ph = ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
    state = tl.load(h + ph, mask=mk[:, None] & mv[None, :], other=0.)
    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        valid = t < T
        ps = ((((ib * T + t) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        tl.store(states + ps, state, mask=valid & mk[:, None] & mv[None, :])
        pk = ((ib * T + t) * H + ih) * K + rk
        pv = ((ib * T + t) * H + ih) * V + rv
        k0 = tl.load(k + pk, mask=valid & mk, other=0.).to(tl.float32)
        a0 = tl.load(a + pk, mask=valid & mk, other=0.).to(tl.float32)
        b0 = tl.load(b + pk, mask=valid & mk, other=0.).to(tl.float32)
        g0 = tl.load(g + pk, mask=valid & mk, other=0.).to(tl.float32)
        v0 = tl.load(v + pv, mask=valid & mv, other=0.).to(tl.float32)
        read = tl.sum(a0[:, None] * state, axis=0)
        state = state * tl.exp(tl.minimum(g0, 0.))[:, None]
        state += k0[:, None] * v0[None, :] + b0[:, None] * read[None, :]


@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, states, parts, dv,
                     T: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
                     K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                     BLOCK_V: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    rk = tl.arange(0, BK)
    rv = iv * BLOCK_V + tl.arange(0, BLOCK_V)
    mk, mv = rk < K, rv < V
    ds = tl.zeros((BK, BLOCK_V), tl.float32)
    scale: tl.constexpr = K ** -0.5
    N: tl.constexpr = B * T * H * K
    NC: tl.constexpr = tl.cdiv(T, 16)
    for ric in range(NC):
        ic = NC - 1 - ric
        for rj in tl.static_range(0, 16):
            t = ic * 16 + 15 - rj
            if t < T:
                pk = ((ib * T + t) * H + ih) * K + rk
                pv = ((ib * T + t) * H + ih) * V + rv
                ps = ((((ib * T + t) * H + ih) * K + rk[:, None]) * V + rv[None, :])
                q0 = tl.load(q + pk, mask=mk, other=0.).to(tl.float32) * scale
                k0 = tl.load(k + pk, mask=mk, other=0.).to(tl.float32)
                a0 = tl.load(a + pk, mask=mk, other=0.).to(tl.float32)
                b0 = tl.load(b + pk, mask=mk, other=0.).to(tl.float32)
                g0 = tl.load(g + pk, mask=mk, other=0.).to(tl.float32)
                v0 = tl.load(v + pv, mask=mv, other=0.).to(tl.float32)
                d0 = tl.load(do + pv, mask=mv, other=0.).to(tl.float32)
                sp = tl.load(states + ps, mask=mk[:, None] & mv[None, :], other=0.)
                decay = tl.exp(tl.minimum(g0, 0.))
                read = tl.sum(a0[:, None] * sp, axis=0)
                sn = sp * decay[:, None] + k0[:, None] * v0[None, :] + b0[:, None] * read[None, :]
                ds += q0[:, None] * d0[None, :]
                xq = tl.sum(sn * d0[None, :], axis=1) * scale
                xk = tl.sum(ds * v0[None, :], axis=1)
                xb = tl.sum(ds * read[None, :], axis=1)
                dr = tl.sum(ds * b0[:, None], axis=0)
                xa = tl.sum(sp * dr[None, :], axis=1)
                xg = tl.sum(ds * sp * decay[:, None], axis=1)
                xv = tl.sum(ds * k0[:, None], axis=0)
                off = ((ib * T + t) * H + ih) * K + rk
                tl.store(parts + (iv * 5 + 0) * N + off, xq, mask=mk)
                tl.store(parts + (iv * 5 + 1) * N + off, xk, mask=mk)
                tl.store(parts + (iv * 5 + 2) * N + off, xa, mask=mk)
                tl.store(parts + (iv * 5 + 3) * N + off, xb, mask=mk)
                tl.store(parts + (iv * 5 + 4) * N + off, xg, mask=mk)
                tl.store(dv + pv, xv, mask=mv)
                ds = ds * decay[:, None] + a0[:, None] * dr[None, :]


@triton.jit
def _reduce_kernel(parts, dq, dk, da, db, dg, N: tl.constexpr,
                   NV: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = x < N
    r0 = tl.zeros((BLOCK,), tl.float32)
    r1 = tl.zeros((BLOCK,), tl.float32)
    r2 = tl.zeros((BLOCK,), tl.float32)
    r3 = tl.zeros((BLOCK,), tl.float32)
    r4 = tl.zeros((BLOCK,), tl.float32)
    for iv in tl.static_range(0, NV):
        root = iv * 5 * N
        r0 += tl.load(parts + root + 0 * N + x, mask=m, other=0.)
        r1 += tl.load(parts + root + 1 * N + x, mask=m, other=0.)
        r2 += tl.load(parts + root + 2 * N + x, mask=m, other=0.)
        r3 += tl.load(parts + root + 3 * N + x, mask=m, other=0.)
        r4 += tl.load(parts + root + 4 * N + x, mask=m, other=0.)
    tl.store(dq + x, r0, mask=m)
    tl.store(dk + x, r1, mask=m)
    tl.store(da + x, r2, mask=m)
    tl.store(db + x, r3, mask=m)
    tl.store(dg + x, r4, mask=m)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        nc, nv = triton.cdiv(T, BT), triton.cdiv(V, BV)
        out = torch.empty_like(v)
        h = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, h,
            T=T, H=H, K=K, V=V, BK=bk, BLOCK_V=BV,
            num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, h)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, a, b, g, h = ctx.saved_tensors
        do = do.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        nc, nv = triton.cdiv(T, BT), triton.cdiv(V, BWD_BV)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](k, v, a, b, g, h, states,
            T=T, H=H, K=K, V=V, BK=bk, BLOCK_V=BWD_BV,
            num_warps=8, num_stages=2)
        n = B * T * H * K
        parts = torch.zeros((nv, 5, B, T, H, K), device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        _backward_kernel[(nv, B * H)](q, k, v, a, b, g, do, states, parts, dv,
            T=T, B=B, H=H, K=K, V=V, BK=bk, BLOCK_V=BWD_BV,
            num_warps=8, num_stages=2)
        dq, dk, da, db, dg = (torch.empty_like(q), torch.empty_like(k),
                              torch.empty_like(a), torch.empty_like(b), torch.empty_like(g))
        _reduce_kernel[(triton.cdiv(n, 256),)](parts, dq, dk, da, db, dg,
            N=n, NV=nv, BLOCK=256, num_warps=8)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
