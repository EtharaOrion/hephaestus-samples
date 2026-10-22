import torch
import triton
import triton.language as tl

_BT = 16
_FBV = 32
_BBV = 16


@triton.jit
def _bounded_exp(x):
    return tl.exp(tl.maximum(tl.minimum(x, 60.0), -60.0))


@triton.jit
def _forward_kernel(q, k, v, a, b, g, o, h, T: tl.constexpr, H: tl.constexpr,
                    K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    rt = tl.arange(0, 16)
    mk = rk < K
    mv = rv < V
    state = tl.zeros((BK, BV), tl.float32)
    scale = K ** -0.5
    for ic in range(NC):
        hp = h + ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        tl.store(hp, state, mask=mk[:, None] & mv[None, :])
        ts = ic * 16 + rt
        mt = ts < T
        base_k = ((ib * T + ts[:, None]) * H + ih) * K + rk[None, :]
        gg = tl.load(g + base_k, mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32)
        gc = tl.cumsum(gg, axis=0)
        gp = gc - gg
        eg = _bounded_exp(gc)
        ep = _bounded_exp(gp)
        ieg = _bounded_exp(-gc)
        qq = tl.load(q + base_k, mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32) * (eg * scale)
        kk = tl.load(k + base_k, mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32) * ieg
        aa = tl.load(a + base_k, mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32) * ep
        bb = tl.load(b + base_k, mask=mt[:, None] & mk[None, :], other=0.0).to(tl.float32) * ieg
        base_v = ((ib * T + ts[:, None]) * H + ih) * V + rv[None, :]
        vv = tl.load(v + base_v, mask=mt[:, None] & mv[None, :], other=0.0).to(tl.float32)
        ak = tl.dot(aa, tl.trans(kk), input_precision="ieee")
        ab = tl.dot(aa, tl.trans(bb), input_precision="ieee")
        causal = rt[:, None] > rt[None, :]
        ak = tl.where(causal, ak, 0.0)
        ab = tl.where(causal, ab, 0.0)
        rhs = tl.dot(aa, state, input_precision="ieee") + tl.dot(ak, vv, input_precision="ieee")
        eye = tl.where(rt[:, None] == rt[None, :], 1.0, 0.0)
        inv = eye
        power = ab
        for _ in tl.static_range(0, 4):
            inv += tl.dot(power, inv, input_precision="ieee")
            power = tl.dot(power, power, input_precision="ieee")
        uu = tl.dot(inv, rhs, input_precision="ieee")
        qk = tl.dot(qq, tl.trans(kk), input_precision="ieee")
        qb = tl.dot(qq, tl.trans(bb), input_precision="ieee")
        incl = rt[:, None] >= rt[None, :]
        out = tl.dot(qq, state, input_precision="ieee")
        out += tl.dot(tl.where(incl, qk, 0.0), vv, input_precision="ieee")
        out += tl.dot(tl.where(incl, qb, 0.0), uu, input_precision="ieee")
        tl.store(o + base_v, out, mask=mt[:, None] & mv[None, :])
        xend = state + tl.dot(tl.trans(kk), vv, input_precision="ieee")
        xend += tl.dot(tl.trans(bb), uu, input_precision="ieee")
        end_gate = tl.sum(tl.where(rt[:, None] == 15, eg, 0.0), axis=0)
        state = xend * end_gate[:, None]


@triton.jit
def _checkpoint_kernel(k, v, a, b, g, h, T: tl.constexpr, H: tl.constexpr,
                       K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                       BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mk, mv = rk < K, rv < V
    state = tl.zeros((BK, BV), tl.float32)
    for t in range(T):
        if t % 16 == 0:
            ic = t // 16
            hp = h + ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
            tl.store(hp, state, mask=mk[:, None] & mv[None, :])
        kp = ((ib * T + t) * H + ih) * K + rk
        vp = ((ib * T + t) * H + ih) * V + rv
        aa = tl.load(a + kp, mask=mk, other=0.0).to(tl.float32)
        bb = tl.load(b + kp, mask=mk, other=0.0).to(tl.float32)
        kk = tl.load(k + kp, mask=mk, other=0.0).to(tl.float32)
        gg = tl.load(g + kp, mask=mk, other=0.0).to(tl.float32)
        vv = tl.load(v + vp, mask=mv, other=0.0).to(tl.float32)
        read = tl.sum(aa[:, None] * state, axis=0)
        state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]



@triton.jit
def _expand_states_kernel(k, v, a, b, g, h, states, T: tl.constexpr, H: tl.constexpr,
                          K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                          BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    ic = tl.program_id(1)
    bh = tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mk, mv = rk < K, rv < V
    hp = h + ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
    state = tl.load(hp, mask=mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        mt = t < T
        kp = ((ib * T + t) * H + ih) * K + rk
        vp = ((ib * T + t) * H + ih) * V + rv
        old = state
        aa = tl.load(a + kp, mask=mt & mk, other=0.0).to(tl.float32)
        bb = tl.load(b + kp, mask=mt & mk, other=0.0).to(tl.float32)
        kk = tl.load(k + kp, mask=mt & mk, other=0.0).to(tl.float32)
        gg = tl.load(g + kp, mask=mt & mk, other=0.0).to(tl.float32)
        vv = tl.load(v + vp, mask=mt & mv, other=0.0).to(tl.float32)
        read = tl.sum(aa[:, None] * old, axis=0)
        state = old * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        sp = states + ((((ib * T + t) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        tl.store(sp, state, mask=mt & mk[:, None] & mv[None, :])


@triton.jit
def _boundary_adjoint_kernel(q, a, b, g, do, boundaries, T: tl.constexpr, H: tl.constexpr,
                             K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                             BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mk, mv = rk < K, rv < V
    ds = tl.zeros((BK, BV), tl.float32)
    scale = K ** -0.5
    for z in range(NC):
        ic = NC - 1 - z
        bp = boundaries + ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        tl.store(bp, ds, mask=mk[:, None] & mv[None, :])
        for jj in tl.static_range(0, 16):
            j = 15 - jj
            t = ic * 16 + j
            mt = t < T
            kp = ((ib * T + t) * H + ih) * K + rk
            vp = ((ib * T + t) * H + ih) * V + rv
            qq = tl.load(q + kp, mask=mt & mk, other=0.0).to(tl.float32) * scale
            aa = tl.load(a + kp, mask=mt & mk, other=0.0).to(tl.float32)
            bb = tl.load(b + kp, mask=mt & mk, other=0.0).to(tl.float32)
            gg = tl.load(g + kp, mask=mt & mk, other=0.0).to(tl.float32)
            dout = tl.load(do + vp, mask=mt & mv, other=0.0).to(tl.float32)
            ds += qq[:, None] * dout[None, :]
            wr = tl.sum(bb[:, None] * ds, axis=0)
            ds = ds * tl.exp(gg)[:, None] + aa[:, None] * wr[None, :]


@triton.jit
def _backward_chunks_kernel(q, k, v, a, b, g, do, states, h, boundaries,
                            pq, pk, pa, pb, pg, dv,
                            T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                            V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                            BV: tl.constexpr):
    iv = tl.program_id(0)
    ic = tl.program_id(1)
    bh = tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mk, mv = rk < K, rv < V
    bp = boundaries + ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
    ds = tl.load(bp, mask=mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
    scale = K ** -0.5
    for jj in tl.static_range(0, 16):
        j = 15 - jj
        t = ic * 16 + j
        mt = t < T
        kp = ((ib * T + t) * H + ih) * K + rk
        vp = ((ib * T + t) * H + ih) * V + rv
        sp = states + ((((ib * T + t) * H + ih) * K + rk[:, None]) * V + rv[None, :])
        st = tl.load(sp, mask=mt & mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
        qq = tl.load(q + kp, mask=mt & mk, other=0.0).to(tl.float32) * scale
        aa = tl.load(a + kp, mask=mt & mk, other=0.0).to(tl.float32)
        bb = tl.load(b + kp, mask=mt & mk, other=0.0).to(tl.float32)
        kk = tl.load(k + kp, mask=mt & mk, other=0.0).to(tl.float32)
        gg = tl.load(g + kp, mask=mt & mk, other=0.0).to(tl.float32)
        vv = tl.load(v + vp, mask=mt & mv, other=0.0).to(tl.float32)
        dout = tl.load(do + vp, mask=mt & mv, other=0.0).to(tl.float32)
        if j > 0:
            pp = states + ((((ib * T + t - 1) * H + ih) * K + rk[:, None]) * V + rv[None, :])
            prev = tl.load(pp, mask=mt & mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
        else:
            cp = h + ((((ib * NC + ic) * H + ih) * K + rk[:, None]) * V + rv[None, :])
            prev = tl.load(cp, mask=mt & mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
        ds += qq[:, None] * dout[None, :]
        read = tl.sum(aa[:, None] * prev, axis=0)
        wr = tl.sum(bb[:, None] * ds, axis=0)
        dqv = tl.sum(st * dout[None, :], axis=1) * scale
        dkv = tl.sum(ds * vv[None, :], axis=1)
        dbv = tl.sum(ds * read[None, :], axis=1)
        dav = tl.sum(prev * wr[None, :], axis=1)
        dgv = tl.sum(ds * (prev * tl.exp(gg)[:, None]), axis=1)
        part = (((iv * (T * tl.num_programs(2)) + (ib * T + t) * H + ih) * K) + rk)
        tl.store(pq + part, dqv, mask=mt & mk)
        tl.store(pk + part, dkv, mask=mt & mk)
        tl.store(pa + part, dav, mask=mt & mk)
        tl.store(pb + part, dbv, mask=mt & mk)
        tl.store(pg + part, dgv, mask=mt & mk)
        tl.store(dv + vp, tl.sum(kk[:, None] * ds, axis=0), mask=mt & mv)
        ds = ds * tl.exp(gg)[:, None] + aa[:, None] * wr[None, :]


@triton.jit
def _reduce_kernel(pq, pk, pa, pb, pg, dq, dk, da, db, dg, N: tl.constexpr,
                   K: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = x < N * K
    r = x % K
    n = x // K
    sq = tl.zeros((BLOCK,), tl.float32)
    sk = tl.zeros((BLOCK,), tl.float32)
    sa = tl.zeros((BLOCK,), tl.float32)
    sb = tl.zeros((BLOCK,), tl.float32)
    sg = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        p = (iv * N + n) * K + r
        sq += tl.load(pq + p, mask=m, other=0.0)
        sk += tl.load(pk + p, mask=m, other=0.0)
        sa += tl.load(pa + p, mask=m, other=0.0)
        sb += tl.load(pb + p, mask=m, other=0.0)
        sg += tl.load(pg + p, mask=m, other=0.0)
    tl.store(dq + x, sq, mask=m)
    tl.store(dk + x, sk, mask=m)
    tl.store(da + x, sa, mask=m)
    tl.store(db + x, sb, mask=m)
    tl.store(dg + x, sg, mask=m)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        B, T, H, K = q.shape
        V, nc = v.shape[-1], triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _FBV)
        out = torch.empty_like(v)
        h = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, h,
                                     T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_FBV,
                                     num_warps=4, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, h)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, a, b, g, h = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K = q.shape
        V, nc = v.shape[-1], triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        power_of_two_v = V == triton.next_power_of_2(V)
        bv = _BBV if power_of_two_v else triton.next_power_of_2(V)
        nv = triton.cdiv(V, bv)
        if not power_of_two_v:
            _checkpoint_kernel[(nv, B * H)](k, v, a, b, g, h, T=T, H=H, K=K,
                                            V=V, NC=nc, BK=bk, BV=bv, num_warps=4)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        boundaries = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](k, v, a, b, g, h, states,
                                               T=T, H=H, K=K, V=V, NC=nc,
                                               BK=bk, BV=bv, num_warps=4)
        _boundary_adjoint_kernel[(nv, B * H)](q, a, b, g, dout, boundaries,
                                              T=T, H=H, K=K, V=V, NC=nc,
                                              BK=bk, BV=bv, num_warps=4)
        n = B * T * H
        parts = [torch.zeros((nv, n, K), device=q.device, dtype=torch.float32) for _ in range(5)]
        dv = torch.empty_like(v)
        _backward_chunks_kernel[(nv, nc, B * H)](q, k, v, a, b, g, dout, states,
                                                  h, boundaries, *parts, dv, T=T, H=H,
                                                  K=K, V=V, NC=nc, BK=bk, BV=bv,
                                                  num_warps=4)
        grads = [torch.empty_like(x) for x in (q, k, a, b, g)]
        _reduce_kernel[(triton.cdiv(n * K, 256),)](*parts, *grads, N=n, K=K,
                                                   NV=nv, BLOCK=256)
        dq, dk, da, db, dg = grads
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
