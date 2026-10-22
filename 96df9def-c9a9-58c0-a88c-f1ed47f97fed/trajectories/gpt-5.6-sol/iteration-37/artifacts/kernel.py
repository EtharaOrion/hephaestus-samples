import torch
import triton
import triton.language as tl

_BT = 16
_FBV = 32
_BBV = 16


@triton.jit
def _load_tk(x, base, ot, ok, T: tl.constexpr, H: tl.constexpr,
             K: tl.constexpr, BK: tl.constexpr):
    return tl.load(x + base + ot[:, None] * H * K + ok[None, :],
                   mask=(ot[:, None] < T) & (ok[None, :] < K), other=0.0).to(tl.float32)


@triton.jit
def _forward_kernel(q, k, v, a, b, g, out, checkpoints,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                    BV: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    rr = tl.arange(0, 16)
    state = tl.zeros((BK, BV), tl.float32)
    scale: tl.constexpr = K ** -0.5
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V

    for ic in range(NC):
        t0 = ic * 16
        ot = t0 + rr
        valid_t = ot < T
        cp = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        tl.store(checkpoints + cp, state,
                 mask=(ok[:, None] < K) & (ov[None, :] < V))

        q0 = _load_tk(q, base_k, ot, ok, T, H, K, BK) * scale
        k0 = _load_tk(k, base_k, ot, ok, T, H, K, BK)
        a0 = _load_tk(a, base_k, ot, ok, T, H, K, BK)
        b0 = _load_tk(b, base_k, ot, ok, T, H, K, BK)
        g0 = _load_tk(g, base_k, ot, ok, T, H, K, BK)
        vv = tl.load(v + base_v + ot[:, None] * H * V + ov[None, :],
                     mask=valid_t[:, None] & (ov[None, :] < V), other=0.0).to(tl.float32)

        gc = tl.cumsum(g0, axis=0)
        gp = gc - g0
        egp = tl.exp(tl.maximum(gp, -60.0))
        eig = tl.exp(tl.minimum(-gc, 60.0))
        causal = rr[:, None] > rr[None, :]
        lab = tl.dot(a0 * egp, tl.trans(b0 * eig), input_precision="ieee")
        lak = tl.dot(a0 * egp, tl.trans(k0 * eig), input_precision="ieee")
        lab = tl.where(causal & valid_t[:, None] & valid_t[None, :], lab, 0.0)
        lak = tl.where(causal & valid_t[:, None] & valid_t[None, :], lak, 0.0)

        eye = tl.where(rr[:, None] == rr[None, :], 1.0, 0.0)
        inv = eye
        power = lab
        for _ in tl.static_range(0, 4):
            inv += tl.dot(inv, power, input_precision="ieee")
            power = tl.dot(power, power, input_precision="ieee")

        rhs = tl.dot(a0 * egp, state, input_precision="ieee")
        rhs += tl.dot(lak, vv, input_precision="ieee")
        read = tl.dot(inv, rhs, input_precision="ieee")

        aqk = tl.dot(q0 * tl.exp(tl.maximum(gc, -60.0)),
                     tl.trans(k0 * eig), input_precision="ieee")
        aqb = tl.dot(q0 * tl.exp(tl.maximum(gc, -60.0)),
                     tl.trans(b0 * eig), input_precision="ieee")
        incl = rr[:, None] >= rr[None, :]
        aqk = tl.where(incl & valid_t[:, None] & valid_t[None, :], aqk, 0.0)
        aqb = tl.where(incl & valid_t[:, None] & valid_t[None, :], aqb, 0.0)
        oo = tl.dot(q0 * tl.exp(tl.maximum(gc, -60.0)), state,
                    input_precision="ieee")
        oo += tl.dot(aqk, vv, input_precision="ieee")
        oo += tl.dot(aqb, read, input_precision="ieee")
        tl.store(out + base_v + ot[:, None] * H * V + ov[None, :], oo,
                 mask=valid_t[:, None] & (ov[None, :] < V))

        last = tl.minimum(15, T - 1 - t0)
        ge = tl.sum(tl.where(rr[:, None] <= last, g0, 0.0), axis=0)
        w = tl.exp(tl.maximum(ge[None, :] - gc, -60.0))
        w = tl.where(valid_t[:, None], w, 0.0)
        state = state * tl.exp(tl.maximum(ge, -60.0))[:, None]
        state += tl.dot(tl.trans(k0 * w), vv, input_precision="ieee")
        state += tl.dot(tl.trans(b0 * w), read, input_precision="ieee")


@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                          V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                          BV: tl.constexpr):
    iv = tl.program_id(0)
    ic = tl.program_id(1)
    bh = tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V
    cp = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
    state = tl.load(checkpoints + cp,
                    mask=(ok[:, None] < K) & (ov[None, :] < V), other=0.0)
    for j in tl.static_range(0, 16):
        t = ic * 16 + j
        valid = t < T
        sp = (((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        tl.store(states + sp, state,
                 mask=valid & (ok[:, None] < K) & (ov[None, :] < V))
        xoff = base_k + t * H * K + ok
        voff = base_v + t * H * V + ov
        aa = tl.load(a + xoff, mask=valid & (ok < K), other=0.0).to(tl.float32)
        bb = tl.load(b + xoff, mask=valid & (ok < K), other=0.0).to(tl.float32)
        kk = tl.load(k + xoff, mask=valid & (ok < K), other=0.0).to(tl.float32)
        gg = tl.load(g + xoff, mask=valid & (ok < K), other=0.0).to(tl.float32)
        vv = tl.load(v + voff, mask=valid & (ov < V), other=0.0).to(tl.float32)
        read = tl.sum(state * aa[:, None], axis=0)
        state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]


@triton.jit
def _boundary_adjoint_kernel(q, a, b, g, dout, states, boundaries,
                             T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                             V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                             BV: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    ds = tl.zeros((BK, BV), tl.float32)
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V
    scale: tl.constexpr = K ** -0.5
    rr = tl.arange(0, 16)
    for ii in range(NC):
        ic = NC - 1 - ii
        bp = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        tl.store(boundaries + bp, ds,
                 mask=(ok[:, None] < K) & (ov[None, :] < V))
        ot = ic * 16 + 15 - rr
        valid_t = ot < T
        q0 = _load_tk(q, base_k, ot, ok, T, H, K, BK) * scale
        ar = _load_tk(b, base_k, ot, ok, T, H, K, BK)
        bw = _load_tk(a, base_k, ot, ok, T, H, K, BK)
        g0 = _load_tk(g, base_k, ot, ok, T, H, K, BK)
        do = tl.load(dout + base_v + ot[:, None] * H * V + ov[None, :],
                     mask=valid_t[:, None] & (ov[None, :] < V), other=0.0).to(tl.float32)
        kp = q0 * tl.exp(g0)
        kp += bw * tl.sum(ar * q0, axis=1)[:, None]
        gc = tl.cumsum(g0, axis=0)
        gp = gc - g0
        egp = tl.exp(tl.maximum(gp, -60.0))
        eig = tl.exp(tl.minimum(-gc, 60.0))
        causal = rr[:, None] > rr[None, :]
        lab = tl.dot(ar * egp, tl.trans(bw * eig), input_precision="ieee")
        lak = tl.dot(ar * egp, tl.trans(kp * eig), input_precision="ieee")
        lab = tl.where(causal & valid_t[:, None] & valid_t[None, :], lab, 0.0)
        lak = tl.where(causal & valid_t[:, None] & valid_t[None, :], lak, 0.0)
        eye = tl.where(rr[:, None] == rr[None, :], 1.0, 0.0)
        inv = eye
        power = lab
        for _ in tl.static_range(0, 4):
            inv += tl.dot(inv, power, input_precision="ieee")
            power = tl.dot(power, power, input_precision="ieee")
        rhs = tl.dot(ar * egp, ds, input_precision="ieee")
        rhs += tl.dot(lak, do, input_precision="ieee")
        read = tl.dot(inv, rhs, input_precision="ieee")
        ge = tl.sum(g0, axis=0)
        w = tl.exp(tl.maximum(ge[None, :] - gc, -60.0))
        w = tl.where(valid_t[:, None], w, 0.0)
        ds = ds * tl.exp(tl.maximum(ge, -60.0))[:, None]
        ds += tl.dot(tl.trans(kp * w), do, input_precision="ieee")
        ds += tl.dot(tl.trans(bw * w), read, input_precision="ieee")


@triton.jit
def _backward_chunks_kernel(q, k, v, a, b, g, dout, states, boundaries,
                            pq, pk, pa, pb, pg, dv,
                            T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                            V: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr,
                            BK: tl.constexpr, BV: tl.constexpr):
    iv = tl.program_id(0)
    ic = tl.program_id(1)
    bh = tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    base_k = (ib * T * H + ih) * K
    base_v = (ib * T * H + ih) * V
    bp = (((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :]
    ds = tl.load(boundaries + bp,
                 mask=(ok[:, None] < K) & (ov[None, :] < V), other=0.0)
    scale: tl.constexpr = K ** -0.5
    for jj in tl.static_range(0, 16):
        t = ic * 16 + 15 - jj
        valid = t < T
        xoff = base_k + t * H * K + ok
        voff = base_v + t * H * V + ov
        sp = (((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        ss = tl.load(states + sp,
                     mask=valid & (ok[:, None] < K) & (ov[None, :] < V), other=0.0)
        qq = tl.load(q + xoff, mask=valid & (ok < K), other=0.0).to(tl.float32) * scale
        kk = tl.load(k + xoff, mask=valid & (ok < K), other=0.0).to(tl.float32)
        aa = tl.load(a + xoff, mask=valid & (ok < K), other=0.0).to(tl.float32)
        bb = tl.load(b + xoff, mask=valid & (ok < K), other=0.0).to(tl.float32)
        gg = tl.load(g + xoff, mask=valid & (ok < K), other=0.0).to(tl.float32)
        vv = tl.load(v + voff, mask=valid & (ov < V), other=0.0).to(tl.float32)
        do = tl.load(dout + voff, mask=valid & (ov < V), other=0.0).to(tl.float32)
        cur = ds + qq[:, None] * do[None, :]
        read = tl.sum(ss * aa[:, None], axis=0)
        st = ss * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        dr = tl.sum(cur * bb[:, None], axis=0)
        pbase = iv * (T * H * K * tl.num_programs(2) // H) + ((ib * T + t) * H + ih) * K + ok
        maskk = valid & (ok < K)
        tl.store(pq + pbase, tl.sum(st * do[None, :], axis=1) * scale, mask=maskk)
        tl.store(pk + pbase, tl.sum(cur * vv[None, :], axis=1), mask=maskk)
        tl.store(pb + pbase, tl.sum(cur * read[None, :], axis=1), mask=maskk)
        tl.store(pa + pbase, tl.sum(ss * dr[None, :], axis=1), mask=maskk)
        tl.store(pg + pbase, tl.sum(cur * (ss * tl.exp(gg)[:, None]), axis=1), mask=maskk)
        dvi = tl.sum(cur * kk[:, None], axis=0)
        tl.store(dv + voff, dvi, mask=valid & (ov < V))
        prev = cur * tl.exp(gg)[:, None] + aa[:, None] * dr[None, :]
        ds = tl.where(valid, prev, ds)


@triton.jit
def _reduce_partials_kernel(pq, pk, pa, pb, pg, dq, dk, da, db, dg,
                            N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    sq = tl.zeros((BLOCK,), tl.float32)
    sk = tl.zeros((BLOCK,), tl.float32)
    sa = tl.zeros((BLOCK,), tl.float32)
    sb = tl.zeros((BLOCK,), tl.float32)
    sg = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        off = iv * N + idx
        sq += tl.load(pq + off, mask=mask, other=0.0)
        sk += tl.load(pk + off, mask=mask, other=0.0)
        sa += tl.load(pa + off, mask=mask, other=0.0)
        sb += tl.load(pb + off, mask=mask, other=0.0)
        sg += tl.load(pg + off, mask=mask, other=0.0)
    tl.store(dq + idx, sq, mask=mask)
    tl.store(dk + idx, sk, mask=mask)
    tl.store(da + idx, sa, mask=mask)
    tl.store(db + idx, sb, mask=mask)
    tl.store(dg + idx, sg, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        nv = triton.cdiv(V, _FBV)
        bk = max(16, triton.next_power_of_2(K))
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, out, checkpoints,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_FBV,
            num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        nv = triton.cdiv(V, _BBV)
        bk = max(16, triton.next_power_of_2(K))
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        boundaries = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](
            k, v, alpha, beta, gk, checkpoints, states,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BBV,
            num_warps=4, num_stages=2)
        _boundary_adjoint_kernel[(nv, B * H)](
            q, alpha, beta, gk, dout, states, boundaries,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BBV,
            num_warps=4, num_stages=2)
        n = B * T * H * K
        parts = torch.empty((5, nv, n), device=q.device, dtype=torch.float32)
        dq, dk, da, db = (torch.empty_like(q), torch.empty_like(k),
                          torch.empty_like(alpha), torch.empty_like(beta))
        dg = torch.empty_like(gk)
        dv = torch.empty_like(v)
        _backward_chunks_kernel[(nv, nc, B * H)](
            q, k, v, alpha, beta, gk, dout, states, boundaries,
            parts[0], parts[1], parts[2], parts[3], parts[4], dv,
            T=T, H=H, K=K, V=V, NC=nc, NV=nv, BK=bk, BV=_BBV,
            num_warps=4, num_stages=2)
        _reduce_partials_kernel[(triton.cdiv(n, 256),)](
            parts[0], parts[1], parts[2], parts[3], parts[4], dq, dk, da, db, dg,
            N=n, NV=nv, BLOCK=256, num_warps=4)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
