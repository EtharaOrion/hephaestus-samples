
import torch
import triton
import triton.language as tl

_BT = 8
_FBV = 32
_BBV = 16


@triton.jit
def _chunk_data(q, k, a, b, g, base, H: tl.constexpr, K: tl.constexpr,
                BK: tl.constexpr, T: tl.constexpr, CT: tl.constexpr):
    tt = tl.arange(0, 16)
    kk = tl.arange(0, BK)
    valid_t = (base + tt < T) & (tt < CT)
    offs = ((base + tt)[:, None] * H * K + kk[None, :])
    mask = valid_t[:, None] & (kk[None, :] < K)
    gg = tl.load(g + offs, mask=mask, other=0.0).to(tl.float32)
    cc = tl.cumsum(gg, axis=0)
    qq = tl.load(q + offs, mask=mask, other=0.0).to(tl.float32) * (K ** -0.5)
    kkx = tl.load(k + offs, mask=mask, other=0.0).to(tl.float32)
    aa = tl.load(a + offs, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(b + offs, mask=mask, other=0.0).to(tl.float32)
    i = tt[:, None]
    j = tt[None, :]
    causal = j <= i
    strict = j < i
    # Invalid triangular entries use exponent zero before masking the result.
    dq_arg = tl.where(causal[:, :, None], cc[:, None, :] - cc[None, :, :], 0.0)
    cprev = cc - gg
    da_arg = tl.where(strict[:, :, None], cprev[:, None, :] - cc[None, :, :], 0.0)
    d_q = tl.exp(dq_arg)
    d_a = tl.exp(da_arg)
    aqk = tl.sum(qq[:, None, :] * kkx[None, :, :] * d_q, axis=2)
    aqb = tl.sum(qq[:, None, :] * bb[None, :, :] * d_q, axis=2)
    aak = tl.sum(aa[:, None, :] * kkx[None, :, :] * d_a, axis=2)
    lab = tl.sum(aa[:, None, :] * bb[None, :, :] * d_a, axis=2)
    pair_mask = valid_t[:, None] & valid_t[None, :]
    aqk = tl.where(pair_mask & causal, aqk, 0.0)
    aqb = tl.where(pair_mask & causal, aqb, 0.0)
    aak = tl.where(pair_mask & strict, aak, 0.0)
    lab = tl.where(pair_mask & strict, lab, 0.0)
    qg = qq * tl.exp(cc)
    wg = aa * tl.exp(cprev)
    end = tl.sum(gg, axis=0)
    kg = kkx * tl.exp(end[None, :] - cc)
    bg = bb * tl.exp(end[None, :] - cc)
    return valid_t, qg, wg, kg, bg, aqk, aqb, aak, lab, end


@triton.jit
def _forward_kernel(q, k, v, a, b, g, out, bounds,
                    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
                    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                    BV: tl.constexpr, NC: tl.constexpr, CT: tl.constexpr):
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    vk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    mkv = (vk[:, None] < K) & (vv[None, :] < V)
    state = tl.zeros((BK, BV), tl.float32)
    q0 = q + (ib * T * H + ih) * K
    k0 = k + (ib * T * H + ih) * K
    a0 = a + (ib * T * H + ih) * K
    b0 = b + (ib * T * H + ih) * K
    g0 = g + (ib * T * H + ih) * K
    v0 = v + (ib * T * H + ih) * V
    o0 = out + (ib * T * H + ih) * V
    for ic in range(NC):
        base = ic * CT
        hp = bounds + (((ib * NC + ic) * H + ih) * K * V +
                       vk[:, None] * V + vv[None, :])
        tl.store(hp, state, mask=mkv)
        valid, qg, wg, kg, bg, aqk, aqb, aak, lab, end = _chunk_data(
            q0, k0, a0, b0, g0, base, H, K, BK, T, CT)
        tt = tl.arange(0, 16)
        vp = v0 + (base + tt)[:, None] * H * V + vv[None, :]
        mtv = valid[:, None] & (vv[None, :] < V)
        vx = tl.load(vp, mask=mtv, other=0.0).to(tl.float32)
        rhs = tl.dot(aak, vx, input_precision="ieee")
        rhs += tl.dot(wg, state, input_precision="ieee")
        rr = rhs
        rows = tl.arange(0, 16)[:, None]
        cols = tl.arange(0, BV)[None, :]
        for i in tl.static_range(0, CT):
            li = tl.sum(tl.where(rows == i, lab, 0.0), axis=0)
            ri = tl.sum(li[:, None] * rr, axis=0)
            rr = tl.where((rows == i) & (cols < BV), rhs + ri[None, :], rr)
        ox = tl.dot(aqk, vx, input_precision="ieee")
        ox += tl.dot(aqb, rr, input_precision="ieee")
        ox += tl.dot(qg, state, input_precision="ieee")
        op = o0 + (base + tt)[:, None] * H * V + vv[None, :]
        tl.store(op, ox, mask=mtv)
        state *= tl.exp(end)[:, None]
        state += tl.dot(tl.trans(kg), vx, input_precision="ieee")
        state += tl.dot(tl.trans(bg), rr, input_precision="ieee")


@triton.jit
def _expand_states_kernel(k, v, a, b, g, bounds, states,
                          B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
                          K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                          BV: tl.constexpr, NC: tl.constexpr, CT: tl.constexpr):
    iv, ic, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    mkv = (kk[:, None] < K) & (vv[None, :] < V)
    hp = bounds + (((ib * NC + ic) * H + ih) * K * V +
                   kk[:, None] * V + vv[None, :])
    state = tl.load(hp, mask=mkv, other=0.0).to(tl.float32)
    for j in tl.static_range(0, CT):
        t = ic * CT + j
        valid = t < T
        sp = states + (((ib * T + t) * H + ih) * K * V +
                       kk[:, None] * V + vv[None, :])
        tl.store(sp, state, mask=mkv & valid)
        kp = k + ((ib * T + t) * H + ih) * K + kk
        ap = a + ((ib * T + t) * H + ih) * K + kk
        bp = b + ((ib * T + t) * H + ih) * K + kk
        gp = g + ((ib * T + t) * H + ih) * K + kk
        vp = v + ((ib * T + t) * H + ih) * V + vv
        kx = tl.load(kp, mask=(kk < K) & valid, other=0.0).to(tl.float32)
        ax = tl.load(ap, mask=(kk < K) & valid, other=0.0).to(tl.float32)
        bx = tl.load(bp, mask=(kk < K) & valid, other=0.0).to(tl.float32)
        gx = tl.load(gp, mask=(kk < K) & valid, other=0.0).to(tl.float32)
        vx = tl.load(vp, mask=(vv < V) & valid, other=0.0).to(tl.float32)
        read = tl.sum(state * ax[:, None], axis=0)
        new_state = state * tl.exp(gx)[:, None] + kx[:, None] * vx[None, :] + bx[:, None] * read[None, :]
        state = tl.where(valid, new_state, state)


@triton.jit
def _boundary_adjoint_kernel(q, a, b, g, dout, states, dbounds,
                             B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
                             K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                             BV: tl.constexpr, NC: tl.constexpr, CT: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib = bh // H
    ih = bh - ib * H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    mkv = (kk[:, None] < K) & (vv[None, :] < V)
    ds = tl.zeros((BK, BV), tl.float32)
    for cr in range(NC):
        ic = NC - 1 - cr
        dp = dbounds + (((ib * NC + ic) * H + ih) * K * V +
                        kk[:, None] * V + vv[None, :])
        tl.store(dp, ds, mask=mkv)
        for jr in tl.static_range(0, CT):
            j = CT - 1 - jr
            t = ic * CT + j
            valid = t < T
            basek = ((ib * T + t) * H + ih) * K
            basev = ((ib * T + t) * H + ih) * V
            qx = tl.load(q + basek + kk, mask=(kk < K) & valid, other=0.0).to(tl.float32) * (K ** -0.5)
            ax = tl.load(a + basek + kk, mask=(kk < K) & valid, other=0.0).to(tl.float32)
            bx = tl.load(b + basek + kk, mask=(kk < K) & valid, other=0.0).to(tl.float32)
            gx = tl.load(g + basek + kk, mask=(kk < K) & valid, other=0.0).to(tl.float32)
            do = tl.load(dout + basev + vv, mask=(vv < V) & valid, other=0.0).to(tl.float32)
            total = ds + qx[:, None] * do[None, :]
            dr = tl.sum(total * bx[:, None], axis=0)
            prev = total * tl.exp(gx)[:, None] + ax[:, None] * dr[None, :]
            ds = tl.where(valid, prev, ds)


@triton.jit
def _backward_chunks_kernel(q, k, v, a, b, g, dout, states, dbounds,
                            pq, pk, pa, pb, pg, dv,
                            B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
                            K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                            BV: tl.constexpr, NC: tl.constexpr, CT: tl.constexpr,
                            NV: tl.constexpr):
    iv, ic, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib = bh // H
    ih = bh - ib * H
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    mkv = (kk[:, None] < K) & (vv[None, :] < V)
    dp = dbounds + (((ib * NC + ic) * H + ih) * K * V +
                    kk[:, None] * V + vv[None, :])
    ds = tl.load(dp, mask=mkv, other=0.0).to(tl.float32)
    for jr in tl.static_range(0, CT):
        j = CT - 1 - jr
        t = ic * CT + j
        valid = t < T
        basek = ((ib * T + t) * H + ih) * K
        basev = ((ib * T + t) * H + ih) * V
        qx = tl.load(q + basek + kk, mask=(kk < K) & valid, other=0.0).to(tl.float32) * (K ** -0.5)
        kx = tl.load(k + basek + kk, mask=(kk < K) & valid, other=0.0).to(tl.float32)
        ax = tl.load(a + basek + kk, mask=(kk < K) & valid, other=0.0).to(tl.float32)
        bx = tl.load(b + basek + kk, mask=(kk < K) & valid, other=0.0).to(tl.float32)
        gx = tl.load(g + basek + kk, mask=(kk < K) & valid, other=0.0).to(tl.float32)
        vx = tl.load(v + basev + vv, mask=(vv < V) & valid, other=0.0).to(tl.float32)
        do = tl.load(dout + basev + vv, mask=(vv < V) & valid, other=0.0).to(tl.float32)
        sp = states + (((ib * T + t) * H + ih) * K * V +
                       kk[:, None] * V + vv[None, :])
        sold = tl.load(sp, mask=mkv & valid, other=0.0).to(tl.float32)
        decay = tl.exp(gx)
        read = tl.sum(sold * ax[:, None], axis=0)
        snew = sold * decay[:, None] + kx[:, None] * vx[None, :] + bx[:, None] * read[None, :]
        total = ds + qx[:, None] * do[None, :]
        dr = tl.sum(total * bx[:, None], axis=0)
        dqx = tl.sum(snew * do[None, :], axis=1) * (K ** -0.5)
        dkx = tl.sum(total * vx[None, :], axis=1)
        dax = tl.sum(sold * dr[None, :], axis=1)
        dbx = tl.sum(total * read[None, :], axis=1)
        dgx = tl.sum(total * sold * decay[:, None], axis=1)
        pp = (((ib * T + t) * H + ih) * NV + iv) * K + kk
        pm = (kk < K) & valid
        tl.store(pq + pp, dqx, mask=pm)
        tl.store(pk + pp, dkx, mask=pm)
        tl.store(pa + pp, dax, mask=pm)
        tl.store(pb + pp, dbx, mask=pm)
        tl.store(pg + pp, dgx, mask=pm)
        dvx = tl.sum(total * kx[:, None], axis=0)
        tl.store(dv + basev + vv, dvx, mask=(vv < V) & valid)
        prev = total * decay[:, None] + ax[:, None] * dr[None, :]
        ds = tl.where(valid, prev, ds)


@triton.jit
def _reduce_partials_kernel(pq, pk, pa, pb, pg, dq, dk, da, db, dg,
                            N: tl.constexpr, K: tl.constexpr, NV: tl.constexpr,
                            BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    kk = offs % K
    row = offs // K
    base = row * NV * K + kk
    sq = tl.zeros((BLOCK,), tl.float32)
    sk = tl.zeros((BLOCK,), tl.float32)
    sa = tl.zeros((BLOCK,), tl.float32)
    sb = tl.zeros((BLOCK,), tl.float32)
    sg = tl.zeros((BLOCK,), tl.float32)
    for iv in range(NV):
        p = base + iv * K
        sq += tl.load(pq + p, mask=mask, other=0.0)
        sk += tl.load(pk + p, mask=mask, other=0.0)
        sa += tl.load(pa + p, mask=mask, other=0.0)
        sb += tl.load(pb + p, mask=mask, other=0.0)
        sg += tl.load(pg + p, mask=mask, other=0.0)
    tl.store(dq + offs, sq, mask=mask)
    tl.store(dk + offs, sk, mask=mask)
    tl.store(da + offs, sa, mask=mask)
    tl.store(db + offs, sb, mask=mask)
    tl.store(dg + offs, sg, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        ct = 16 if K == bk and T % 16 == 0 else _BT
        nc = triton.cdiv(T, ct)
        nv = triton.cdiv(V, _FBV)
        out = torch.empty_like(v)
        bounds = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _forward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, out, bounds,
            B=B, T=T, H=H, K=K, V=V, BK=bk, BV=_FBV, NC=nc, CT=ct,
            num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, bounds)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, bounds = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        bk = max(16, triton.next_power_of_2(K))
        ct = 16 if K == bk and T % 16 == 0 else _BT
        nc = triton.cdiv(T, ct)
        bv = _BBV if V % _BBV == 0 else max(16, triton.next_power_of_2(V))
        nv = triton.cdiv(V, bv)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        dbounds = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](
            k, v, alpha, beta, gk, bounds, states,
            B=B, T=T, H=H, K=K, V=V, BK=bk, BV=bv, NC=nc, CT=ct,
            num_warps=4, num_stages=2)
        _boundary_adjoint_kernel[(nv, B * H)](
            q, alpha, beta, gk, dout, states, dbounds,
            B=B, T=T, H=H, K=K, V=V, BK=bk, BV=bv, NC=nc, CT=ct,
            num_warps=4, num_stages=2)
        part_shape = (B, T, H, nv, K)
        pq = torch.zeros(part_shape, device=q.device, dtype=torch.float32)
        pk = torch.zeros_like(pq)
        pa = torch.zeros_like(pq)
        pb = torch.zeros_like(pq)
        pg = torch.zeros_like(pq)
        dv = torch.empty_like(v)
        _backward_chunks_kernel[(nv, nc, B * H)](
            q, k, v, alpha, beta, gk, dout, states, dbounds,
            pq, pk, pa, pb, pg, dv,
            B=B, T=T, H=H, K=K, V=V, BK=bk, BV=bv, NC=nc, CT=ct, NV=nv,
            num_warps=4, num_stages=2)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        da = torch.empty_like(alpha)
        db = torch.empty_like(beta)
        dg = torch.empty_like(gk)
        n = B * T * H * K
        _reduce_partials_kernel[(triton.cdiv(n, 256),)](
            pq, pk, pa, pb, pg, dq, dk, da, db, dg,
            N=n, K=K, NV=nv, BLOCK=256, num_warps=4)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
