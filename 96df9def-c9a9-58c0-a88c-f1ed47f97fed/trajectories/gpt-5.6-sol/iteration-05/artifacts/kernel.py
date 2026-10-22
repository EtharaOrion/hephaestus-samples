"""Chunk-parallel DPLR generalized delta rule."""

import torch
import triton
import triton.language as tl

CHUNK = 16
BT = tl.constexpr(16)
BV = 16


@triton.jit
def _bounded_exp(x):
    return tl.exp(tl.maximum(-80.0, tl.minimum(x, 0.0)))


@triton.jit
def _chunk_forward_kernel(
    q, k, v, a, b, g, out, checkpoints,
    T, NC, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV_: tl.constexpr, SCALE: tl.constexpr,
):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    pk = tl.arange(0, BK)
    pv = iv * BV_ + tl.arange(0, BV_)
    mk = pk < K
    mv = pv < V
    state = tl.zeros((BK, BV_), tl.float32)
    ti = tl.arange(0, BT)
    lower = ti[:, None] > ti[None, :]
    causal = ti[:, None] >= ti[None, :]

    for ic in range(NC):
        start = ic * BT
        valid_t = start + ti < T
        cp = checkpoints + ((((ib * NC + ic) * H + ih) * K + pk[:, None]) * V + pv[None, :])
        tl.store(cp, state, mask=mk[:, None] & mv[None, :])

        xk = start + ti[:, None]
        base_k = ((ib * T + xk) * H + ih) * K + pk[None, :]
        mtk = valid_t[:, None] & mk[None, :]
        qq = tl.load(q + base_k, mask=mtk, other=0.0).to(tl.float32) * SCALE
        kk = tl.load(k + base_k, mask=mtk, other=0.0).to(tl.float32)
        aa = tl.load(a + base_k, mask=mtk, other=0.0).to(tl.float32)
        bb = tl.load(b + base_k, mask=mtk, other=0.0).to(tl.float32)
        gg = tl.load(g + base_k, mask=mtk, other=0.0).to(tl.float32)
        G = tl.cumsum(gg, axis=0)

        d_now = _bounded_exp(G[:, None, :] - G[None, :, :])
        d_old = _bounded_exp(G[:, None, :] - gg[:, None, :] - G[None, :, :])
        qd = qq[:, None, :] * d_now
        ad = aa[:, None, :] * d_old
        Aqk = tl.sum(qd * kk[None, :, :], axis=2)
        Aqb = tl.sum(qd * bb[None, :, :], axis=2)
        Aak = tl.sum(ad * kk[None, :, :], axis=2)
        Lab = tl.sum(ad * bb[None, :, :], axis=2)
        Aqk = tl.where(causal & valid_t[:, None] & valid_t[None, :], Aqk, 0.0)
        Aqb = tl.where(causal & valid_t[:, None] & valid_t[None, :], Aqb, 0.0)
        Aak = tl.where(lower & valid_t[:, None] & valid_t[None, :], Aak, 0.0)
        Lab = tl.where(lower & valid_t[:, None] & valid_t[None, :], Lab, 0.0)

        for row in tl.static_range(1, BT):
            pick = ti == row
            lr = tl.sum(tl.where(pick[:, None], Lab, 0.0), axis=0)
            lr += tl.sum(lr[:, None] * Lab, axis=0) * (ti < row)
            Lab = tl.where(pick[:, None], lr[None, :], Lab)
        Linv = Lab + (ti[:, None] == ti[None, :])

        base_v = ((ib * T + start + ti[:, None]) * H + ih) * V + pv[None, :]
        mtv = valid_t[:, None] & mv[None, :]
        vv = tl.load(v + base_v, mask=mtv, other=0.0).to(tl.float32)
        u = tl.dot(Aak, vv, input_precision="ieee")
        u = tl.dot(Linv, u, input_precision="ieee")
        ag = _bounded_exp(G - gg) * aa
        w = tl.dot(Linv, ag, input_precision="ieee")
        vnew = u + tl.dot(w, state, input_precision="ieee")
        oo = tl.dot(Aqk, vv, input_precision="ieee")
        oo += tl.dot(Aqb, vnew, input_precision="ieee")
        oo += tl.dot(qq * _bounded_exp(G), state, input_precision="ieee")
        tl.store(out + base_v, oo, mask=mtv)

        last = tl.minimum(T - start, BT) - 1
        Glast = tl.sum(tl.where(ti[:, None] == last, G, 0.0), axis=0)
        tail = _bounded_exp(Glast[None, :] - G)
        state *= _bounded_exp(Glast)[:, None]
        state += tl.dot(tl.trans(kk * tail), vv, input_precision="ieee")
        state += tl.dot(tl.trans(bb * tail), vnew, input_precision="ieee")


@triton.jit
def _expand_states_kernel(
    k, v, a, b, g, checkpoints, states,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NC: tl.constexpr, BK: tl.constexpr, BV_: tl.constexpr,
):
    ic = tl.program_id(0)
    iv = tl.program_id(1)
    ibh = tl.program_id(2)
    ib = ibh // H
    ih = ibh - ib * H
    pk = tl.arange(0, BK)
    pv = iv * BV_ + tl.arange(0, BV_)
    mk = pk < K
    mv = pv < V
    cp = checkpoints + ((((ib * NC + ic) * H + ih) * K + pk[:, None]) * V + pv[None, :])
    state = tl.load(cp, mask=mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
    for j in tl.static_range(BT):
        t = ic * BT + j
        valid = t < T
        p_k = ((ib * T + t) * H + ih) * K + pk
        p_v = ((ib * T + t) * H + ih) * V + pv
        kk = tl.load(k + p_k, mask=valid & mk, other=0.0).to(tl.float32)
        aa = tl.load(a + p_k, mask=valid & mk, other=0.0).to(tl.float32)
        bb = tl.load(b + p_k, mask=valid & mk, other=0.0).to(tl.float32)
        gg = tl.load(g + p_k, mask=valid & mk, other=0.0).to(tl.float32)
        vv = tl.load(v + p_v, mask=valid & mv, other=0.0).to(tl.float32)
        read = tl.sum(state * aa[:, None], axis=0)
        state = state * _bounded_exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        ps = states + ((((ib * T + t) * H + ih) * K + pk[:, None]) * V + pv[None, :])
        tl.store(ps, state, mask=valid & mk[:, None] & mv[None, :])


@triton.jit
def _chunk_backward_kernel(
    q, k, v, a, b, g, do, checkpoints, states,
    dq_part, dk_part, da_part, db_part, dg_part, dv,
    T, NC, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NV: tl.constexpr, BK: tl.constexpr, BV_: tl.constexpr,
    BSTRIDE: tl.constexpr, SCALE: tl.constexpr,
):
    iv = tl.program_id(0)
    ibh = tl.program_id(1)
    ib = ibh // H
    ih = ibh - ib * H
    pk = tl.arange(0, BK)
    pv = iv * BV_ + tl.arange(0, BV_)
    mk = pk < K
    mv = pv < V
    ds = tl.zeros((BK, BV_), tl.float32)

    for rc in range(NC):
        ic = NC - 1 - rc
        for rj in tl.static_range(BT):
            j = BT - 1 - rj
            t = ic * BT + j
            valid = t < T
            p_k = ((ib * T + t) * H + ih) * K + pk
            p_v = ((ib * T + t) * H + ih) * V + pv
            qq = tl.load(q + p_k, mask=valid & mk, other=0.0).to(tl.float32)
            kk = tl.load(k + p_k, mask=valid & mk, other=0.0).to(tl.float32)
            aa = tl.load(a + p_k, mask=valid & mk, other=0.0).to(tl.float32)
            bb = tl.load(b + p_k, mask=valid & mk, other=0.0).to(tl.float32)
            gg = tl.load(g + p_k, mask=valid & mk, other=0.0).to(tl.float32)
            vv = tl.load(v + p_v, mask=valid & mv, other=0.0).to(tl.float32)
            d_o = tl.load(do + p_v, mask=valid & mv, other=0.0).to(tl.float32)
            ps = states + ((((ib * T + t) * H + ih) * K + pk[:, None]) * V + pv[None, :])
            st = tl.load(ps, mask=valid & mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
            if j == 0:
                pp = checkpoints + ((((ib * NC + ic) * H + ih) * K + pk[:, None]) * V + pv[None, :])
            else:
                pp = states + ((((ib * T + t - 1) * H + ih) * K + pk[:, None]) * V + pv[None, :])
            prev = tl.load(pp, mask=valid & mk[:, None] & mv[None, :], other=0.0).to(tl.float32)

            dqv = tl.sum(st * d_o[None, :], axis=1) * SCALE
            ds += qq[:, None] * (d_o[None, :] * SCALE)
            read = tl.sum(prev * aa[:, None], axis=0)
            dkv = tl.sum(ds * vv[None, :], axis=1)
            dbv = tl.sum(ds * read[None, :], axis=1)
            drv = tl.sum(ds * bb[:, None], axis=0)
            dav = tl.sum(prev * drv[None, :], axis=1)
            decay = _bounded_exp(gg)
            dgv = tl.sum(ds * (prev * decay[:, None]), axis=1)
            dvv = tl.sum(ds * kk[:, None], axis=0)

            part_base = iv * BSTRIDE + ((ib * T + t) * H + ih) * K + pk
            pmask = valid & mk
            tl.store(dq_part + part_base, dqv, mask=pmask)
            tl.store(dk_part + part_base, dkv, mask=pmask)
            tl.store(da_part + part_base, dav, mask=pmask)
            tl.store(db_part + part_base, dbv, mask=pmask)
            tl.store(dg_part + part_base, dgv, mask=pmask)
            tl.store(dv + p_v, dvv, mask=valid & mv)
            ds = ds * decay[:, None] + aa[:, None] * drv[None, :]


@triton.jit
def _reduce_partials_kernel(
    dq_part, dk_part, da_part, db_part, dg_part,
    dq, dk, da, db, dg, N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    xq = tl.zeros((BLOCK,), tl.float32)
    xk = tl.zeros((BLOCK,), tl.float32)
    xa = tl.zeros((BLOCK,), tl.float32)
    xb = tl.zeros((BLOCK,), tl.float32)
    xg = tl.zeros((BLOCK,), tl.float32)
    for iv in tl.static_range(NV):
        off = iv * N + offs
        xq += tl.load(dq_part + off, mask=mask, other=0.0)
        xk += tl.load(dk_part + off, mask=mask, other=0.0)
        xa += tl.load(da_part + off, mask=mask, other=0.0)
        xb += tl.load(db_part + off, mask=mask, other=0.0)
        xg += tl.load(dg_part + off, mask=mask, other=0.0)
    tl.store(dq + offs, xq, mask=mask)
    tl.store(dk + offs, xk, mask=mask)
    tl.store(da + offs, xa, mask=mask)
    tl.store(db + offs, xb, mask=mask)
    tl.store(dg + offs, xg, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, CHUNK)
        nv = triton.cdiv(V, BV)
        bk = max(16, triton.next_power_of_2(K))
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _chunk_forward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, out, checkpoints,
            T, H=H, K=K, V=V, NC=nc, BK=bk, BV_=BV,
            SCALE=K ** -0.5, num_warps=16, num_stages=2,
        )
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc = triton.cdiv(T, CHUNK)
        nv = triton.cdiv(V, BV)
        bk = max(16, triton.next_power_of_2(K))
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nc, nv, B * H)](
            k, v, alpha, beta, gk, checkpoints, states,
            T, H=H, K=K, V=V, NC=nc, BK=bk, BV_=BV,
            num_warps=4, num_stages=2,
        )
        shape = (nv, B, T, H, K)
        dq_p = torch.empty(shape, device=q.device, dtype=torch.float32)
        dk_p = torch.empty(shape, device=q.device, dtype=torch.float32)
        da_p = torch.empty(shape, device=q.device, dtype=torch.float32)
        db_p = torch.empty(shape, device=q.device, dtype=torch.float32)
        dg_p = torch.empty(shape, device=q.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        _chunk_backward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, dout, checkpoints, states,
            dq_p, dk_p, da_p, db_p, dg_p, dv,
            T, H=H, K=K, V=V, NC=nc, NV=nv, BK=bk, BV_=BV,
            BSTRIDE=B * T * H * K, SCALE=K ** -0.5,
            num_warps=4, num_stages=2,
        )
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        da, db = torch.empty_like(alpha), torch.empty_like(beta)
        dg = torch.empty_like(gk)
        n = B * T * H * K
        _reduce_partials_kernel[(triton.cdiv(n, 256),)](
            dq_p, dk_p, da_p, db_p, dg_p, dq, dk, da, db, dg,
            N=n, NV=nv, BLOCK=256, num_warps=4,
        )
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
