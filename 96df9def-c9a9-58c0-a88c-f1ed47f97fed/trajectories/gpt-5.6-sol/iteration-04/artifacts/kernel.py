import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BT = 16
BV = 64


@triton.jit
def _chunk_forward_kernel(q, k, v, a, b, g, out, checkpoints, vnew_store,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                          V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                          BT_: tl.constexpr, BV_: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV_ + tl.arange(0, BV_)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV_), tl.float32)
    cp = (((ibh * (NC + 1)) * K + ok[:, None]) * V + ov[None, :]).to(tl.int64)
    tl.store(checkpoints + cp, state, mask=mk[:, None] & mv[None, :])
    ti = tl.arange(0, BT_)
    scale = 1.0 / tl.sqrt(K + 0.0)
    for ic in range(NC):
        ts, mt = ic * BT_ + ti, ic * BT_ + ti < T
        pk = ((ib * T + ts[:, None]) * H + ih) * K + ok[None, :]
        pv = ((ib * T + ts[:, None]) * H + ih) * V + ov[None, :]
        mtk, mtv = mt[:, None] & mk[None, :], mt[:, None] & mv[None, :]
        qq = tl.load(q + pk, mask=mtk, other=0.).to(tl.float32) * scale
        kk = tl.load(k + pk, mask=mtk, other=0.).to(tl.float32)
        aa = tl.load(a + pk, mask=mtk, other=0.).to(tl.float32)
        bb = tl.load(b + pk, mask=mtk, other=0.).to(tl.float32)
        gg = tl.load(g + pk, mask=mtk, other=0.).to(tl.float32)
        vv = tl.load(v + pv, mask=mtv, other=0.).to(tl.float32)
        G = tl.cumsum(gg, axis=0)
        causal = ti[:, None, None] >= ti[None, :, None]
        strict = ti[:, None, None] > ti[None, :, None]
        valid = mt[:, None, None] & mt[None, :, None]
        rel = libdevice.exp(G[:, None, :] - G[None, :, :])
        oldrel = libdevice.exp(G[:, None, :] - gg[:, None, :] - G[None, :, :])
        aqk = tl.sum(qq[:, None, :] * kk[None, :, :] * tl.where(causal & valid, rel, 0.), 2)
        aqb = tl.sum(qq[:, None, :] * bb[None, :, :] * tl.where(causal & valid, rel, 0.), 2)
        aak = tl.sum(aa[:, None, :] * kk[None, :, :] * tl.where(strict & valid, oldrel, 0.), 2)
        closure = tl.sum(aa[:, None, :] * bb[None, :, :] * tl.where(strict & valid, oldrel, 0.), 2)
        cols = tl.arange(0, BT_)
        for row_id in tl.static_range(1, BT_):
            pick = ti == row_id
            row = tl.sum(tl.where(pick[:, None], closure, 0.), 0)
            row += tl.sum(row[:, None] * closure, 0) * (cols < row_id)
            closure = tl.where(pick[:, None], row[None, :], closure)
        closure += ti[:, None] == ti[None, :]
        u = tl.dot(closure, tl.dot(aak, vv, input_precision="ieee"), input_precision="ieee")
        w = tl.dot(closure, aa * libdevice.exp(G - gg), input_precision="ieee")
        vn = u + tl.dot(w, state, input_precision="ieee")
        oo = tl.dot(qq * libdevice.exp(G), state, input_precision="ieee")
        oo += tl.dot(aqk, vv, input_precision="ieee") + tl.dot(aqb, vn, input_precision="ieee")
        tl.store(out + pv, oo, mask=mtv)
        tl.store(vnew_store + pv, vn, mask=mtv)
        count = tl.minimum(T - ic * BT_, BT_)
        glast = tl.sum(tl.where(ti[:, None] == count - 1, G, 0.), 0)
        tail = libdevice.exp(glast[None, :] - G)
        state = state * libdevice.exp(glast)[:, None]
        state += tl.dot(tl.trans(kk * tail), vv, input_precision="ieee")
        state += tl.dot(tl.trans(bb * tail), vn, input_precision="ieee")
        cp = ((((ibh * (NC + 1) + ic + 1) * K + ok[:, None]) * V) + ov[None, :]).to(tl.int64)
        tl.store(checkpoints + cp, state, mask=mk[:, None] & mv[None, :])



@triton.jit
def _chunk_states_kernel(k, v, a, b, g, checkpoints, states,
                         T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                         V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                         BT_: tl.constexpr, BV_: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV_ + tl.arange(0, BV_)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV_), tl.float32)
    for ic in range(NC):
        for j in tl.static_range(0, BT_):
            it = ic * BT_ + j
            active = it < T
            pk = ((ib * T + it) * H + ih) * K + ok
            pv = ((ib * T + it) * H + ih) * V + ov
            kj = tl.load(k + pk, mask=active & mk, other=0.).to(tl.float32)
            aj = tl.load(a + pk, mask=active & mk, other=0.).to(tl.float32)
            bj = tl.load(b + pk, mask=active & mk, other=0.).to(tl.float32)
            gj = tl.load(g + pk, mask=active & mk, other=0.).to(tl.float32)
            vj = tl.load(v + pv, mask=active & mv, other=0.).to(tl.float32)
            read = tl.sum(aj[:, None] * state, 0)
            candidate = state * libdevice.exp(gj)[:, None]
            candidate += kj[:, None] * vj[None, :] + bj[:, None] * read[None, :]
            state = tl.where(active, candidate, state)
            ps = ((ibh * T + it) * K + ok[:, None]) * V + ov[None, :]
            tl.store(states + ps.to(tl.int64), state, mask=active & mk[:, None] & mv[None, :])


@triton.jit
def _chunk_backward_kernel(q, k, v, a, b, g, dout, checkpoints, vnew_store,
                           states, dv_out, partial, T: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
                           K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
                           NV: tl.constexpr, BK: tl.constexpr, BT_: tl.constexpr, BV_: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV_ + tl.arange(0, BV_)
    ti = tl.arange(0, BT_)
    mk, mv = ok < K, ov < V
    ds = tl.zeros((BK, BV_), tl.float32)
    scale = 1.0 / tl.sqrt(K + 0.0)
    plane = NV * B * T * H * K
    for rc in range(NC):
        ic = NC - 1 - rc
        ts, mt = ic * BT_ + ti, ic * BT_ + ti < T
        pk = ((ib * T + ts[:, None]) * H + ih) * K + ok[None, :]
        pv = ((ib * T + ts[:, None]) * H + ih) * V + ov[None, :]
        mtk, mtv = mt[:, None] & mk[None, :], mt[:, None] & mv[None, :]
        qq = tl.load(q + pk, mask=mtk, other=0.).to(tl.float32) * scale
        kk = tl.load(k + pk, mask=mtk, other=0.).to(tl.float32)
        aa = tl.load(a + pk, mask=mtk, other=0.).to(tl.float32)
        bb = tl.load(b + pk, mask=mtk, other=0.).to(tl.float32)
        gg = tl.load(g + pk, mask=mtk, other=0.).to(tl.float32)
        vv = tl.load(v + pv, mask=mtv, other=0.).to(tl.float32)
        dd = tl.load(dout + pv, mask=mtv, other=0.).to(tl.float32)
        vn = tl.load(vnew_store + pv, mask=mtv, other=0.).to(tl.float32)
        G = tl.cumsum(gg, axis=0)
        cp = ((((ibh * (NC + 1) + ic) * K + ok[:, None]) * V) + ov[None, :]).to(tl.int64)
        s0 = tl.load(checkpoints + cp, mask=mk[:, None] & mv[None, :], other=0.)
        for rj in tl.static_range(0, BT_):
            j = BT_ - 1 - rj
            active = ic * BT_ + j < T
            gj = tl.sum(tl.where(ti[:, None] == j, gg, 0.), 0)
            Gj = tl.sum(tl.where(ti[:, None] == j, G, 0.), 0)
            qj = tl.sum(tl.where(ti[:, None] == j, qq, 0.), 0)
            kj = tl.sum(tl.where(ti[:, None] == j, kk, 0.), 0)
            aj = tl.sum(tl.where(ti[:, None] == j, aa, 0.), 0)
            bj = tl.sum(tl.where(ti[:, None] == j, bb, 0.), 0)
            vj = tl.sum(tl.where(ti[:, None] == j, vv, 0.), 0)
            dj = tl.sum(tl.where(ti[:, None] == j, dd, 0.), 0)
            now_off = ((ibh * T + ic * BT_ + j) * K + ok[:, None]) * V + ov[None, :]
            snow = tl.load(states + now_off.to(tl.int64), mask=mk[:, None] & mv[None, :], other=0.)
            if j == 0:
                sold = s0
            else:
                old_off = ((ibh * T + ic * BT_ + j - 1) * K + ok[:, None]) * V + ov[None, :]
                sold = tl.load(states + old_off.to(tl.int64), mask=mk[:, None] & mv[None, :], other=0.)
            total = ds + qj[:, None] * dj[None, :]
            read = tl.reshape(tl.dot(aj[None, :], sold, input_precision="ieee"), (BV_,))
            dread = tl.reshape(tl.dot(bj[None, :], total, input_precision="ieee"), (BV_,))
            vals0 = tl.reshape(tl.dot(snow, dj[:, None], input_precision="ieee"), (BK,)) * scale
            vals1 = tl.reshape(tl.dot(total, vj[:, None], input_precision="ieee"), (BK,))
            vals2 = tl.reshape(tl.dot(sold, dread[:, None], input_precision="ieee"), (BK,))
            vals3 = tl.reshape(tl.dot(total, read[:, None], input_precision="ieee"), (BK,))
            vals4 = tl.sum(total * (libdevice.exp(gj)[:, None] * sold), 1)
            t_abs = ic * BT_ + j
            elem = ((ib * T + t_abs) * H + ih) * K + ok
            part = iv * T * H * K + elem
            pm = active & mk
            tl.store(partial + 0 * plane + part, vals0, mask=pm)
            tl.store(partial + 1 * plane + part, vals1, mask=pm)
            tl.store(partial + 2 * plane + part, vals2, mask=pm)
            tl.store(partial + 3 * plane + part, vals3, mask=pm)
            tl.store(partial + 4 * plane + part, vals4, mask=pm)
            voff = ((ib * T + t_abs) * H + ih) * V + ov
            tl.store(dv_out + voff, tl.sum(total * kj[:, None], 0), mask=active & mv)
            ds = tl.where(active, total * libdevice.exp(gj)[:, None] + aj[:, None] * dread[None, :], ds)


@triton.jit
def _reduce_partials_kernel(partial, dq, dk, da, db, dg,
                            N: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    x0 = tl.zeros((BLOCK,), tl.float32)
    x1 = tl.zeros((BLOCK,), tl.float32)
    x2 = tl.zeros((BLOCK,), tl.float32)
    x3 = tl.zeros((BLOCK,), tl.float32)
    x4 = tl.zeros((BLOCK,), tl.float32)
    for iv in tl.static_range(0, NV):
        p = iv * N + off
        x0 += tl.load(partial + (0 * NV + iv) * N + off, mask=mask, other=0.)
        x1 += tl.load(partial + (1 * NV + iv) * N + off, mask=mask, other=0.)
        x2 += tl.load(partial + (2 * NV + iv) * N + off, mask=mask, other=0.)
        x3 += tl.load(partial + (3 * NV + iv) * N + off, mask=mask, other=0.)
        x4 += tl.load(partial + (4 * NV + iv) * N + off, mask=mask, other=0.)
    tl.store(dq + off, x0, mask=mask)
    tl.store(dk + off, x1, mask=mask)
    tl.store(da + off, x2, mask=mask)
    tl.store(db + off, x3, mask=mask)
    tl.store(dg + off, x4, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        nc, nv = triton.cdiv(T, BT), triton.cdiv(V, BV)
        bk = max(16, triton.next_power_of_2(K))
        out = torch.empty_like(v)
        checkpoints = torch.empty((B * H, nc + 1, K, V), device=q.device, dtype=torch.float32)
        vnew = torch.empty((B, T, H, V), device=q.device, dtype=torch.float32)
        _chunk_forward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, out, checkpoints, vnew,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BT_=BT, BV_=BV,
            num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints, vnew)
        ctx.shape = B, T, H, K, V, nc, nv, bk
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints, vnew = ctx.saved_tensors
        B, T, H, K, V, nc, nv, bk = ctx.shape
        dout = dout.contiguous()
        states = torch.empty((B * H, T, K, V), device=q.device, dtype=torch.float32)
        _chunk_states_kernel[(nv, B * H)](
            k, v, alpha, beta, gk, checkpoints, states,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BT_=BT, BV_=BV,
            num_warps=8, num_stages=1)
        dv = torch.empty_like(v)
        partial = torch.zeros((5, nv, B, T, H, K), device=q.device, dtype=torch.float32)
        _chunk_backward_kernel[(nv, B * H)](
            q, k, v, alpha, beta, gk, dout, checkpoints, vnew, states, dv, partial,
            T=T, B=B, H=H, K=K, V=V, NC=nc, NV=nv, BK=bk, BT_=BT, BV_=BV,
            num_warps=8, num_stages=1)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        da, db, dg = torch.empty_like(alpha), torch.empty_like(beta), torch.empty_like(gk)
        n = B * T * H * K
        _reduce_partials_kernel[(triton.cdiv(n, 256),)](
            partial, dq, dk, da, db, dg, N=n, NV=nv, BLOCK=256, num_warps=8)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
