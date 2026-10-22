import torch
import triton
import triton.language as tl

_BT = 16
_BV = 32
_BWD_BV = 16


@triton.jit
def _chunk_forward_kernel(q, k, v, a, b, g, out, checkpoints,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                          NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    ok, ov, ot = tl.arange(0, BK), iv * BV + tl.arange(0, BV), tl.arange(0, 16)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV), tl.float32)
    scale: tl.constexpr = K ** -0.5
    row, col = tl.arange(0, 16)[:, None], tl.arange(0, 16)[None, :]
    eye, lower, causal = row == col, row > col, row >= col
    for ic in range(NC):
        cp = checkpoints + ((((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(cp, state, mask=mk[:, None] & mv[None, :])
        tt = ic * 16 + ot
        valid = tt < T
        xoff = (ib * T + tt[:, None]) * H * K + ih * K + ok[None, :]
        voff = (ib * T + tt[:, None]) * H * V + ih * V + ov[None, :]
        xm = valid[:, None] & mk[None, :]
        qm = tl.load(q + xoff, mask=xm, other=0.).to(tl.float32) * scale
        km = tl.load(k + xoff, mask=xm, other=0.).to(tl.float32)
        am = tl.load(a + xoff, mask=xm, other=0.).to(tl.float32)
        bm = tl.load(b + xoff, mask=xm, other=0.).to(tl.float32)
        gm = tl.load(g + xoff, mask=xm, other=0.).to(tl.float32)
        vm = tl.load(v + voff, mask=valid[:, None] & mv[None, :], other=0.).to(tl.float32)
        gc = tl.cumsum(gm, axis=0)
        gp = gc - gm
        gend = tl.sum(gc * (ot[:, None] == 15), axis=0)
        qi, ai = qm * tl.exp(gc), am * tl.exp(gp)
        kend = km * tl.exp(gend[None, :] - gc)
        bend = bm * tl.exp(gend[None, :] - gc)
        di = gc[:, None, :] - gc[None, :, :]
        dp = gp[:, None, :] - gc[None, :, :]
        mm = valid[:, None] & valid[None, :]
        aqk = tl.where(causal & mm, tl.sum(qm[:, None, :] * km[None, :, :] * tl.exp(di), axis=2), 0.)
        aqb = tl.where(causal & mm, tl.sum(qm[:, None, :] * bm[None, :, :] * tl.exp(di), axis=2), 0.)
        aak = tl.where(lower & mm, tl.sum(am[:, None, :] * km[None, :, :] * tl.exp(dp), axis=2), 0.)
        lab = tl.where(lower & mm, tl.sum(am[:, None, :] * bm[None, :, :] * tl.exp(dp), axis=2), 0.)
        inv = tl.where(eye, 1., 0.)
        power = lab
        for _ in tl.static_range(0, 4):
            inv += tl.dot(power, inv, input_precision='ieee')
            power = tl.dot(power, power, input_precision='ieee')
        u = tl.dot(inv, tl.dot(aak, vm, input_precision='ieee'), input_precision='ieee')
        w = tl.dot(inv, ai, input_precision='ieee')
        read = u + tl.dot(w, state, input_precision='ieee')
        om = tl.dot(aqk, vm, input_precision='ieee') + tl.dot(aqb, read, input_precision='ieee')
        om += tl.dot(qi, state, input_precision='ieee')
        tl.store(out + voff, om, mask=valid[:, None] & mv[None, :])
        state *= tl.exp(gend)[:, None]
        state += tl.dot(tl.trans(kend), vm, input_precision='ieee')
        state += tl.dot(tl.trans(bend), read, input_precision='ieee')



@triton.jit
def _exact_checkpoints_kernel(k, v, a, b, g, checkpoints,
                              T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                              NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV), tl.float32)
    for ic in range(NC):
        cp = checkpoints + ((((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(cp, state, mask=mk[:, None] & mv[None, :])
        for j in tl.static_range(0, 16):
            t, valid = ic * 16 + j, ic * 16 + j < T
            xp = (ib * T + t) * H * K + ih * K + ok
            vp = (ib * T + t) * H * V + ih * V + ov
            kk = tl.load(k + xp, mask=valid & mk, other=0.).to(tl.float32)
            aa = tl.load(a + xp, mask=valid & mk, other=0.).to(tl.float32)
            bb = tl.load(b + xp, mask=valid & mk, other=0.).to(tl.float32)
            gg = tl.load(g + xp, mask=valid & mk, other=0.).to(tl.float32)
            vv = tl.load(v + vp, mask=valid & mv, other=0.).to(tl.float32)
            aa16 = tl.broadcast_to(aa[None, :], (16, BK))
            read16 = tl.dot(aa16, state, input_precision='ieee')
            read = tl.sum(read16 * (tl.arange(0, 16)[:, None] == 0), axis=0)
            state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]


@triton.jit
def _expand_states_kernel(k, v, a, b, g, checkpoints, states,
                          T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                          NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, ic, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ib, ih = bh // H, bh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    cp = checkpoints + ((((ib * NC + ic) * H + ih) * K + ok[:, None]) * V + ov[None, :])
    state = tl.load(cp, mask=mk[:, None] & mv[None, :], other=0.).to(tl.float32)
    for j in tl.static_range(0, 16):
        t, valid = ic * 16 + j, ic * 16 + j < T
        sp = states + ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
        tl.store(sp, state, mask=valid & mk[:, None] & mv[None, :])
        xp, vp = (ib * T + t) * H * K + ih * K + ok, (ib * T + t) * H * V + ih * V + ov
        kk = tl.load(k + xp, mask=valid & mk, other=0.).to(tl.float32)
        aa = tl.load(a + xp, mask=valid & mk, other=0.).to(tl.float32)
        bb = tl.load(b + xp, mask=valid & mk, other=0.).to(tl.float32)
        gg = tl.load(g + xp, mask=valid & mk, other=0.).to(tl.float32)
        vv = tl.load(v + vp, mask=valid & mv, other=0.).to(tl.float32)
        aa16 = tl.broadcast_to(aa[None, :], (16, BK))
        read16 = tl.dot(aa16, state, input_precision='ieee')
        read = tl.sum(read16 * (tl.arange(0, 16)[:, None] == 0), axis=0)
        state = state * tl.exp(gg)[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]


@triton.jit
def _backward_kernel(q, k, v, a, b, g, do, states, dv, pq, pk, pa, pb, pg,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     NC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, N: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    dstate = tl.zeros((BK, BV), tl.float32)
    scale: tl.constexpr = K ** -0.5
    for rc in range(NC):
        ic = NC - 1 - rc
        for rj in tl.static_range(0, 16):
            t, valid = ic * 16 + 15 - rj, ic * 16 + 15 - rj < T
            xp, vp = (ib * T + t) * H * K + ih * K + ok, (ib * T + t) * H * V + ih * V + ov
            sp = states + ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
            old = tl.load(sp, mask=valid & mk[:, None] & mv[None, :], other=0.).to(tl.float32)
            qq = tl.load(q + xp, mask=valid & mk, other=0.).to(tl.float32)
            kk = tl.load(k + xp, mask=valid & mk, other=0.).to(tl.float32)
            aa = tl.load(a + xp, mask=valid & mk, other=0.).to(tl.float32)
            bb = tl.load(b + xp, mask=valid & mk, other=0.).to(tl.float32)
            gg = tl.load(g + xp, mask=valid & mk, other=0.).to(tl.float32)
            vv = tl.load(v + vp, mask=valid & mv, other=0.).to(tl.float32)
            dd = tl.load(do + vp, mask=valid & mv, other=0.).to(tl.float32)
            decay = tl.exp(gg)
            aa16 = tl.broadcast_to(aa[None, :], (16, BK))
            read16 = tl.dot(aa16, old, input_precision='ieee')
            read = tl.sum(read16 * (tl.arange(0, 16)[:, None] == 0), axis=0)
            new = old * decay[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
            grad = dstate + qq[:, None] * (scale * dd)[None, :]
            dqv = tl.sum(new * dd[None, :], axis=1) * scale
            dkv = tl.sum(grad * vv[None, :], axis=1)
            dbv = tl.sum(grad * read[None, :], axis=1)
            dr = tl.sum(bb[:, None] * grad, axis=0)
            dav = tl.sum(old * dr[None, :], axis=1)
            dgv = tl.sum(grad * old, axis=1) * decay
            dvv = tl.sum(kk[:, None] * grad, axis=0)
            tl.store(dv + vp, dvv, mask=valid & mv)
            po = iv * N + (ib * T + t) * H * K + ih * K + ok
            pm = valid & mk
            tl.store(pq + po, dqv, mask=pm); tl.store(pk + po, dkv, mask=pm)
            tl.store(pa + po, dav, mask=pm); tl.store(pb + po, dbv, mask=pm); tl.store(pg + po, dgv, mask=pm)
            dstate = grad * decay[:, None] + aa[:, None] * dr[None, :]


@triton.jit
def _reduce_partials_kernel(pq, pk, pa, pb, pg, dq, dk, da, db, dg,
                            N: tl.constexpr, NV: tl.constexpr):
    x = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = x < N
    sq = tl.zeros((256,), tl.float32); sk = tl.zeros((256,), tl.float32)
    sa = tl.zeros((256,), tl.float32); sb = tl.zeros((256,), tl.float32); sg = tl.zeros((256,), tl.float32)
    for iv in tl.static_range(0, NV):
        off = iv * N + x
        sq += tl.load(pq + off, mask=mask, other=0.); sk += tl.load(pk + off, mask=mask, other=0.)
        sa += tl.load(pa + off, mask=mask, other=0.); sb += tl.load(pb + off, mask=mask, other=0.)
        sg += tl.load(pg + off, mask=mask, other=0.)
    tl.store(dq + x, sq, mask=mask); tl.store(dk + x, sk, mask=mask)
    tl.store(da + x, sa, mask=mask); tl.store(db + x, sb, mask=mask); tl.store(dg + x, sg, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape; V = v.shape[-1]
        nc, bk, nv = triton.cdiv(T, _BT), max(16, triton.next_power_of_2(K)), triton.cdiv(V, _BV)
        out = torch.empty_like(v)
        checkpoints = torch.empty((B, nc, H, K, V), device=q.device, dtype=torch.float32)
        _chunk_forward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, checkpoints,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=_BV, num_warps=8, num_stages=2)
        exact_bv = max(16, triton.next_power_of_2(V))
        _exact_checkpoints_kernel[(1, B * H)](k, v, alpha, beta, gk, checkpoints,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=exact_bv, num_warps=4, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        B, T, H, K = q.shape; V = v.shape[-1]
        nc = triton.cdiv(T, _BT)
        bk = max(16, triton.next_power_of_2(K))
        bv = _BWD_BV
        nv = triton.cdiv(V, bv)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_states_kernel[(nv, nc, B * H)](k, v, alpha, beta, gk, checkpoints, states,
            T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv, num_warps=4, num_stages=2)
        shape = (nv, B, T, H, K)
        pq = torch.zeros(shape, device=q.device, dtype=torch.float32); pk = torch.zeros_like(pq)
        pa = torch.zeros_like(pq); pb = torch.zeros_like(pq); pg = torch.zeros_like(pq)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        da, db, dg = torch.empty_like(alpha), torch.empty_like(beta), torch.empty_like(gk)
        n = B * T * H * K
        _backward_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, dout.contiguous(), states, dv,
            pq, pk, pa, pb, pg, T=T, H=H, K=K, V=V, NC=nc, BK=bk, BV=bv, N=n,
            num_warps=4, num_stages=2)
        _reduce_partials_kernel[(triton.cdiv(n, 256),)](pq, pk, pa, pb, pg, dq, dk, da, db, dg,
            N=n, NV=nv, num_warps=4)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
