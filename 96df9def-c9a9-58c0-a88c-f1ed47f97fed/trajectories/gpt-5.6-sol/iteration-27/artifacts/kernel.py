import torch
import triton
import triton.language as tl

_FWD_BV = 32
_BWD_BV = 16


@triton.jit
def _forward_chunks(q, k, v, a, b, g, out, states,
                    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov, ot = tl.arange(0, BK), iv * BV + tl.arange(0, BV), tl.arange(0, 16)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BV), tl.float32)
    scale: tl.constexpr = K ** -0.5
    for ic in range(tl.cdiv(T, 16)):
        ts, mt = ic * 16 + ot, ic * 16 + ot < T
        xk = ((ib * T + ts[:, None]) * H + ih) * K + ok[None, :]
        xv = ((ib * T + ts[:, None]) * H + ih) * V + ov[None, :]
        mtk, mtv = mt[:, None] & mk[None, :], mt[:, None] & mv[None, :]
        qq = tl.load(q + xk, mask=mtk, other=0.).to(tl.float32)
        kk = tl.load(k + xk, mask=mtk, other=0.).to(tl.float32)
        aa = tl.load(a + xk, mask=mtk, other=0.).to(tl.float32)
        bb = tl.load(b + xk, mask=mtk, other=0.).to(tl.float32)
        gg = tl.load(g + xk, mask=mtk, other=0.).to(tl.float32)
        vv = tl.load(v + xv, mask=mtv, other=0.).to(tl.float32)
        gp = tl.cumsum(gg, axis=0)
        gm = gp - gg
        center = tl.sum(tl.where(ot[:, None] == 8, gm, 0.), axis=0)
        ag = aa * tl.exp(tl.maximum(tl.minimum(gm - center[None, :], 30.), -30.))
        inv = tl.exp(tl.maximum(tl.minimum(center[None, :] - gp, 30.), -30.))
        bg, kg = bb * inv, kk * inv
        qg = qq * tl.exp(tl.maximum(tl.minimum(gp - center[None, :], 30.), -30.)) * scale
        sc = state * tl.exp(center[:, None])
        lab = tl.where(ot[:, None] > ot[None, :],
                       tl.dot(ag, tl.trans(bg), input_precision="ieee"), 0.)
        lak = tl.where(ot[:, None] > ot[None, :],
                       tl.dot(ag, tl.trans(kg), input_precision="ieee"), 0.)
        rr = tl.dot(ag, sc, input_precision="ieee") + tl.dot(lak, vv, input_precision="ieee")
        for j in tl.static_range(1, 16):
            row = tl.sum(tl.where(ot[:, None] == j, rr, 0.), axis=0)
            lab_row = tl.sum(tl.where(ot[:, None] == j, lab, 0.), axis=0)
            row += tl.sum(tl.where(ot[:, None] < j, lab_row[:, None] * rr, 0.), axis=0)
            rr = tl.where(ot[:, None] == j, row[None, :], rr)
        incl = ot[:, None] >= ot[None, :]
        aqk = tl.where(incl, tl.dot(qg, tl.trans(kg), input_precision="ieee"), 0.)
        aqb = tl.where(incl, tl.dot(qg, tl.trans(bg), input_precision="ieee"), 0.)
        oo = tl.dot(qg, sc, input_precision="ieee")
        oo += tl.dot(aqk, vv, input_precision="ieee") + tl.dot(aqb, rr, input_precision="ieee")
        tl.store(out + xv, oo, mask=mtv)
        xs = sc
        snap = state
        for j in tl.static_range(0, 16):
            bj = tl.sum(tl.where(ot[:, None] == j, bg, 0.), axis=0)
            kj = tl.sum(tl.where(ot[:, None] == j, kg, 0.), axis=0)
            rj = tl.sum(tl.where(ot[:, None] == j, rr, 0.), axis=0)
            vj = tl.sum(tl.where(ot[:, None] == j, vv, 0.), axis=0)
            xs += bj[:, None] * rj[None, :] + kj[:, None] * vj[None, :]
            aj0 = tl.sum(tl.where(ot[:, None] == j, aa, 0.), axis=0)
            bj0 = tl.sum(tl.where(ot[:, None] == j, bb, 0.), axis=0)
            kj0 = tl.sum(tl.where(ot[:, None] == j, kk, 0.), axis=0)
            gj0 = tl.sum(tl.where(ot[:, None] == j, gg, 0.), axis=0)
            read0 = tl.sum(snap * aj0[:, None], axis=0)
            snap = snap * tl.exp(gj0)[:, None] + kj0[:, None] * vj[None, :] + bj0[:, None] * read0[None, :]
            tj = ic * 16 + j
            ps = ((((ib * T + tj) * H + ih) * K + ok[:, None]) * V + ov[None, :])
            tl.store(states + ps, snap, mask=(tj < T) & mk[:, None] & mv[None, :])
        state = snap


@triton.jit
def _backward_scan(q, k, v, a, b, g, do, states, dv, partial,
                   T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                   V: tl.constexpr, B: tl.constexpr, NV: tl.constexpr,
                   BK: tl.constexpr, BV: tl.constexpr):
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mk, mv = ok < K, ov < V
    ds = tl.zeros((BK, BV), tl.float32)
    scale: tl.constexpr = K ** -0.5
    for cb in range(tl.cdiv(T, 16)):
        ic = tl.cdiv(T, 16) - 1 - cb
        for jj in tl.static_range(0, 16):
            t, valid = ic * 16 + 15 - jj, ic * 16 + 15 - jj < T
            xk = ((ib * T + t) * H + ih) * K + ok
            xv = ((ib * T + t) * H + ih) * V + ov
            qq = tl.load(q + xk, mask=valid & mk, other=0.).to(tl.float32)
            kk = tl.load(k + xk, mask=valid & mk, other=0.).to(tl.float32)
            aa = tl.load(a + xk, mask=valid & mk, other=0.).to(tl.float32)
            bb = tl.load(b + xk, mask=valid & mk, other=0.).to(tl.float32)
            gg = tl.load(g + xk, mask=valid & mk, other=0.).to(tl.float32)
            vt = tl.load(v + xv, mask=valid & mv, other=0.).to(tl.float32)
            dot = tl.load(do + xv, mask=valid & mv, other=0.).to(tl.float32)
            ps = ((((ib * T + t) * H + ih) * K + ok[:, None]) * V + ov[None, :])
            cur = tl.load(states + ps, mask=valid & mk[:, None] & mv[None, :], other=0.)
            pp = ((((ib * T + t - 1) * H + ih) * K + ok[:, None]) * V + ov[None, :])
            prev = tl.load(states + pp, mask=(t > 0) & valid & mk[:, None] & mv[None, :], other=0.)
            ds += qq[:, None] * (dot[None, :] * scale)
            dqv = tl.sum(cur * dot[None, :], axis=1) * scale
            dkv = tl.sum(ds * vt[None, :], axis=1)
            dvt = tl.sum(ds * kk[:, None], axis=0)
            read = tl.sum(prev * aa[:, None], axis=0)
            dbv = tl.sum(ds * read[None, :], axis=1)
            dr = tl.sum(ds * bb[:, None], axis=0)
            dav = tl.sum(prev * dr[None, :], axis=1)
            eg = tl.exp(gg)
            dgv = tl.sum(ds * prev * eg[:, None], axis=1)
            ds = ds * eg[:, None] + aa[:, None] * dr[None, :]
            tl.store(dv + xv, dvt, mask=valid & mv)
            base = (((((iv * B + ib) * T + t) * H + ih) * 5) * K + ok)
            tl.store(partial + base + 0 * K, dqv, mask=valid & mk)
            tl.store(partial + base + 1 * K, dkv, mask=valid & mk)
            tl.store(partial + base + 2 * K, dav, mask=valid & mk)
            tl.store(partial + base + 3 * K, dbv, mask=valid & mk)
            tl.store(partial + base + 4 * K, dgv, mask=valid & mk)


@triton.jit
def _reduce_v_partials(partial, dq, dk, da, db, dg,
                       N: tl.constexpr, K: tl.constexpr, NV: tl.constexpr):
    x = tl.program_id(0) * 256 + tl.arange(0, 256)
    m = x < N
    r0 = tl.zeros((256,), tl.float32)
    r1 = tl.zeros((256,), tl.float32)
    r2 = tl.zeros((256,), tl.float32)
    r3 = tl.zeros((256,), tl.float32)
    r4 = tl.zeros((256,), tl.float32)
    group = x // K
    channel = x - group * K
    for iv in tl.static_range(0, NV):
        p = iv * N * 5 + group * 5 * K + channel
        r0 += tl.load(partial + p + 0 * K, mask=m, other=0.)
        r1 += tl.load(partial + p + 1 * K, mask=m, other=0.)
        r2 += tl.load(partial + p + 2 * K, mask=m, other=0.)
        r3 += tl.load(partial + p + 3 * K, mask=m, other=0.)
        r4 += tl.load(partial + p + 4 * K, mask=m, other=0.)
    tl.store(dq + x, r0, mask=m)
    tl.store(dk + x, r1, mask=m)
    tl.store(da + x, r2, mask=m)
    tl.store(db + x, r3, mask=m)
    tl.store(dg + x, r4, mask=m)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V, bk = v.shape[-1], max(16, triton.next_power_of_2(K))
        out = torch.empty_like(v)
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _forward_chunks[(triton.cdiv(V, _FWD_BV), B * H)](
            q, k, v, alpha, beta, gk, out, states, T=T, H=H, K=K,
            V=V, BK=bk, BV=_FWD_BV, num_warps=8, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, states)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, states = ctx.saved_tensors
        B, T, H, K = q.shape
        V, bk = v.shape[-1], max(16, triton.next_power_of_2(K))
        nv = triton.cdiv(V, _BWD_BV)
        dv = torch.empty_like(v)
        partial = torch.zeros((nv, B, T, H, 5, K), device=q.device, dtype=torch.float32)
        _backward_scan[(nv, B * H)](
            q, k, v, alpha, beta, gk, dout.contiguous(), states, dv, partial,
            T=T, H=H, K=K, V=V, B=B, NV=nv, BK=bk, BV=_BWD_BV,
            num_warps=4, num_stages=2)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        da, db, dg = torch.empty_like(alpha), torch.empty_like(beta), torch.empty_like(gk)
        n = B * T * H * K
        _reduce_v_partials[(triton.cdiv(n, 256),)](
            partial, dq, dk, da, db, dg, N=n, K=K, NV=nv, num_warps=4)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
