import torch
import triton
import triton.language as tl

BT = 16
BV = 32


@triton.jit
def _chunk_fwd_kernel(q, k, v, a, b, g, out, checkpoints, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr, BLOCK_V: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    ot, ok = tl.arange(0, 16), tl.arange(0, BK)
    ov = iv * BLOCK_V + tl.arange(0, BLOCK_V)
    mk, mv = ok < K, ov < V
    state = tl.zeros((BK, BLOCK_V), tl.float32)
    eye = ot[:, None] == ot[None, :]
    lower = ot[:, None] > ot[None, :]
    causal = ot[:, None] >= ot[None, :]
    scale: tl.constexpr = K ** -0.5
    for ic in range(NC):
        tt = ic * 16 + ot
        mt = tt < T
        basek = (ib * T * H + tt * H + ih) * K
        basev = (ib * T * H + tt * H + ih) * V
        mtk, mtv = mt[:, None] & mk[None, :], mt[:, None] & mv[None, :]
        qq = tl.load(q + basek[:, None] + ok[None, :], mask=mtk, other=0.).to(tl.float32)
        kk = tl.load(k + basek[:, None] + ok[None, :], mask=mtk, other=0.).to(tl.float32)
        aa = tl.load(a + basek[:, None] + ok[None, :], mask=mtk, other=0.).to(tl.float32)
        bb = tl.load(b + basek[:, None] + ok[None, :], mask=mtk, other=0.).to(tl.float32)
        gg = tl.load(g + basek[:, None] + ok[None, :], mask=mtk, other=0.).to(tl.float32)
        vv = tl.load(v + basev[:, None] + ov[None, :], mask=mtv, other=0.).to(tl.float32)
        G = tl.cumsum(gg, axis=0)
        Gprev = G - gg
        w = aa * tl.exp(tl.maximum(Gprev, -80.))
        kb = kk * tl.exp(tl.minimum(-G, 80.))
        betab = bb * tl.exp(tl.minimum(-G, 80.))
        valid2 = mt[:, None] & mt[None, :]
        L = tl.where(lower & valid2, tl.dot(w, tl.trans(betab), input_precision="ieee"), 0.)
        Aak = tl.where(lower & valid2, tl.dot(w, tl.trans(kb), input_precision="ieee"), 0.)
        inv = tl.where(eye, 1., 0.) + L
        power = tl.dot(L, L, input_precision="ieee")
        inv += tl.dot(inv, power, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")
        inv += tl.dot(inv, power, input_precision="ieee")
        power = tl.dot(power, power, input_precision="ieee")
        inv += tl.dot(inv, power, input_precision="ieee")
        cp = checkpoints + ((ic * tl.num_programs(1) + bh) * K + ok[:, None]) * V + ov[None, :]
        tl.store(cp, state, mask=mk[:, None] & mv[None, :])
        rr = tl.dot(inv, tl.dot(w, state, input_precision="ieee") + tl.dot(Aak, vv, input_precision="ieee"), input_precision="ieee")
        qg = qq * tl.exp(G) * scale
        Aqk = tl.where(causal & valid2, tl.dot(qg, tl.trans(kb), input_precision="ieee"), 0.)
        Aqb = tl.where(causal & valid2, tl.dot(qg, tl.trans(betab), input_precision="ieee"), 0.)
        oo = tl.dot(qg, state, input_precision="ieee") + tl.dot(Aqk, vv, input_precision="ieee") + tl.dot(Aqb, rr, input_precision="ieee")
        tl.store(out + basev[:, None] + ov[None, :], oo, mask=mtv)
        last = tl.minimum((ic + 1) * 16, T) - ic * 16 - 1
        Glast = tl.sum(tl.where(ot[:, None] == last, G, 0.), axis=0)
        state = state * tl.exp(Glast)[:, None]
        state += tl.dot(tl.trans(kk * tl.exp(Glast[None, :] - G)), vv, input_precision="ieee")
        state += tl.dot(tl.trans(bb * tl.exp(Glast[None, :] - G)), rr, input_precision="ieee")


@triton.jit
def _expand_chunk_states_kernel(k, v, a, b, g, checkpoints, states, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, BH: tl.constexpr, BK: tl.constexpr, BLOCK_V: tl.constexpr):
    iv, ic_bh = tl.program_id(0), tl.program_id(1)
    ic, bh = ic_bh // BH, ic_bh % BH
    ib, ih = bh // H, bh % H
    ot, ok = tl.arange(0, 16), tl.arange(0, BK)
    ov = iv * BLOCK_V + tl.arange(0, BLOCK_V)
    tt = ic * 16 + ot
    mt, mk, mv = tt < T, ok < K, ov < V
    mtk, mtv = mt[:, None] & mk[None, :], mt[:, None] & mv[None, :]
    basek = (ib * T * H + tt * H + ih) * K
    basev = (ib * T * H + tt * H + ih) * V
    kk = tl.load(k + basek[:, None] + ok[None, :], mask=mtk, other=0.).to(tl.float32)
    aa = tl.load(a + basek[:, None] + ok[None, :], mask=mtk, other=0.).to(tl.float32)
    bb = tl.load(b + basek[:, None] + ok[None, :], mask=mtk, other=0.).to(tl.float32)
    gg = tl.load(g + basek[:, None] + ok[None, :], mask=mtk, other=0.).to(tl.float32)
    vv = tl.load(v + basev[:, None] + ov[None, :], mask=mtv, other=0.).to(tl.float32)
    cp = checkpoints + ((ic * BH + bh) * K + ok[:, None]) * V + ov[None, :]
    state0 = tl.load(cp, mask=mk[:, None] & mv[None, :], other=0.)
    G = tl.cumsum(gg, axis=0)
    Gprev = G - gg
    w = aa * tl.exp(tl.maximum(Gprev, -80.))
    kb = kk * tl.exp(tl.minimum(-G, 80.))
    betab = bb * tl.exp(tl.minimum(-G, 80.))
    lower = ot[:, None] > ot[None, :]
    valid2 = mt[:, None] & mt[None, :]
    L = tl.where(lower & valid2, tl.dot(w, tl.trans(betab), input_precision="ieee"), 0.)
    Aak = tl.where(lower & valid2, tl.dot(w, tl.trans(kb), input_precision="ieee"), 0.)
    inv = tl.where(ot[:, None] == ot[None, :], 1., 0.) + L
    power = tl.dot(L, L, input_precision="ieee")
    inv += tl.dot(inv, power, input_precision="ieee")
    power = tl.dot(power, power, input_precision="ieee")
    inv += tl.dot(inv, power, input_precision="ieee")
    power = tl.dot(power, power, input_precision="ieee")
    inv += tl.dot(inv, power, input_precision="ieee")
    rr = tl.dot(inv, tl.dot(w, state0, input_precision="ieee") + tl.dot(Aak, vv, input_precision="ieee"), input_precision="ieee")
    for jt in tl.static_range(16):
        sel_prev = tl.sum(tl.where(ot[:, None] == jt, Gprev, 0.), axis=0)
        coeff = tl.where((ot < jt)[:, None] & mt[:, None], tl.exp(tl.minimum(sel_prev[None, :] - G, 0.)), 0.)
        st = state0 * tl.exp(sel_prev)[:, None]
        st += tl.dot(tl.trans(kk * coeff), vv, input_precision="ieee")
        st += tl.dot(tl.trans(bb * coeff), rr, input_precision="ieee")
        target = ic * 16 + jt
        ps = states + (((ib * T + target) * H + ih) * K + ok[:, None]) * V + ov[None, :]
        tl.store(ps, st, mask=(target < T) & mk[:, None] & mv[None, :])


@triton.jit
def _reverse_scan_kernel(q, k, v, a, b, g, do, states, partial, dv, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BLOCK_V: tl.constexpr, NV: tl.constexpr, NROW: tl.constexpr):
    iv, bh = tl.program_id(0), tl.program_id(1)
    ib, ih = bh // H, bh % H
    ok = tl.arange(0, BK)
    ov = iv * BLOCK_V + tl.arange(0, BLOCK_V)
    mk, mv = ok < K, ov < V
    ds = tl.zeros((BK, BLOCK_V), tl.float32)
    scale: tl.constexpr = K ** -0.5
    for rev in range(T):
        t = T - 1 - rev
        row = (ib * T + t) * H + ih
        pk, pv = row * K + ok, row * V + ov
        qq = tl.load(q + pk, mask=mk, other=0.).to(tl.float32)
        kk = tl.load(k + pk, mask=mk, other=0.).to(tl.float32)
        aa = tl.load(a + pk, mask=mk, other=0.).to(tl.float32)
        bb = tl.load(b + pk, mask=mk, other=0.).to(tl.float32)
        gg = tl.load(g + pk, mask=mk, other=0.).to(tl.float32)
        vv = tl.load(v + pv, mask=mv, other=0.).to(tl.float32)
        dd = tl.load(do + pv, mask=mv, other=0.).to(tl.float32)
        sold = tl.load(states + (row * K + ok[:, None]) * V + ov[None, :], mask=mk[:, None] & mv[None, :], other=0.)
        decay = tl.exp(gg)
        read = tl.sum(sold * aa[:, None], axis=0)
        snew = sold * decay[:, None] + kk[:, None] * vv[None, :] + bb[:, None] * read[None, :]
        ds += qq[:, None] * (dd * scale)[None, :]
        vals0 = tl.sum(snew * dd[None, :], axis=1) * scale
        vals1 = tl.sum(ds * vv[None, :], axis=1)
        dvv = tl.sum(ds * kk[:, None], axis=0)
        vals3 = tl.sum(ds * read[None, :], axis=1)
        dread = tl.sum(ds * bb[:, None], axis=0)
        vals2 = tl.sum(sold * dread[None, :], axis=1)
        vals4 = tl.sum(ds * sold * decay[:, None], axis=1)
        for field in tl.static_range(5):
            val = tl.where(field == 0, vals0, tl.where(field == 1, vals1, tl.where(field == 2, vals2, tl.where(field == 3, vals3, vals4))))
            tl.store(partial + ((field * NV + iv) * NROW + row) * K + ok, val, mask=mk)
        tl.store(dv + pv, dvv, mask=mv)
        ds = ds * decay[:, None] + aa[:, None] * dread[None, :]


@triton.jit
def _reduce_partials_kernel(partial, dq, dk, da, db, dg, n: tl.constexpr, K: tl.constexpr, NV: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < n
    row, kk = ids // K, ids % K
    for field in tl.static_range(5):
        acc = tl.zeros((BLOCK,), tl.float32)
        for iv in tl.static_range(NV):
            acc += tl.load(partial + ((field * NV + iv) * (n // K) + row) * K + kk, mask=mask, other=0.)
        if field == 0: tl.store(dq + ids, acc, mask=mask)
        elif field == 1: tl.store(dk + ids, acc, mask=mask)
        elif field == 2: tl.store(da + ids, acc, mask=mask)
        elif field == 3: tl.store(db + ids, acc, mask=mask)
        else: tl.store(dg + ids, acc, mask=mask)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V, nc, nv = v.shape[-1], triton.cdiv(T, 16), triton.cdiv(v.shape[-1], BV)
        bk = max(16, triton.next_power_of_2(K))
        out = torch.empty_like(v)
        checkpoints = torch.empty((nc, B * H, K, V), device=q.device, dtype=torch.float32)
        _chunk_fwd_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, out, checkpoints, T=T, H=H, K=K, V=V, NC=nc, BK=bk, BLOCK_V=BV, num_warps=4, num_stages=2)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, checkpoints)
        ctx.shape = B, T, H, K, V, bk, nc, nv
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, alpha, beta, gk, checkpoints = ctx.saved_tensors
        B, T, H, K, V, bk, nc, nv = ctx.shape
        dout = dout.contiguous()
        states = torch.empty((B, T, H, K, V), device=q.device, dtype=torch.float32)
        _expand_chunk_states_kernel[(nv, nc * B * H)](k, v, alpha, beta, gk, checkpoints, states, T=T, H=H, K=K, V=V, NC=nc, BH=B*H, BK=bk, BLOCK_V=BV, num_warps=4, num_stages=2)
        nrow = B * T * H
        partial = torch.empty((5, nv, nrow, K), device=q.device, dtype=torch.float32)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        da = torch.empty_like(alpha)
        db = torch.empty_like(beta)
        dg = torch.empty_like(gk)
        dv = torch.empty_like(v)
        _reverse_scan_kernel[(nv, B * H)](q, k, v, alpha, beta, gk, dout, states, partial, dv, T=T, H=H, K=K, V=V, BK=bk, BLOCK_V=BV, NV=nv, NROW=nrow, num_warps=4, num_stages=2)
        n = nrow * K
        _reduce_partials_kernel[(triton.cdiv(n, 256),)](partial, dq, dk, da, db, dg, n=n, K=K, NV=nv, BLOCK=256, num_warps=4)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(q, k, v, alpha, beta, gk):
    return _Dplr.apply(q, k, v, alpha, beta, gk)
