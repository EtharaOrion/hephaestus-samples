
import torch
import triton
import triton.language as tl

BV = 32
C = 16


@triton.jit
def _build_wy(kp, gp, bp, ap, wp, T: tl.constexpr, HQ: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, NC: tl.constexpr,
              BK: tl.constexpr):
    ic = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HV
    h = bh - b * HV
    hq = h // (HV // HQ)
    t = ic * 16 + tl.arange(0, 16)
    xk = tl.arange(0, BK)
    valid = t < T
    ko = ((b * T + t[:, None]) * HQ + hq) * K + xk[None, :]
    kk = tl.load(kp + ko, mask=valid[:, None] & (xk[None, :] < K), other=0.).to(tl.float32)
    go = (b * T + t) * HV + h
    gg = tl.load(gp + go, mask=valid, other=0.).to(tl.float32)
    be = tl.load(bp + go, mask=valid, other=0.).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    rows = tl.arange(0, 16)[:, None]
    cols = tl.arange(0, 16)[None, :]
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    em = tl.exp(tl.minimum(cs[:, None] - cs[None, :], 0.0))
    m = be[:, None] * em * gram
    m = tl.where((rows > cols) & valid[:, None] & valid[None, :], m, 0.)
    inv = tl.zeros((16, 16), tl.float32)
    for i in range(16):
        row = tl.where(tl.arange(0, 16) == i, 1., 0.)
        for j in range(i):
            mij = tl.sum(tl.where((rows == i) & (cols == j), m, 0.))
            invj = tl.sum(tl.where(rows == j, inv, 0.), axis=0)
            row -= mij * invj
        inv = tl.where(rows == i, row[None, :], inv)
    inv = tl.where(valid[:, None] & valid[None, :], inv, 0.)
    base = be[:, None] * tl.exp(cs)[:, None] * kk
    ww = tl.dot(inv, base, input_precision="tf32")
    ao = (((b * HV + h) * NC + ic) * 16 + rows) * 16 + cols
    wo = ((b * T + t[:, None]) * HV + h) * K + xk[None, :]
    tl.store(ap + ao, inv)
    tl.store(wp + wo, ww, mask=valid[:, None] & (xk[None, :] < K))


@triton.jit
def _build_u(vp, bp, ap, up, T: tl.constexpr, HV: tl.constexpr,
             V: tl.constexpr, NC: tl.constexpr, BV_: tl.constexpr):
    p = tl.program_id(0)
    iv = p % tl.cdiv(V, BV_)
    z = p // tl.cdiv(V, BV_)
    ic = z % NC
    bh = z // NC
    b = bh // HV
    h = bh - b * HV
    t = ic * 16 + tl.arange(0, 16)
    y = iv * BV_ + tl.arange(0, BV_)
    valid = t < T
    vo = ((b * T + t[:, None]) * HV + h) * V + y[None, :]
    so = (b * T + t) * HV + h
    vv = tl.load(vp + vo, mask=valid[:, None] & (y[None, :] < V), other=0.).to(tl.float32)
    be = tl.load(bp + so, mask=valid, other=0.).to(tl.float32)
    rr = tl.arange(0, 16)[:, None]
    cc = tl.arange(0, 16)[None, :]
    ao = (((b * HV + h) * NC + ic) * 16 + rr) * 16 + cc
    aa = tl.load(ap + ao)
    uu = tl.dot(aa, be[:, None] * vv, input_precision="tf32")
    tl.store(up + vo, uu, mask=valid[:, None] & (y[None, :] < V))


@triton.jit
def _scan_state(kp, gp, wp, up, hp, xp, T: tl.constexpr, HQ: tl.constexpr,
                HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                NC: tl.constexpr, BK: tl.constexpr, BV_: tl.constexpr):
    p = tl.program_id(0)
    iv = p % tl.cdiv(V, BV_)
    bh = p // tl.cdiv(V, BV_)
    b = bh // HV
    h = bh - b * HV
    hq = h // (HV // HQ)
    rk = tl.arange(0, BK)[:, None]
    cv = iv * BV_ + tl.arange(0, BV_)[None, :]
    state = tl.zeros((BK, BV_), tl.float32)
    for ic in range(NC):
        t = ic * 16 + tl.arange(0, 16)
        valid = t < T
        hoff = ((((b * HV + h) * NC + ic) * K + rk) * V + cv).to(tl.int64)
        tl.store(hp + hoff, state, mask=(rk < K) & (cv < V))
        ko = ((b * T + t[:, None]) * HQ + hq) * K + tl.arange(0, BK)[None, :]
        wo = ((b * T + t[:, None]) * HV + h) * K + tl.arange(0, BK)[None, :]
        uo = ((b * T + t[:, None]) * HV + h) * V + cv
        kk = tl.load(kp + ko, mask=valid[:, None] & (tl.arange(0, BK)[None, :] < K), other=0.).to(tl.float32)
        ww = tl.load(wp + wo, mask=valid[:, None] & (tl.arange(0, BK)[None, :] < K), other=0.)
        uu = tl.load(up + uo, mask=valid[:, None] & (cv < V), other=0.)
        xx = uu - tl.dot(ww, state, input_precision="tf32")
        tl.store(xp + uo, xx, mask=valid[:, None] & (cv < V))
        go = (b * T + t) * HV + h
        cs = tl.cumsum(tl.load(gp + go, mask=valid, other=0.).to(tl.float32), axis=0)
        last = tl.sum(tl.where(valid & (tl.arange(0, 16) == tl.sum(valid.to(tl.int32), axis=0) - 1), cs, 0.), axis=0)
        rr = tl.exp(last - cs)[:, None] * kk
        state = tl.exp(last) * state + tl.dot(tl.trans(rr), xx, input_precision="tf32")


@triton.jit
def _emit(qp, kp, gp, hp, xp, op, T: tl.constexpr, HQ: tl.constexpr,
          HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
          BK: tl.constexpr, BV_: tl.constexpr):
    p = tl.program_id(0)
    iv = p % tl.cdiv(V, BV_)
    z = p // tl.cdiv(V, BV_)
    ic = z % NC
    bh = z // NC
    b = bh // HV
    h = bh - b * HV
    hq = h // (HV // HQ)
    t = ic * 16 + tl.arange(0, 16)
    kx = tl.arange(0, BK)
    vy = iv * BV_ + tl.arange(0, BV_)
    valid = t < T
    qo = ((b * T + t[:, None]) * HQ + hq) * K + kx[None, :]
    ko = qo
    q = tl.load(qp + qo, mask=valid[:, None] & (kx[None, :] < K), other=0.).to(tl.float32) * (1.0 / tl.sqrt(float(K)))
    k = tl.load(kp + ko, mask=valid[:, None] & (kx[None, :] < K), other=0.).to(tl.float32)
    go = (b * T + t) * HV + h
    cs = tl.cumsum(tl.load(gp + go, mask=valid, other=0.).to(tl.float32), axis=0)
    r = tl.arange(0, 16)[:, None]
    c = tl.arange(0, 16)[None, :]
    zmat = tl.exp(tl.minimum(cs[:, None] - cs[None, :], 0.0)) * tl.dot(q, tl.trans(k), input_precision="tf32")
    zmat = tl.where((r >= c) & valid[:, None] & valid[None, :], zmat, 0.)
    hoff = ((((b * HV + h) * NC + ic) * K + kx[:, None]) * V + vy[None, :]).to(tl.int64)
    xo = ((b * T + t[:, None]) * HV + h) * V + vy[None, :]
    state = tl.load(hp + hoff, mask=(kx[:, None] < K) & (vy[None, :] < V), other=0.)
    xx = tl.load(xp + xo, mask=valid[:, None] & (vy[None, :] < V), other=0.)
    out = tl.exp(cs)[:, None] * tl.dot(q, state, input_precision="tf32") + tl.dot(zmat, xx, input_precision="tf32")
    tl.store(op + xo, out, mask=valid[:, None] & (vy[None, :] < V))


@triton.jit
def _adjoint_scan(qp, kp, gp, bp, ap, xp, dop, dsp, T: tl.constexpr,
                  HQ: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                  V: tl.constexpr, NC: tl.constexpr, BK: tl.constexpr,
                  BV_: tl.constexpr):
    p = tl.program_id(0)
    iv = p % tl.cdiv(V, BV_)
    bh = p // tl.cdiv(V, BV_)
    b, h = bh // HV, bh % HV
    hq = h // (HV // HQ)
    x = tl.arange(0, BK)
    y = iv * BV_ + tl.arange(0, BV_)
    ds = tl.zeros((BK, BV_), tl.float32)
    scale = 1.0 / tl.sqrt(float(K))
    for rev in range(NC):
        ic = NC - 1 - rev
        t = ic * 16 + tl.arange(0, 16)
        valid = t < T
        qo = ((b * T + t[:, None]) * HQ + hq) * K + x[None, :]
        vo = ((b * T + t[:, None]) * HV + h) * V + y[None, :]
        so = (b * T + t) * HV + h
        qs = tl.load(qp + qo, mask=valid[:, None] & (x[None, :] < K), other=0.).to(tl.float32) * scale
        kk = tl.load(kp + qo, mask=valid[:, None] & (x[None, :] < K), other=0.).to(tl.float32)
        do = tl.load(dop + vo, mask=valid[:, None] & (y[None, :] < V), other=0.).to(tl.float32)
        xx = tl.load(xp + vo, mask=valid[:, None] & (y[None, :] < V), other=0.)
        be = tl.load(bp + so, mask=valid, other=0.).to(tl.float32)
        cs = tl.cumsum(tl.load(gp + so, mask=valid, other=0.).to(tl.float32), axis=0)
        nv = tl.sum(valid.to(tl.int32), axis=0)
        last = tl.sum(tl.where(valid & (tl.arange(0, 16) == nv - 1), cs, 0.), axis=0)
        row, col = tl.arange(0, 16)[:, None], tl.arange(0, 16)[None, :]
        lower = (row >= col) & valid[:, None] & valid[None, :]
        z = tl.where(lower, tl.exp(tl.minimum(cs[:, None] - cs[None, :], 0.0)) * tl.dot(qs, tl.trans(kk), input_precision="tf32"), 0.)
        dx = tl.dot(tl.trans(z), do, input_precision="tf32") + tl.dot(tl.exp(last - cs)[:, None] * kk, ds, input_precision="tf32")
        aa = tl.load(ap + (((b * HV + h) * NC + ic) * 16 + row) * 16 + col)
        dr = tl.dot(tl.trans(aa), dx, input_precision="tf32")
        dr = tl.where(valid[:, None] & (y[None, :] < V), dr, 0.)
        hs = ((((b * HV + h) * NC + ic) * K + x[:, None]) * V + y[None, :]).to(tl.int64)
        tl.store(dsp + hs, ds, mask=(x[:, None] < K) & (y[None, :] < V))
        ds = tl.exp(last) * ds + tl.dot(tl.trans(tl.exp(cs)[:, None] * qs), do, input_precision="tf32")
        ds -= tl.dot(tl.trans((be * tl.exp(cs))[:, None] * kk), dr, input_precision="tf32")


@triton.jit
def _chunk_grad(qp, kp, vp, gp, bp, ap, hp, xp, dop, dsp, pqp, pkp, dvp,
             pgp, pbp, T: tl.constexpr, HQ: tl.constexpr, HV: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, NC: tl.constexpr,
             BK: tl.constexpr, BV_: tl.constexpr):
    p = tl.program_id(0)
    iv = p % tl.cdiv(V, BV_)
    zpid = p // tl.cdiv(V, BV_)
    ic = zpid % NC
    bh = zpid // NC
    b = bh // HV
    h = bh - b * HV
    hq = h // (HV // HQ)
    kx = tl.arange(0, BK)
    vy = iv * BV_ + tl.arange(0, BV_)
    scale = 1.0 / tl.sqrt(float(K))
    t = ic * 16 + tl.arange(0, 16)
    valid = t < T
    qo = ((b * T + t[:, None]) * HQ + hq) * K + kx[None, :]
    vo = ((b * T + t[:, None]) * HV + h) * V + vy[None, :]
    so = (b * T + t) * HV + h
    q0 = tl.load(qp + qo, mask=valid[:, None] & (kx[None, :] < K), other=0.).to(tl.float32)
    qs = q0 * scale
    kk = tl.load(kp + qo, mask=valid[:, None] & (kx[None, :] < K), other=0.).to(tl.float32)
    vv = tl.load(vp + vo, mask=valid[:, None] & (vy[None, :] < V), other=0.).to(tl.float32)
    do = tl.load(dop + vo, mask=valid[:, None] & (vy[None, :] < V), other=0.).to(tl.float32)
    xx = tl.load(xp + vo, mask=valid[:, None] & (vy[None, :] < V), other=0.)
    be = tl.load(bp + so, mask=valid, other=0.).to(tl.float32)
    gg = tl.load(gp + so, mask=valid, other=0.).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    nvalid = tl.sum(valid.to(tl.int32), axis=0)
    last = tl.sum(tl.where(valid & (tl.arange(0, 16) == nvalid - 1), cs, 0.), axis=0)
    row = tl.arange(0, 16)[:, None]
    col = tl.arange(0, 16)[None, :]
    lower = (row >= col) & valid[:, None] & valid[None, :]
    estr = tl.exp(tl.minimum(cs[:, None] - cs[None, :], 0.0))
    zmat = tl.where(lower, estr * tl.dot(qs, tl.trans(kk), input_precision="tf32"), 0.)
    hoff = ((((b * HV + h) * NC + ic) * K + kx[:, None]) * V + vy[None, :]).to(tl.int64)
    state = tl.load(hp + hoff, mask=(kx[:, None] < K) & (vy[None, :] < V), other=0.)
    ds = tl.load(dsp + hoff, mask=(kx[:, None] < K) & (vy[None, :] < V), other=0.)
    rr = tl.exp(last - cs)[:, None] * kk
    dx = tl.dot(tl.trans(zmat), do, input_precision="tf32") + tl.dot(rr, ds, input_precision="tf32")
    ar = tl.arange(0, 16)[:, None]
    ac = tl.arange(0, 16)[None, :]
    ao = (((b * HV + h) * NC + ic) * 16 + ar) * 16 + ac
    aa = tl.load(ap + ao)
    dr = tl.dot(tl.trans(aa), dx, input_precision="tf32")
    dr = tl.where(valid[:, None] & (vy[None, :] < V), dr, 0.)
    xx = tl.where(valid[:, None] & (vy[None, :] < V), xx, 0.)
    dm = -tl.dot(dr, tl.trans(xx), input_precision="tf32")
    dm = tl.where((row > col) & valid[:, None] & valid[None, :], dm, 0.)
    dproj = tl.dot(do, tl.trans(xx), input_precision="tf32")
    pair = tl.where(lower, estr * dproj, 0.)
    dq = scale * (tl.exp(cs)[:, None] * tl.dot(do, tl.trans(state), input_precision="tf32") + tl.dot(pair, kk, input_precision="tf32"))
    dk = tl.dot(tl.trans(pair), qs, input_precision="tf32")
    kds = tl.dot(kk, ds, input_precision="tf32")
    dk += tl.exp(last - cs)[:, None] * tl.dot(xx, tl.trans(ds), input_precision="tf32")
    pred = tl.dot(kk, state, input_precision="tf32")
    dk -= (be * tl.exp(cs))[:, None] * tl.dot(dr, tl.trans(state), input_precision="tf32")
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    nm = tl.where(row > col, dm * estr * be[:, None], 0.)
    dk += tl.dot(nm, kk, input_precision="tf32") + tl.dot(tl.trans(nm), kk, input_precision="tf32")
    dv = be[:, None] * dr
    db = tl.sum(dr * (vv - tl.exp(cs)[:, None] * pred), axis=1)
    db += tl.sum(tl.where(row > col, dm * estr * gram, 0.), axis=1)
    f = tl.where(lower, pair * tl.dot(qs, tl.trans(kk), input_precision="tf32"), 0.)
    dc = tl.sum(f, axis=1) - tl.sum(f, axis=0)
    dc += tl.exp(cs) * tl.sum(do * tl.dot(qs, state, input_precision="tf32"), axis=1)
    rg = tl.exp(last - cs) * tl.sum(kds * xx, axis=1)
    dc -= rg
    dc += tl.where(tl.arange(0, 16) == nvalid - 1, tl.sum(rg, axis=0), 0.)
    rp = -be * tl.exp(cs) * tl.sum(dr * pred, axis=1)
    dc += rp
    gm = dm * (be[:, None] * estr * gram)
    dc += tl.sum(gm, axis=1) - tl.sum(gm, axis=0)
    endg = tl.exp(last) * tl.sum(tl.sum(ds * state, axis=1), axis=0)
    dc += tl.where(tl.arange(0, 16) == nvalid - 1, endg, 0.)
    dg = tl.flip(tl.cumsum(tl.flip(dc, dim=0), axis=0), dim=0)
    base = ((((b * T + t[:, None]) * HV + h) * tl.cdiv(V, BV_) + iv) * BK + kx[None, :]).to(tl.int64)
    tl.store(pqp + base, dq, mask=valid[:, None] & (kx[None, :] < K))
    tl.store(pkp + base, dk, mask=valid[:, None] & (kx[None, :] < K))
    tl.store(dvp + vo, dv, mask=valid[:, None] & (vy[None, :] < V))
    sb = (((b * T + t) * HV + h) * tl.cdiv(V, BV_) + iv).to(tl.int64)
    tl.store(pgp + sb, dg, mask=valid)
    tl.store(pbp + sb, db, mask=valid)



@triton.jit
def _sum_qk(pq, pk, dq, dk, N: tl.constexpr, HQ: tl.constexpr,
            HV: tl.constexpr, K: tl.constexpr, NV: tl.constexpr,
            BK: tl.constexpr):
    n = tl.program_id(0)
    bth = n // HQ
    hq = n - bth * HQ
    x = tl.arange(0, BK)
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    rep = HV // HQ
    for r in range(rep):
        hv = hq * rep + r
        for iv in range(NV):
            off = (((bth * HV + hv) * NV + iv) * BK + x).to(tl.int64)
            aq += tl.load(pq + off, mask=x < K, other=0.)
            ak += tl.load(pk + off, mask=x < K, other=0.)
    out = (bth * HQ + hq) * K + x
    tl.store(dq + out, aq, mask=x < K)
    tl.store(dk + out, ak, mask=x < K)


@triton.jit
def _sum_scalar(pg, pb, dg, db, N: tl.constexpr, NV: tl.constexpr):
    n = tl.program_id(0)
    ag = 0.0
    ab = 0.0
    for iv in range(NV):
        ag += tl.load(pg + n * NV + iv)
        ab += tl.load(pb + n * NV + iv)
    tl.store(dg + n, ag)
    tl.store(db + n, ab)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, HQ, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        NC, NV = triton.cdiv(T, C), triton.cdiv(V, BV)
        BK = triton.next_power_of_2(K)
        A = torch.empty((B, HV, NC, C, C), device=q.device, dtype=torch.float32)
        W = torch.empty((B, T, HV, K), device=q.device, dtype=torch.float32)
        U = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        X = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        H = torch.empty((B, HV, NC, K, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_wy[(NC, B * HV)](k, g, beta, A, W, T, HQ, HV, K, NC, BK, num_warps=4)
        _build_u[(B * HV * NC * NV,)](v, beta, A, U, T, HV, V, NC, BV, num_warps=4)
        _scan_state[(B * HV * NV,)](k, g, W, U, H, X, T, HQ, HV, K, V, NC, BK, BV, num_warps=4)
        _emit[(B * HV * NC * NV,)](q, k, g, H, X, out, T, HQ, HV, K, V, NC, BK, BV, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, A, H, X)
        ctx.meta = (B, T, HQ, HV, K, V, NC, NV, BK)
        return out

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        q, k, v, g, beta, A, H, X = ctx.saved_tensors
        B, T, HQ, HV, K, V, NC, NV, BK = ctx.meta
        pq = torch.empty((B, T, HV, NV, BK), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq)
        pg = torch.empty((B, T, HV, NV), device=q.device, dtype=torch.float32)
        pb = torch.empty_like(pg)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dg = torch.empty_like(g)
        db = torch.empty_like(beta)
        DS = torch.empty((B, HV, NC, K, V), device=q.device, dtype=torch.float32)
        _adjoint_scan[(B * HV * NV,)](q, k, g, beta, A, X, do, DS,
                                      T, HQ, HV, K, V, NC, BK, BV, num_warps=4)
        _chunk_grad[(B * HV * NC * NV,)](q, k, v, g, beta, A, H, X, do, DS,
                                         pq, pk, dv, pg, pb, T, HQ, HV, K, V,
                                         NC, BK, BV, num_warps=4)
        _sum_qk[(B * T * HQ,)](pq, pk, dq, dk, B * T * HQ, HQ, HV, K, NV, BK, num_warps=4)
        _sum_scalar[(B * T * HV,)](pg, pb, dg, db, B * T * HV, NV, num_warps=4)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
