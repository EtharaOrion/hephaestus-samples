
import torch
import triton
import triton.language as tl

C = 16
BV = 16


@triton.jit
def _make_inverse(kp, gp, bp, ap, T: tl.constexpr, K: tl.constexpr,
                  H: tl.constexpr, HV: tl.constexpr, BK: tl.constexpr):
    c = tl.program_id(0)
    xh = tl.program_id(1)
    b = xh // HV
    h = xh - b * HV
    kh = h // (HV // H)
    it = c * 16 + tl.arange(0, 16)
    ik = tl.arange(0, BK)
    mt = it < T
    k = tl.load(kp + ((b * T + it[:, None]) * H + kh) * K + ik[None, :],
                mask=mt[:, None] & (ik[None, :] < K), other=0.).to(tl.float32)
    g = tl.load(gp + (b * T + it) * HV + h, mask=mt, other=0.).to(tl.float32)
    beta = tl.load(bp + (b * T + it) * HV + h, mask=mt, other=0.).to(tl.float32)
    p = tl.cumsum(g, axis=0)
    rows = tl.arange(0, 16)[:, None]
    cols = tl.arange(0, 16)[None, :]
    e = tl.exp(tl.minimum(p[:, None] - p[None, :], 0.0))
    gram = tl.dot(k, tl.trans(k), input_precision="tf32")
    lower = rows > cols
    m = tl.where(lower & mt[:, None] & mt[None, :], beta[:, None] * e * gram, 0.)
    inv = tl.where(rows == cols, 1., 0.)
    for i in tl.static_range(1, 16):
        mi = tl.sum(tl.where(rows == i, m, 0.), axis=0)
        vals = -tl.sum(mi[:, None] * inv, axis=0)
        vals += tl.arange(0, 16) == i
        inv = tl.where(rows == i, vals[None, :], inv)
    base = ((b * ((T + 16 - 1) // 16) + c) * HV + h) * 16 * 16
    tl.store(ap + base + rows * 16 + cols, inv,
             mask=mt[:, None] & mt[None, :])


@triton.jit
def _scan_state(kp, vp, gp, bp, ap, sp, rp,
                T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                H: tl.constexpr, HV: tl.constexpr, NC: tl.constexpr,
                BK: tl.constexpr, BV_: tl.constexpr):
    z = tl.program_id(0)
    ivb = z % ((V + BV_ - 1) // BV_)
    xh = z // ((V + BV_ - 1) // BV_)
    b = xh // HV
    h = xh - b * HV
    kh = h // (HV // H)
    ik = tl.arange(0, BK)
    iv = ivb * BV_ + tl.arange(0, BV_)
    mk = ik < K
    mv = iv < V
    state = tl.zeros((BK, BV_), tl.float32)
    rr = tl.arange(0, 16)[:, None]
    cc = tl.arange(0, 16)[None, :]
    for c in range(NC):
        it = c * 16 + tl.arange(0, 16)
        mt = it < T
        sb = ((((b * NC + c) * HV + h) * K + ik[:, None]) * V + iv[None, :]).to(tl.int64)
        tl.store(sp + sb, state, mask=mk[:, None] & mv[None, :])
        k = tl.load(kp + ((b * T + it[:, None]) * H + kh) * K + ik[None, :],
                    mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
        vv = tl.load(vp + ((b * T + it[:, None]) * HV + h) * V + iv[None, :],
                     mask=mt[:, None] & mv[None, :], other=0.).to(tl.float32)
        g = tl.load(gp + (b * T + it) * HV + h, mask=mt, other=0.).to(tl.float32)
        beta = tl.load(bp + (b * T + it) * HV + h, mask=mt, other=0.).to(tl.float32)
        p = tl.cumsum(g, axis=0)
        a = tl.exp(p)
        abase = ((b * NC + c) * HV + h) * 16 * 16
        inv = tl.load(ap + abase + rr * 16 + cc,
                      mask=mt[:, None] & mt[None, :], other=0.)
        pred = tl.dot(k, state, input_precision="tf32")
        rhs = beta[:, None] * (vv - a[:, None] * pred)
        r = tl.dot(inv, rhs, input_precision="tf32")
        rb = (((b * T + it[:, None]) * HV + h) * V + iv[None, :]).to(tl.int64)
        tl.store(rp + rb, r, mask=mt[:, None] & mv[None, :])
        last = tl.minimum(16 - 1, T - c * 16 - 1)
        pe = tl.sum(tl.where(tl.arange(0, 16) == last, p, 0.), axis=0)
        f = tl.exp(pe - p)
        state = tl.exp(pe) * state + tl.dot(tl.trans(k), f[:, None] * r, input_precision="tf32")


@triton.jit
def _chunk_output(qp, kp, gp, sp, rp, op,
                  T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  H: tl.constexpr, HV: tl.constexpr, NC: tl.constexpr,
                  BK: tl.constexpr, BV_: tl.constexpr, SCALE: tl.constexpr):
    ivb = tl.program_id(0)
    c = tl.program_id(1)
    xh = tl.program_id(2)
    b = xh // HV
    h = xh - b * HV
    kh = h // (HV // H)
    it = c * 16 + tl.arange(0, 16)
    ik = tl.arange(0, BK)
    iv = ivb * BV_ + tl.arange(0, BV_)
    mt, mk, mv = it < T, ik < K, iv < V
    q = tl.load(qp + ((b * T + it[:, None]) * H + kh) * K + ik[None, :],
                mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32) * SCALE
    k = tl.load(kp + ((b * T + it[:, None]) * H + kh) * K + ik[None, :],
                mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    g = tl.load(gp + (b * T + it) * HV + h, mask=mt, other=0.).to(tl.float32)
    p = tl.cumsum(g, axis=0)
    rows = tl.arange(0, 16)[:, None]
    cols = tl.arange(0, 16)[None, :]
    e = tl.exp(tl.minimum(p[:, None] - p[None, :], 0.0))
    d = tl.where((rows >= cols) & mt[:, None] & mt[None, :],
                 e * tl.dot(q, tl.trans(k), input_precision="tf32"), 0.)
    sb = ((((b * NC + c) * HV + h) * K + ik[:, None]) * V + iv[None, :]).to(tl.int64)
    state = tl.load(sp + sb, mask=mk[:, None] & mv[None, :], other=0.)
    rb = (((b * T + it[:, None]) * HV + h) * V + iv[None, :]).to(tl.int64)
    r = tl.load(rp + rb, mask=mt[:, None] & mv[None, :], other=0.)
    out = tl.exp(p)[:, None] * tl.dot(q, state, input_precision="tf32") + tl.dot(d, r, input_precision="tf32")
    tl.store(op + rb, out, mask=mt[:, None] & mv[None, :])




@triton.jit
def _scan_adjoint(qp, kp, gp, bp, ap, dop, hp,
                  T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  H: tl.constexpr, HV: tl.constexpr, NC: tl.constexpr,
                  BK: tl.constexpr, BV_: tl.constexpr, SCALE: tl.constexpr):
    z = tl.program_id(0)
    nv = (V + BV_ - 1) // BV_
    ivb = z % nv
    xh = z // nv
    b = xh // HV
    h = xh - b * HV
    kh = h // (HV // H)
    ik = tl.arange(0, BK)
    iv = ivb * BV_ + tl.arange(0, BV_)
    mk, mv = ik < K, iv < V
    dh = tl.zeros((BK, BV_), tl.float32)
    rows = tl.arange(0, 16)[:, None]
    cols = tl.arange(0, 16)[None, :]
    for cr in range(NC):
        c = NC - 1 - cr
        hb = ((((b * NC + c) * HV + h) * K + ik[:, None]) * V + iv[None, :]).to(tl.int64)
        tl.store(hp + hb, dh, mask=mk[:, None] & mv[None, :])
        it = c * 16 + tl.arange(0, 16)
        mt = it < T
        q = tl.load(qp + ((b * T + it[:, None]) * H + kh) * K + ik[None, :], mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32) * SCALE
        k = tl.load(kp + ((b * T + it[:, None]) * H + kh) * K + ik[None, :], mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
        g = tl.load(gp + (b * T + it) * HV + h, mask=mt, other=0.).to(tl.float32)
        beta = tl.load(bp + (b * T + it) * HV + h, mask=mt, other=0.).to(tl.float32)
        rb = (((b * T + it[:, None]) * HV + h) * V + iv[None, :]).to(tl.int64)
        do = tl.load(dop + rb, mask=mt[:, None] & mv[None, :], other=0.).to(tl.float32)
        p = tl.cumsum(g, axis=0)
        aa = tl.exp(p)
        last = tl.minimum(15, T - c * 16 - 1)
        pe = tl.sum(tl.where(tl.arange(0, 16) == last, p, 0.), axis=0)
        f = tl.exp(pe - p)
        dmat = tl.where((rows >= cols) & mt[:, None] & mt[None, :],
                        tl.exp(tl.minimum(p[:, None] - p[None, :], 0.0)) * tl.dot(q, tl.trans(k), input_precision="tf32"), 0.)
        dr = tl.dot(tl.trans(dmat), do, input_precision="tf32") + f[:, None] * tl.dot(k, dh, input_precision="tf32")
        abase = ((b * NC + c) * HV + h) * 16 * 16
        inv = tl.load(ap + abase + rows * 16 + cols, mask=mt[:, None] & mt[None, :], other=0.)
        dz = tl.dot(tl.trans(inv), dr, input_precision="tf32")
        dh = tl.dot(tl.trans(aa[:, None] * q), do, input_precision="tf32") + tl.exp(pe) * dh
        dh -= tl.dot(tl.trans((beta * aa)[:, None] * k), dz, input_precision="tf32")

@triton.jit
def _reverse_chunks(qp, kp, vp, gp, bp, ap, sp, rp, dop, hp,
                    dqpp, dkpp, dvp, dgpp, dbpp,
                    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    H: tl.constexpr, HV: tl.constexpr, NC: tl.constexpr,
                    NV: tl.constexpr, BK: tl.constexpr, BV_: tl.constexpr,
                    SCALE: tl.constexpr):
    ivb = tl.program_id(0)
    c = tl.program_id(1)
    xh = tl.program_id(2)
    b = xh // HV
    h = xh - b * HV
    kh = h // (HV // H)
    ik = tl.arange(0, BK)
    iv = ivb * BV_ + tl.arange(0, BV_)
    mk, mv = ik < K, iv < V
    hb = ((((b * NC + c) * HV + h) * K + ik[:, None]) * V + iv[None, :]).to(tl.int64)
    dh = tl.load(hp + hb, mask=mk[:, None] & mv[None, :], other=0.)
    rows = tl.arange(0, 16)[:, None]
    cols = tl.arange(0, 16)[None, :]
    it = c * 16 + tl.arange(0, 16)
    mt = it < T
    q = tl.load(qp + ((b * T + it[:, None]) * H + kh) * K + ik[None, :], mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32) * SCALE
    k = tl.load(kp + ((b * T + it[:, None]) * H + kh) * K + ik[None, :], mask=mt[:, None] & mk[None, :], other=0.).to(tl.float32)
    vv = tl.load(vp + ((b * T + it[:, None]) * HV + h) * V + iv[None, :], mask=mt[:, None] & mv[None, :], other=0.).to(tl.float32)
    g = tl.load(gp + (b * T + it) * HV + h, mask=mt, other=0.).to(tl.float32)
    beta = tl.load(bp + (b * T + it) * HV + h, mask=mt, other=0.).to(tl.float32)
    p = tl.cumsum(g, axis=0)
    aa = tl.exp(p)
    e = tl.exp(tl.minimum(p[:, None] - p[None, :], 0.0))
    lower_eq = (rows >= cols) & mt[:, None] & mt[None, :]
    lower = (rows > cols) & mt[:, None] & mt[None, :]
    qk = tl.dot(q, tl.trans(k), input_precision="tf32")
    dmat = tl.where(lower_eq, e * qk, 0.)
    sb = ((((b * NC + c) * HV + h) * K + ik[:, None]) * V + iv[None, :]).to(tl.int64)
    state = tl.load(sp + sb, mask=mk[:, None] & mv[None, :], other=0.)
    rb = (((b * T + it[:, None]) * HV + h) * V + iv[None, :]).to(tl.int64)
    r = tl.load(rp + rb, mask=mt[:, None] & mv[None, :], other=0.)
    do = tl.load(dop + rb, mask=mt[:, None] & mv[None, :], other=0.).to(tl.float32)
    last = tl.minimum(16 - 1, T - c * 16 - 1)
    pe = tl.sum(tl.where(tl.arange(0, 16) == last, p, 0.), axis=0)
    ae = tl.exp(pe)
    f = tl.exp(pe - p)

    ds = tl.dot(tl.trans(aa[:, None] * q), do, input_precision="tf32") + ae * dh
    dr = tl.dot(tl.trans(dmat), do, input_precision="tf32") + f[:, None] * tl.dot(k, dh, input_precision="tf32")
    dq = aa[:, None] * tl.dot(do, tl.trans(state), input_precision="tf32")
    dd = tl.dot(do, tl.trans(r), input_precision="tf32")
    de = tl.where(lower_eq, dd * qk, 0.)
    dq += tl.dot(tl.where(lower_eq, dd * e, 0.), k, input_precision="tf32")
    dk = tl.dot(tl.trans(tl.where(lower_eq, dd * e, 0.)), q, input_precision="tf32")
    dk += f[:, None] * tl.dot(r, tl.trans(dh), input_precision="tf32")
    dp = aa * tl.sum(tl.dot(q, state, input_precision="tf32") * do, axis=1)
    df = tl.sum(tl.dot(k, dh, input_precision="tf32") * r, axis=1)
    dp += tl.sum(de * e, axis=1) - tl.sum(de * e, axis=0)
    dp += tl.where(tl.arange(0, 16) == last, ae * tl.sum(state * dh), 0.)
    dp += tl.where(tl.arange(0, 16) == last, tl.sum(df * f), 0.) - df * f

    abase = ((b * NC + c) * HV + h) * 16 * 16
    inv = tl.load(ap + abase + rows * 16 + cols, mask=mt[:, None] & mt[None, :], other=0.)
    dz = tl.dot(tl.trans(inv), dr, input_precision="tf32")
    dm = -tl.dot(dz, tl.trans(r), input_precision="tf32")
    dm = tl.where(lower, dm, 0.)
    pred = tl.dot(k, state, input_precision="tf32")
    dv = beta[:, None] * dz
    db = tl.sum(dz * (vv - aa[:, None] * pred), axis=1)
    dk -= (beta * aa)[:, None] * tl.dot(dz, tl.trans(state), input_precision="tf32")
    ds -= tl.dot(tl.trans((beta * aa)[:, None] * k), dz, input_precision="tf32")
    dp -= beta * aa * tl.sum(dz * pred, axis=1)
    gram = tl.dot(k, tl.trans(k), input_precision="tf32")
    db += tl.sum(dm * e * gram, axis=1)
    dgram = dm * beta[:, None] * e
    dk += tl.dot(dgram, k, input_precision="tf32") + tl.dot(tl.trans(dgram), k, input_precision="tf32")
    deme = dm * beta[:, None] * gram * e
    dp += tl.sum(deme, axis=1) - tl.sum(deme, axis=0)
    dg = tl.cumsum(dp + 0.0, axis=0, reverse=True)

    part = ((((b * T + it[:, None]) * HV + h) * NV + ivb) * K + ik[None, :]).to(tl.int64)
    tl.store(dqpp + part, dq * SCALE, mask=mt[:, None] & mk[None, :])
    tl.store(dkpp + part, dk, mask=mt[:, None] & mk[None, :])
    tl.store(dvp + rb, dv, mask=mt[:, None] & mv[None, :])
    ps = (((b * T + it) * HV + h) * NV + ivb).to(tl.int64)
    tl.store(dgpp + ps, dg, mask=mt)
    tl.store(dbpp + ps, db, mask=mt)

@triton.jit
def _reduce_qk(dqpp, dkpp, dqp, dkp, T: tl.constexpr, K: tl.constexpr,
               H: tl.constexpr, HV: tl.constexpr, NV: tl.constexpr,
               BK: tl.constexpr):
    t = tl.program_id(0)
    x = tl.program_id(1)
    b = x // H
    h = x - b * H
    ik = tl.arange(0, BK)
    accq = tl.zeros((BK,), tl.float32)
    acck = tl.zeros((BK,), tl.float32)
    rep = HV // H
    for r in range(rep):
        vh = h * rep + r
        for j in range(NV):
            off = ((((b * T + t) * HV + vh) * NV + j) * K + ik).to(tl.int64)
            accq += tl.load(dqpp + off, mask=ik < K, other=0.)
            acck += tl.load(dkpp + off, mask=ik < K, other=0.)
    out = ((b * T + t) * H + h) * K + ik
    tl.store(dqp + out, accq, mask=ik < K)
    tl.store(dkp + out, acck, mask=ik < K)


@triton.jit
def _reduce_scalars(dgpp, dbpp, dgp, dbp, T: tl.constexpr,
                    HV: tl.constexpr, NV: tl.constexpr):
    t = tl.program_id(0)
    xh = tl.program_id(1)
    b = xh // HV
    h = xh - b * HV
    ag = 0.0
    ab = 0.0
    for j in range(NV):
        off = ((b * T + t) * HV + h) * NV + j
        ag += tl.load(dgpp + off)
        ab += tl.load(dbpp + off)
    out = (b * T + t) * HV + h
    tl.store(dgp + out, ag)
    tl.store(dbp + out, ab)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        NC, NV = triton.cdiv(T, C), triton.cdiv(V, BV)
        BK = triton.next_power_of_2(K)
        a = torch.empty((B, NC, HV, C, C), device=q.device, dtype=torch.float32)
        s = torch.empty((B, NC, HV, K, V), device=q.device, dtype=torch.float32)
        r = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _make_inverse[(NC, B * HV)](k, g, beta, a, T, K, H, HV, BK)
        _scan_state[(B * HV * NV,)](k, v, g, beta, a, s, r, T, K, V, H, HV, NC, BK, BV)
        _chunk_output[(NV, NC, B * HV)](q, k, g, s, r, out, T, K, V, H, HV, NC, BK, BV, K ** -0.5)
        ctx.save_for_backward(q, k, v, g, beta, a, s, r)
        ctx.dims = (B, T, H, K, HV, V, NC, NV, BK)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, a, s, r = ctx.saved_tensors
        do = do.contiguous()
        B, T, H, K, HV, V, NC, NV, BK = ctx.dims
        dqpart = torch.empty((B, T, HV, NV, K), device=q.device, dtype=torch.float32)
        dkpart = torch.empty((B, T, HV, NV, K), device=q.device, dtype=torch.float32)
        dgpart = torch.empty((B, T, HV, NV), device=q.device, dtype=torch.float32)
        dbpart = torch.empty((B, T, HV, NV), device=q.device, dtype=torch.float32)
        hpart = torch.empty((B, NC, HV, K, V), device=q.device, dtype=torch.float32)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        dv = torch.empty_like(v)
        dg, db = torch.empty_like(g), torch.empty_like(beta)
        _scan_adjoint[(B * HV * NV,)](q, k, g, beta, a, do, hpart,
            T, K, V, H, HV, NC, BK, BV, K ** -0.5)
        _reverse_chunks[(NV, NC, B * HV)](q, k, v, g, beta, a, s, r, do, hpart,
            dqpart, dkpart, dv, dgpart, dbpart, T, K, V, H, HV, NC, NV, BK, BV, K ** -0.5)
        _reduce_qk[(T, B * H)](dqpart, dkpart, dq, dk, T, K, H, HV, NV, BK)
        _reduce_scalars[(T, B * HV)](dgpart, dbpart, dg, db, T, HV, NV)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
