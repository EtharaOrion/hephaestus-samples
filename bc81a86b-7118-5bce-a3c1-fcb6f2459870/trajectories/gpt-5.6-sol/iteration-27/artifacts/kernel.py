import torch
import triton
import triton.language as tl

BC = 16
BV = 32


@triton.jit
def _build_wy(qp, kp, gp, bp, Ap, Pp, Gp, T: tl.constexpr, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
              NC: tl.constexpr, SCALE: tl.constexpr):
    c = tl.program_id(0)
    z = tl.program_id(1).to(tl.int64)
    b, hv = z // HV, z % HV
    h = hv // (HV // H)
    r = tl.arange(0, 16)
    x = tl.arange(0, BK)
    t = c * 16 + r
    valid = t < T
    ko = (((b * T + t[:, None]) * H + h) * K + x[None, :])
    k = tl.load(kp + ko, mask=valid[:, None] & (x[None, :] < K), other=0.0)
    q = tl.load(qp + ko, mask=valid[:, None] & (x[None, :] < K), other=0.0)
    g = tl.load(gp + (b * T + t) * HV + hv, mask=valid, other=0.0).to(tl.float32)
    be = tl.load(bp + (b * T + t) * HV + hv, mask=valid, other=0.0).to(tl.float32)
    G = tl.cumsum(g, axis=0)
    rr, ss = r[:, None], r[None, :]
    de = tl.minimum(G[:, None] - G[None, :], 0.0)
    gram = tl.dot(k.to(tl.float32), tl.trans(k.to(tl.float32)), input_precision="tf32")
    M = tl.where((rr > ss) & valid[:, None] & valid[None, :], be[:, None] * tl.exp(de) * gram, 0.0)
    A = tl.where(rr == ss, 1.0, 0.0)
    for i in range(1, 16):
        row = -tl.sum(tl.where((rr == i) & (ss < i), M, 0.0)[:, :, None] * A[None, :, :], axis=1)
        A = tl.where(rr == i, row + (ss == i), A)
    ao = (((z * NC + c) * 16 + rr) * 16 + ss)
    P = tl.where((rr >= ss) & valid[:, None] & valid[None, :], tl.exp(de) * tl.dot(q.to(tl.float32) * SCALE, tl.trans(k.to(tl.float32)), input_precision="tf32"), 0.0)
    tl.store(Ap + ao, A)
    tl.store(Pp + ao, P)
    tl.store(Gp + (b * T + t) * HV + hv, G, mask=valid)


@triton.jit
def _scan_state(kp, vp, bp, Ap, Gp, Sp, Dp,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                NC: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv, z = p % NV, p // NV
    b, hv = z // HV, z % HV
    h = hv // (HV // H)
    x = tl.arange(0, BK)
    yy = tl.arange(0, 32)
    y = iv * 32 + yy
    r = tl.arange(0, 16)
    rr, ss = r[:, None], r[None, :]
    state = tl.zeros((BK, 32), tl.float32)
    for c in range(NC):
        t = c * 16 + r
        valid = t < T
        so = ((((z * NC + c) * K + x[:, None]) * NV + iv) * 32 + yy[None, :])
        tl.store(Sp + so, state, mask=(x[:, None] < K) & (y[None, :] < V))
        ko = (((b * T + t[:, None]) * H + h) * K + x[None, :])
        vo = (((b * T + t[:, None]) * HV + hv) * V + y[None, :])
        k = tl.load(kp + ko, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        v = tl.load(vp + vo, mask=valid[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
        be = tl.load(bp + (b * T + t) * HV + hv, mask=valid, other=0.0).to(tl.float32)
        G = tl.load(Gp + (b * T + t) * HV + hv, mask=valid, other=0.0)
        A = tl.load(Ap + (((z * NC + c) * 16 + rr) * 16 + ss))
        pred = tl.dot(k, state, input_precision="tf32")
        rhs = be[:, None] * (v - tl.exp(tl.minimum(G, 0.0))[:, None] * pred)
        d = tl.dot(A, rhs, input_precision="tf32")
        tl.store(Dp + vo, d, mask=valid[:, None] & (y[None, :] < V))
        last = tl.maximum(tl.minimum(T - c * 16, 16), 1) - 1
        Ge = tl.sum(tl.where(r == last, G, 0.0), axis=0)
        ek = tl.exp(tl.minimum(Ge - G, 0.0))[:, None] * k
        state = tl.exp(tl.minimum(Ge, 0.0)) * state + tl.dot(tl.trans(ek), d, input_precision="tf32")


@triton.jit
def _emit(qp, Pp, Gp, Sp, Dp, op,
          T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
          K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
          NC: tl.constexpr, NV: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv, p = p % NV, p // NV
    c, z = p % NC, p // NC
    b, hv = z // HV, z % HV
    h = hv // (HV // H)
    r = tl.arange(0, 16)
    x = tl.arange(0, BK)
    yy = tl.arange(0, 32)
    y = iv * 32 + yy
    t = c * 16 + r
    valid = t < T
    qo = (((b * T + t[:, None]) * H + h) * K + x[None, :])
    q = tl.load(qp + qo, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * SCALE
    G = tl.load(Gp + (b * T + t) * HV + hv, mask=valid, other=0.0)
    state = tl.load(Sp + ((((z * NC + c) * K + x[:, None]) * NV + iv) * 32 + yy[None, :]), mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    d = tl.load(Dp + (((b * T + t[:, None]) * HV + hv) * V + y[None, :]), mask=valid[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
    rr, ss = r[:, None], r[None, :]
    P = tl.load(Pp + (((z * NC + c) * 16 + rr) * 16 + ss))
    out = tl.exp(tl.minimum(G, 0.0))[:, None] * tl.dot(q, state, input_precision="tf32") + tl.dot(P, d, input_precision="tf32")
    tl.store(op + (((b * T + t[:, None]) * HV + hv) * V + y[None, :]), out, mask=valid[:, None] & (y[None, :] < V))


@triton.jit
def _adjoint(qp, kp, bp, Gp, Ap, Pp, dop, Ep,
             T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
             NC: tl.constexpr, NV: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv, z = p % NV, p // NV
    b, hv = z // HV, z % HV
    h = hv // (HV // H)
    x = tl.arange(0, BK)
    yy = tl.arange(0, 32)
    y = iv * 32 + yy
    r = tl.arange(0, 16)
    ds = tl.zeros((BK, 32), tl.float32)
    for ci in range(NC):
        c = NC - 1 - ci
        t = c * 16 + r
        valid = t < T
        tl.store(Ep + ((((z * NC + c) * K + x[:, None]) * NV + iv) * 32 + yy[None, :]), ds, mask=(x[:, None] < K) & (y[None, :] < V))
        qko = (((b * T + t[:, None]) * H + h) * K + x[None, :])
        q = tl.load(qp + qko, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * SCALE
        k = tl.load(kp + qko, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        be = tl.load(bp + (b * T + t) * HV + hv, mask=valid, other=0.0).to(tl.float32)
        G = tl.load(Gp + (b * T + t) * HV + hv, mask=valid, other=0.0)
        do = tl.load(dop + (((b * T + t[:, None]) * HV + hv) * V + y[None, :]), mask=valid[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
        rr, ss = r[:, None], r[None, :]
        A = tl.load(Ap + (((z * NC + c) * 16 + rr) * 16 + ss))
        P = tl.load(Pp + (((z * NC + c) * 16 + rr) * 16 + ss))
        last = tl.maximum(tl.minimum(T - c * 16, 16), 1) - 1
        Ge = tl.sum(tl.where(r == last, G, 0.0), axis=0)
        kb = tl.exp(tl.minimum(Ge - G, 0.0))[:, None] * k
        dbar = tl.dot(tl.trans(P), do, input_precision="tf32") + tl.dot(kb, ds, input_precision="tf32")
        X = tl.dot(tl.trans(A), dbar, input_precision="tf32")
        R = be[:, None] * tl.exp(tl.minimum(G, 0.0))[:, None] * k
        ds = tl.exp(tl.minimum(Ge, 0.0)) * ds + tl.dot(tl.trans(tl.exp(tl.minimum(G, 0.0))[:, None] * q), do, input_precision="tf32") - tl.dot(tl.trans(R), X, input_precision="tf32")


@triton.jit
def _chunk_grad(qp, kp, vp, bp, Gp, Ap, Pp, Sp, Dp, dop, Ep, Qp, Kp, Vp, gp, Bp,
                T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                NC: tl.constexpr, NV: tl.constexpr, SCALE: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    iv, p = p % NV, p // NV
    c, z = p % NC, p // NC
    b, hv = z // HV, z % HV
    h = hv // (HV // H)
    r = tl.arange(0, 16)
    x = tl.arange(0, BK)
    yy = tl.arange(0, 32)
    y = iv * 32 + yy
    t = c * 16 + r
    valid = t < T
    qko = (((b * T + t[:, None]) * H + h) * K + x[None, :])
    q = tl.load(qp + qko, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * SCALE
    k = tl.load(kp + qko, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    vo = (((b * T + t[:, None]) * HV + hv) * V + y[None, :])
    v = tl.load(vp + vo, mask=valid[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
    d = tl.load(Dp + vo, mask=valid[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
    do = tl.load(dop + vo, mask=valid[:, None] & (y[None, :] < V), other=0.0).to(tl.float32)
    be = tl.load(bp + (b * T + t) * HV + hv, mask=valid, other=0.0).to(tl.float32)
    G = tl.load(Gp + (b * T + t) * HV + hv, mask=valid, other=0.0)
    state = tl.load(Sp + ((((z * NC + c) * K + x[:, None]) * NV + iv) * 32 + yy[None, :]), mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    ds = tl.load(Ep + ((((z * NC + c) * K + x[:, None]) * NV + iv) * 32 + yy[None, :]), mask=(x[:, None] < K) & (y[None, :] < V), other=0.0)
    rr, ss = r[:, None], r[None, :]
    A = tl.load(Ap + (((z * NC + c) * 16 + rr) * 16 + ss))
    E = tl.exp(tl.minimum(G[:, None] - G[None, :], 0.0))
    gram = tl.dot(k, tl.trans(k), input_precision="tf32")
    P = tl.load(Pp + (((z * NC + c) * 16 + rr) * 16 + ss))
    pbar = tl.dot(do, tl.trans(d), input_precision="tf32")
    last = tl.maximum(tl.minimum(T - c * 16, 16), 1) - 1
    Ge = tl.sum(tl.where(r == last, G, 0.0), axis=0)
    ee = tl.exp(tl.minimum(Ge - G, 0.0))
    dbar = tl.dot(tl.trans(P), do, input_precision="tf32") + tl.dot(ee[:, None] * k, ds, input_precision="tf32")
    X = tl.dot(tl.trans(A), dbar, input_precision="tf32")
    mbar = tl.where((rr > ss) & valid[:, None] & valid[None, :], -tl.dot(X, tl.trans(d), input_precision="tf32"), 0.0)
    coeff = mbar * be[:, None] * E
    rbar = -tl.dot(X, tl.trans(state), input_precision="tf32")
    qg = (tl.exp(tl.minimum(G, 0.0))[:, None] * tl.dot(do, tl.trans(state), input_precision="tf32") + tl.dot(pbar * E * (rr >= ss), k, input_precision="tf32")) * SCALE
    kg = be[:, None] * tl.exp(tl.minimum(G, 0.0))[:, None] * rbar
    kg += tl.dot(coeff, k, input_precision="tf32") + tl.dot(tl.trans(coeff), k, input_precision="tf32")
    kg += tl.dot(tl.trans(pbar * E * (rr >= ss)), q, input_precision="tf32")
    endkg = tl.dot(d, tl.trans(ds), input_precision="tf32")
    kg += ee[:, None] * endkg
    vg = be[:, None] * X
    bg = tl.sum(X * v, axis=1) + tl.sum(rbar * (tl.exp(tl.minimum(G, 0.0))[:, None] * k), axis=1) + tl.sum(mbar * E * gram, axis=1)
    R = be[:, None] * tl.exp(tl.minimum(G, 0.0))[:, None] * k
    gb = tl.sum(rbar * R, axis=1)
    zm = mbar * tl.where(rr > ss, be[:, None] * E * gram, 0.0)
    gb += tl.sum(zm, axis=1) - tl.sum(zm, axis=0)
    zp = pbar * P
    gb += tl.sum(zp, axis=1) - tl.sum(zp, axis=0)
    direct = tl.exp(tl.minimum(G, 0.0))[:, None] * tl.dot(q, state, input_precision="tf32")
    gb += tl.sum(do * direct, axis=1)
    zend = ee * tl.sum(endkg * k, axis=1)
    gb -= zend
    gb += tl.where(r == last, tl.sum(zend, axis=0) + tl.sum(ds * (tl.exp(tl.minimum(Ge, 0.0)) * state)), 0.0)
    dg = tl.cumsum(gb, axis=0, reverse=True)
    partq = (((z * NV + iv) * T + t[:, None]) * BK + x[None, :])
    tl.store(Qp + partq, qg, mask=valid[:, None] & (x[None, :] < K))
    tl.store(Kp + partq, kg, mask=valid[:, None] & (x[None, :] < K))
    tl.store(Vp + vo, vg, mask=valid[:, None] & (y[None, :] < V))
    ps = (z * NV + iv) * T + t
    tl.store(gp + ps, dg, mask=valid)
    tl.store(Bp + ps, bg, mask=valid)


@triton.jit
def _reduce_qk(Qp, Kp, qo, ko, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, BK)
    h, p = p % H, p // H
    t, b = p % T, p // T
    qv = tl.zeros((BK,), tl.float32)
    kv = tl.zeros((BK,), tl.float32)
    for j in range(HV // H):
        z = b * HV + h * (HV // H) + j
        for iv in range(NV):
            off = ((z * NV + iv) * T + t) * BK + x
            qv += tl.load(Qp + off, mask=x < K, other=0.0)
            kv += tl.load(Kp + off, mask=x < K, other=0.0)
    out = ((b * T + t) * H + h) * K + x
    tl.store(qo + out, qv, mask=x < K)
    tl.store(ko + out, kv, mask=x < K)


@triton.jit
def _reduce_scalar(gp, Bp, go, bo, T: tl.constexpr, HV: tl.constexpr, NV: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    hv, p = p % HV, p // HV
    t, b = p % T, p // T
    z = b * HV + hv
    dg = 0.0
    db = 0.0
    for iv in range(NV):
        off = (z * NV + iv) * T + t
        dg += tl.load(gp + off)
        db += tl.load(Bp + off)
    out = (b * T + t) * HV + hv
    tl.store(go + out, dg)
    tl.store(bo + out, db)


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2:]
        BK = triton.next_power_of_2(K)
        NC, NV = triton.cdiv(T, BC), triton.cdiv(V, BV)
        A = torch.empty((B * HV, NC, BC, BC), device=q.device, dtype=torch.float32)
        P = torch.empty_like(A, dtype=torch.float32)
        G = torch.empty((B, T, HV), device=q.device, dtype=torch.float32)
        S = torch.empty((B * HV, NC, K, NV, BV), device=q.device, dtype=torch.float32)
        D = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _build_wy[(NC, B * HV)](q, k, g, beta, A, P, G, T, H, HV, K, BK, NC, K ** -0.5)
        _scan_state[(B * HV * NV,)](k, v, beta, A, G, S, D, T, H, HV, K, V, BK, NC, NV)
        _emit[(B * HV * NC * NV,)](q, P, G, S, D, out, T, H, HV, K, V, BK, NC, NV, K ** -0.5)
        ctx.save_for_backward(q, k, v, beta, G, A, P, S, D)
        ctx.dims = B, T, H, HV, K, V, BK, NC, NV
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, beta, G, A, P, S, D = ctx.saved_tensors
        B, T, H, HV, K, V, BK, NC, NV = ctx.dims
        do = do.contiguous()
        E = torch.empty_like(S, dtype=torch.float32)
        pq = torch.empty((B * HV, NV, T, BK), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq, dtype=torch.float32)
        pg = torch.empty((B * HV, NV, T), device=q.device, dtype=torch.float32)
        pb = torch.empty_like(pg, dtype=torch.float32)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dg = torch.empty((B, T, HV), device=q.device, dtype=torch.float32)
        db = torch.empty_like(beta)
        _adjoint[(B * HV * NV,)](q, k, beta, G, A, P, do, E, T, H, HV, K, V, BK, NC, NV, K ** -0.5)
        _chunk_grad[(B * HV * NC * NV,)](q, k, v, beta, G, A, P, S, D, do, E, pq, pk, dv, pg, pb, T, H, HV, K, V, BK, NC, NV, K ** -0.5)
        _reduce_qk[(B * T * H,)](pq, pk, dq, dk, T, H, HV, K, BK, NV)
        _reduce_scalar[(B * T * HV,)](pg, pb, dg, db, T, HV, NV)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)
