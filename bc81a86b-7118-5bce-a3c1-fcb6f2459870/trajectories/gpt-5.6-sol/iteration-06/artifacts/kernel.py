import math

import torch
import triton
import triton.language as tl


CHUNK = 16
VALUE_BLOCK = 16


@triton.jit
def _chunk_prepare(kp, gp, bp, ap, cp, T: tl.constexpr, H: tl.constexpr,
                   HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
                   NC: tl.constexpr, C: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    b = p // HV
    hv = p - b * HV
    hk = hv // (HV // H)
    i = tl.arange(0, C)
    x = tl.arange(0, BK)
    t = c * C + i
    valid = t < T
    ko = ((b * T + t[:, None]) * H + hk) * K + x[None, :]
    kk = tl.load(kp + ko, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    go = (b * T + t) * HV + hv
    gg = tl.load(gp + go, mask=valid, other=0.0).to(tl.float32)
    bb = tl.load(bp + go, mask=valid, other=0.0).to(tl.float32)
    G = tl.cumsum(gg, axis=0)
    gram = tl.dot(kk, tl.trans(kk), input_precision="tf32")
    ii = i[:, None]
    jj = i[None, :]
    lower = (ii > jj) & valid[:, None] & valid[None, :]
    L = tl.where(lower, bb[:, None] * tl.exp(G[:, None] - G[None, :]) * gram, 0.0)
    M = tl.where(ii == jj, 1.0, 0.0)
    for r in tl.static_range(1, C):
        row = tl.where(i == r, 1.0, 0.0)
        for s in tl.static_range(0, r):
            lrs = tl.sum(tl.where((ii == r) & (jj == s), L, 0.0))
            ms = tl.sum(tl.where(ii == s, M, 0.0), axis=0)
            row -= lrs * ms
        M = tl.where(ii == r, row[None, :], M)
    ao = (p * NC + c) * C * C + ii * C + jj
    tl.store(ap + ao, M)
    tl.store(cp + p * T + t, G, mask=valid)


@triton.jit
def _state_forward(kp, vp, gp, bp, ap, cp, sp, up,
                   T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                   K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                   NC: tl.constexpr, NV: tl.constexpr, C: tl.constexpr,
                   BV: tl.constexpr):
    z = tl.program_id(0).to(tl.int64)
    p = z // NV
    nv = z - p * NV
    b = p // HV
    hv = p - b * HV
    hk = hv // (HV // H)
    x = tl.arange(0, BK)
    y = tl.arange(0, BV)
    i = tl.arange(0, C)
    state = tl.zeros((BK, BV), tl.float32)
    for c in range(NC):
        t = c * C + i
        valid = t < T
        so = ((p * NC + c) * K + x[:, None]) * V + nv * BV + y[None, :]
        tl.store(sp + so, state, mask=(x[:, None] < K) & (nv * BV + y[None, :] < V))
        ko = ((b * T + t[:, None]) * H + hk) * K + x[None, :]
        kk = tl.load(kp + ko, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        vo = ((b * T + t[:, None]) * HV + hv) * V + nv * BV + y[None, :]
        vv = tl.load(vp + vo, mask=valid[:, None] & (nv * BV + y[None, :] < V), other=0.0).to(tl.float32)
        go = (b * T + t) * HV + hv
        bb = tl.load(bp + go, mask=valid, other=0.0).to(tl.float32)
        G = tl.load(cp + p * T + t, mask=valid, other=0.0)
        ii = i[:, None]
        jj = i[None, :]
        M = tl.load(ap + (p * NC + c) * C * C + ii * C + jj)
        pred0 = tl.dot(kk, state, input_precision="tf32")
        rhs = bb[:, None] * (vv - tl.exp(G)[:, None] * pred0)
        uu = tl.dot(M, rhs, input_precision="tf32")
        uo = ((b * T + t[:, None]) * HV + hv) * V + nv * BV + y[None, :]
        tl.store(up + uo, uu, mask=valid[:, None] & (nv * BV + y[None, :] < V))
        n = tl.minimum(C, T - c * C)
        last = tl.sum(tl.where(i == n - 1, G, 0.0), axis=0)
        weights = tl.exp(last - G) * valid
        state = tl.exp(last) * state + tl.dot(tl.trans(kk), weights[:, None] * uu,
                                               input_precision="tf32")


@triton.jit
def _chunk_output(qp, kp, cp, sp, up, op, T: tl.constexpr,
                  H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                  V: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
                  NV: tl.constexpr, C: tl.constexpr, BV: tl.constexpr,
                  SCALE: tl.constexpr):
    z = tl.program_id(0).to(tl.int64)
    nv = z % NV
    z //= NV
    c = z % NC
    p = z // NC
    b = p // HV
    hv = p - b * HV
    hk = hv // (HV // H)
    i = tl.arange(0, C)
    x = tl.arange(0, BK)
    y = tl.arange(0, BV)
    t = c * C + i
    valid = t < T
    qo = ((b * T + t[:, None]) * H + hk) * K + x[None, :]
    q = tl.load(qp + qo, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * SCALE
    k = tl.load(kp + qo, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
    G = tl.load(cp + p * T + t, mask=valid, other=0.0)
    so = ((p * NC + c) * K + x[:, None]) * V + nv * BV + y[None, :]
    s = tl.load(sp + so, mask=(x[:, None] < K) & (nv * BV + y[None, :] < V), other=0.0)
    uo = ((b * T + t[:, None]) * HV + hv) * V + nv * BV + y[None, :]
    u = tl.load(up + uo, mask=valid[:, None] & (nv * BV + y[None, :] < V), other=0.0)
    score = tl.dot(q, tl.trans(k), input_precision="tf32")
    ii = i[:, None]
    jj = i[None, :]
    score *= tl.where((ii >= jj) & valid[:, None] & valid[None, :],
                      tl.exp(G[:, None] - G[None, :]), 0.0)
    out = tl.exp(G)[:, None] * tl.dot(q, s, input_precision="tf32")
    out += tl.dot(score, u, input_precision="tf32")
    oo = ((b * T + t[:, None]) * HV + hv) * V + nv * BV + y[None, :]
    tl.store(op + oo, out, mask=valid[:, None] & (nv * BV + y[None, :] < V))


@triton.jit
def _chunk_backward(qp, kp, vp, gp, bp, dop, ap, cp, sp, up,
                    dpp, dvp, dgp, dbp, T: tl.constexpr, H: tl.constexpr,
                    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    B: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, NV: tl.constexpr,
                    C: tl.constexpr, BV: tl.constexpr, SCALE: tl.constexpr):
    z = tl.program_id(0).to(tl.int64)
    p = z // NV
    nv = z - p * NV
    b = p // HV
    hv = p - b * HV
    hk = hv // (HV // H)
    x = tl.arange(0, BK)
    y = tl.arange(0, BV)
    i = tl.arange(0, C)
    ii = i[:, None]
    jj = i[None, :]
    ds = tl.zeros((BK, BV), tl.float32)
    for cr in range(NC):
        c = NC - 1 - cr
        t = c * C + i
        valid = t < T
        qko = ((b * T + t[:, None]) * H + hk) * K + x[None, :]
        q = tl.load(qp + qko, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32) * SCALE
        k = tl.load(kp + qko, mask=valid[:, None] & (x[None, :] < K), other=0.0).to(tl.float32)
        vo = ((b * T + t[:, None]) * HV + hv) * V + nv * BV + y[None, :]
        v = tl.load(vp + vo, mask=valid[:, None] & (nv * BV + y[None, :] < V), other=0.0).to(tl.float32)
        do = tl.load(dop + vo, mask=valid[:, None] & (nv * BV + y[None, :] < V), other=0.0).to(tl.float32)
        u = tl.load(up + vo, mask=valid[:, None] & (nv * BV + y[None, :] < V), other=0.0)
        so = ((p * NC + c) * K + x[:, None]) * V + nv * BV + y[None, :]
        s0 = tl.load(sp + so, mask=(x[:, None] < K) & (nv * BV + y[None, :] < V), other=0.0)
        go = (b * T + t) * HV + hv
        beta = tl.load(bp + go, mask=valid, other=0.0).to(tl.float32)
        G = tl.load(cp + p * T + t, mask=valid, other=0.0)
        M = tl.load(ap + (p * NC + c) * C * C + ii * C + jj)
        mask_l = (ii > jj) & valid[:, None] & valid[None, :]
        mask_o = (ii >= jj) & valid[:, None] & valid[None, :]
        eg = tl.exp(G)
        gram_qk = tl.dot(q, tl.trans(k), input_precision="tf32")
        E = tl.where(mask_o, tl.exp(G[:, None] - G[None, :]), 0.0)
        co = gram_qk * E
        X = eg[:, None] * q
        n = tl.minimum(C, T - c * C)
        Glast = tl.sum(tl.where(i == n - 1, G, 0.0), axis=0)
        er = tl.exp(Glast - G) * valid
        R = er[:, None] * k

        dX = tl.dot(do, tl.trans(s0), input_precision="tf32")
        dco = tl.dot(do, tl.trans(u), input_precision="tf32")
        du = tl.dot(tl.trans(co), do, input_precision="tf32")
        dR = tl.dot(u, tl.trans(ds), input_precision="tf32")
        du += tl.dot(R, ds, input_precision="tf32")
        ds0 = tl.dot(tl.trans(X), do, input_precision="tf32") + tl.exp(Glast) * ds
        da = tl.sum(ds * s0, axis=1)
        dG = tl.zeros((C,), tl.float32)
        dG += tl.sum(dX * X, axis=1)
        dG += tl.sum(dR * R, axis=1) * -1.0
        dGlast = tl.sum(dR * R) + tl.sum(da) * tl.exp(Glast)

        dce = dco * E
        dq = eg[:, None] * dX + tl.dot(dce, k, input_precision="tf32")
        dk = tl.dot(tl.trans(dce), q, input_precision="tf32") + er[:, None] * dR
        gate_c = dco * co
        dG += tl.sum(gate_c, axis=1) - tl.sum(gate_c, axis=0)

        dz = tl.dot(tl.trans(M), du, input_precision="tf32")
        dA = -tl.dot(dz, tl.trans(u), input_precision="tf32")
        dL = tl.where(mask_l, dA, 0.0)
        P = (beta * eg)[:, None] * k
        dP = -tl.dot(dz, tl.trans(s0), input_precision="tf32")
        ds0 -= tl.dot(tl.trans(P), dz, input_precision="tf32")
        dv = beta[:, None] * dz
        dbeta = tl.sum(dz * v, axis=1) + tl.sum(dP * (eg[:, None] * k), axis=1)
        dk += (beta * eg)[:, None] * dP
        dG += tl.sum(dP * P, axis=1)

        gram = tl.dot(k, tl.trans(k), input_precision="tf32")
        EL = tl.where(mask_l, tl.exp(G[:, None] - G[None, :]), 0.0)
        F = dL * beta[:, None] * EL
        dk += tl.dot(F, k, input_precision="tf32") + tl.dot(tl.trans(F), k, input_precision="tf32")
        dbeta += tl.sum(dL * EL * gram, axis=1)
        gate_l = dL * beta[:, None] * EL * gram
        dG += tl.sum(gate_l, axis=1) - tl.sum(gate_l, axis=0)
        dG = tl.where(i == n - 1, dG + dGlast, dG)
        dg = tl.cumsum(dG, axis=0, reverse=True)

        base = (((p * T + t[:, None]) * NV + nv) * K + x[None, :])
        tl.store(dpp + base, dq * SCALE, mask=valid[:, None] & (x[None, :] < K))
        plane = B * HV * T * NV * K
        tl.store(dpp + plane + base, dk, mask=valid[:, None] & (x[None, :] < K))
        tl.store(dvp + vo, dv, mask=valid[:, None] & (nv * BV + y[None, :] < V))
        scalar = (p * T + t) * NV + nv
        tl.store(dgp + scalar, dg, mask=valid)
        tl.store(dbp + scalar, dbeta, mask=valid)
        ds = ds0


@triton.jit
def _reduce_qk(pp, dqp, dkp, T: tl.constexpr, H: tl.constexpr,
               HV: tl.constexpr, K: tl.constexpr, B: tl.constexpr, BK: tl.constexpr,
               NV: tl.constexpr):
    z = tl.program_id(0).to(tl.int64)
    b = z // H
    h = z - b * H
    t = tl.program_id(1).to(tl.int64)
    x = tl.arange(0, BK)
    dq = tl.zeros((BK,), tl.float32)
    dk = tl.zeros((BK,), tl.float32)
    rep = HV // H
    plane = B * HV * T * NV * K
    for r in range(rep):
        hv = h * rep + r
        p = b * HV + hv
        for nv in range(NV):
            off = (((p * T + t) * NV + nv) * K + x)
            dq += tl.load(pp + off, mask=x < K, other=0.0)
            dk += tl.load(pp + plane + off, mask=x < K, other=0.0)
    out = ((b * T + t) * H + h) * K + x
    tl.store(dqp + out, dq, mask=x < K)
    tl.store(dkp + out, dk, mask=x < K)


@triton.jit
def _reduce_scalars(dgp, dbp, ogp, obp, T: tl.constexpr,
                    HV: tl.constexpr, NV: tl.constexpr, BT: tl.constexpr):
    p = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    t = block * BT + tl.arange(0, BT)
    dg = tl.zeros((BT,), tl.float32)
    db = tl.zeros((BT,), tl.float32)
    for nv in range(NV):
        off = (p * T + t) * NV + nv
        dg += tl.load(dgp + off, mask=t < T, other=0.0)
        db += tl.load(dbp + off, mask=t < T, other=0.0)
    b = p // HV
    hv = p - b * HV
    out = (b * T + t) * HV + hv
    tl.store(ogp + out, dg, mask=t < T)
    tl.store(obp + out, db, mask=t < T)


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2:]
        BK = triton.next_power_of_2(K)
        NC = triton.cdiv(T, CHUNK)
        NV = triton.cdiv(V, VALUE_BLOCK)
        dev = q.device
        accum = torch.empty((B * HV, NC, CHUNK, CHUNK), device=dev, dtype=torch.float32)
        csum = torch.empty((B * HV, T), device=dev, dtype=torch.float32)
        states = torch.empty((B * HV, NC, K, V), device=dev, dtype=torch.float32)
        updates = torch.empty((B, T, HV, V), device=dev, dtype=torch.float32)
        out = torch.empty_like(v)
        _chunk_prepare[(B * HV, NC)](k, g, beta, accum, csum, T=T, H=H, HV=HV,
            K=K, BK=BK, NC=NC, C=CHUNK, num_warps=4)
        _state_forward[(B * HV * NV,)](k, v, g, beta, accum, csum, states, updates,
            T=T, H=H, HV=HV, K=K, V=V, BK=BK, NC=NC, NV=NV, C=CHUNK,
            BV=VALUE_BLOCK, num_warps=8)
        _chunk_output[(B * HV * NC * NV,)](q, k, csum, states, updates, out,
            T=T, H=H, HV=HV, K=K, V=V, BK=BK, NC=NC, NV=NV, C=CHUNK,
            BV=VALUE_BLOCK, SCALE=K ** -0.5, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, accum, csum, states, updates)
        ctx.meta = (B, T, H, HV, K, V, BK, NC, NV)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, accum, csum, states, updates = ctx.saved_tensors
        B, T, H, HV, K, V, BK, NC, NV = ctx.meta
        dout = dout.contiguous()
        dev = q.device
        partial = torch.empty((2, B * HV, T, NV, K), device=dev, dtype=torch.float32)
        dv = torch.empty_like(v)
        pdg = torch.empty((B * HV, T, NV), device=dev, dtype=torch.float32)
        pdb = torch.empty_like(pdg)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dg = torch.empty_like(g)
        db = torch.empty_like(beta)
        _chunk_backward[(B * HV * NV,)](q, k, v, g, beta, dout, accum, csum,
            states, updates, partial, dv, pdg, pdb, T=T, H=H, HV=HV, K=K,
            V=V, B=B, BK=BK, NC=NC, NV=NV, C=CHUNK, BV=VALUE_BLOCK,
            SCALE=K ** -0.5, num_warps=8)
        _reduce_qk[(B * H, T)](partial, dq, dk, T=T, H=H, HV=HV, K=K,
            B=B, BK=BK, NV=NV, num_warps=4)
        _reduce_scalars[(B * HV, triton.cdiv(T, 128))](pdg, pdb, dg, db,
            T=T, HV=HV, NV=NV, BT=128, num_warps=4)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)
