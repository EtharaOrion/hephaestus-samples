import math

import torch
import triton
import triton.language as tl

CHUNK = 16
VALUE_BLOCK = 16


@triton.jit
def _wy_build(kp, gp, bp, ap, T: tl.constexpr, K: tl.constexpr,
              H: tl.constexpr, HV: tl.constexpr, NC: tl.constexpr,
              BK: tl.constexpr, BC: tl.constexpr):
    z = tl.program_id(0)
    c = z % NC
    h = (z // NC) % HV
    b = z // (NC * HV)
    hk = h // (HV // H)
    ii = tl.arange(0, BC)
    kk = tl.arange(0, BK)
    t = c * BC + ii
    tm = t < T
    ko = ((b * T + t[:, None]) * H + hk) * K + kk[None, :]
    x = tl.load(kp + ko, mask=tm[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
    go = (b * T + t) * HV + h
    gg = tl.load(gp + go, mask=tm, other=0.).to(tl.float32)
    be = tl.load(bp + go, mask=tm, other=0.).to(tl.float32)
    G = tl.cumsum(gg, axis=0)
    gram = tl.dot(x, tl.trans(x), input_precision="tf32")
    rows = ii[:, None]
    cols = ii[None, :]
    lower = (rows > cols) & tm[:, None] & tm[None, :]
    ratio = tl.exp(tl.where(lower, G[:, None] - G[None, :], 0.))
    L = tl.where(lower, be[:, None] * ratio * gram, 0.)
    inv = tl.where(rows == cols, 1., 0.)
    for r in tl.static_range(1, BC):
        lrow = tl.sum(tl.where(rows == r, L, 0.), axis=0)
        coeff = tl.sum(lrow[:, None] * inv, axis=0)
        inv = tl.where(rows == r, inv - coeff[None, :], inv)
    ao = ((z * BC + rows) * BC + cols).to(tl.int64)
    tl.store(ap + ao, inv)


@triton.jit
def _state_scan(kp, vp, gp, bp, ap, sp, up,
                T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                H: tl.constexpr, HV: tl.constexpr, NC: tl.constexpr,
                BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr):
    z = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    iv = z % nv
    h = (z // nv) % HV
    b = z // (nv * HV)
    hk = h // (HV // H)
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    ii = tl.arange(0, BC)
    S = tl.zeros((BK, BV), tl.float32)
    for c in range(0, NC):
        t = c * BC + ii
        tm = t < T
        so = ((((b * HV + h) * NC + c) * K + kk[:, None]) * V + vv[None, :]).to(tl.int64)
        tl.store(sp + so, S, mask=(kk[:, None] < K) & (vv[None, :] < V))
        ko = ((b * T + t[:, None]) * H + hk) * K + kk[None, :]
        vo = ((b * T + t[:, None]) * HV + h) * V + vv[None, :]
        x = tl.load(kp + ko, mask=tm[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
        val = tl.load(vp + vo, mask=tm[:, None] & (vv[None, :] < V), other=0.).to(tl.float32)
        go = (b * T + t) * HV + h
        G = tl.cumsum(tl.load(gp + go, mask=tm, other=0.).to(tl.float32), axis=0)
        be = tl.load(bp + go, mask=tm, other=0.).to(tl.float32)
        aa = tl.load(ap + (((b * HV + h) * NC + c) * BC + ii[:, None]) * BC + ii[None, :])
        pred = tl.dot(x, S, input_precision="tf32") * tl.exp(G[:, None])
        rhs = be[:, None] * (val - pred)
        U = tl.dot(aa, rhs, input_precision="tf32")
        uo = ((b * T + t[:, None]) * HV + h) * V + vv[None, :]
        tl.store(up + uo, U, mask=tm[:, None] & (vv[None, :] < V))
        last = tl.sum(tl.where(ii == tl.minimum(BC - 1, T - c * BC - 1), G, 0.), axis=0)
        R = x * tl.exp(last - G)[:, None]
        S = S * tl.exp(last) + tl.dot(tl.trans(R), U, input_precision="tf32")


@triton.jit
def _output_chunks(qp, kp, gp, sp, up, op,
                   T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                   H: tl.constexpr, HV: tl.constexpr, NC: tl.constexpr,
                   BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr):
    z = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    iv = z % nv
    c = (z // nv) % NC
    h = (z // (nv * NC)) % HV
    b = z // (nv * NC * HV)
    hk = h // (HV // H)
    ii = tl.arange(0, BC)
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    t = c * BC + ii
    tm = t < T
    qo = ((b * T + t[:, None]) * H + hk) * K + kk[None, :]
    q = tl.load(qp + qo, mask=tm[:, None] & (kk[None, :] < K), other=0.).to(tl.float32) * (K ** -0.5)
    k = tl.load(kp + qo, mask=tm[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
    go = (b * T + t) * HV + h
    G = tl.cumsum(tl.load(gp + go, mask=tm, other=0.).to(tl.float32), axis=0)
    so = ((((b * HV + h) * NC + c) * K + kk[:, None]) * V + vv[None, :]).to(tl.int64)
    S = tl.load(sp + so, mask=(kk[:, None] < K) & (vv[None, :] < V), other=0.)
    uo = ((b * T + t[:, None]) * HV + h) * V + vv[None, :]
    U = tl.load(up + uo, mask=tm[:, None] & (vv[None, :] < V), other=0.)
    causal = ii[:, None] >= ii[None, :]
    ratio = tl.exp(tl.where(causal, G[:, None] - G[None, :], 0.))
    M = tl.where(causal, tl.dot(q, tl.trans(k), input_precision="tf32") * ratio, 0.)
    O = tl.dot(q, S, input_precision="tf32") * tl.exp(G[:, None]) + tl.dot(M, U, input_precision="tf32")
    tl.store(op + uo, O, mask=tm[:, None] & (vv[None, :] < V))


@triton.jit
def _chunk_reverse(qp, kp, vp, gp, bp, dop, ap, sp, up,
                   dqp, dkp, dvp, dgp, dbp,
                   T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                   H: tl.constexpr, HV: tl.constexpr, NC: tl.constexpr,
                   BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr):
    z = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    iv = z % nv
    h = (z // nv) % HV
    b = z // (nv * HV)
    hk = h // (HV // H)
    ii = tl.arange(0, BC)
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    dS = tl.zeros((BK, BV), tl.float32)
    for rc in range(0, NC):
        c = NC - 1 - rc
        t = c * BC + ii
        tm = t < T
        qko = ((b * T + t[:, None]) * H + hk) * K + kk[None, :]
        vo = ((b * T + t[:, None]) * HV + h) * V + vv[None, :]
        q = tl.load(qp + qko, mask=tm[:, None] & (kk[None, :] < K), other=0.).to(tl.float32) * (K ** -0.5)
        k = tl.load(kp + qko, mask=tm[:, None] & (kk[None, :] < K), other=0.).to(tl.float32)
        val = tl.load(vp + vo, mask=tm[:, None] & (vv[None, :] < V), other=0.).to(tl.float32)
        dO = tl.load(dop + vo, mask=tm[:, None] & (vv[None, :] < V), other=0.).to(tl.float32)
        U = tl.load(up + vo, mask=tm[:, None] & (vv[None, :] < V), other=0.)
        go = (b * T + t) * HV + h
        G = tl.cumsum(tl.load(gp + go, mask=tm, other=0.).to(tl.float32), axis=0)
        be = tl.load(bp + go, mask=tm, other=0.).to(tl.float32)
        so = ((((b * HV + h) * NC + c) * K + kk[:, None]) * V + vv[None, :]).to(tl.int64)
        S = tl.load(sp + so, mask=(kk[:, None] < K) & (vv[None, :] < V), other=0.)
        aa = tl.load(ap + (((b * HV + h) * NC + c) * BC + ii[:, None]) * BC + ii[None, :])
        last_i = tl.minimum(BC - 1, T - c * BC - 1)
        last = tl.sum(tl.where(ii == last_i, G, 0.), axis=0)
        D = tl.exp(G)
        causal = ii[:, None] >= ii[None, :]
        E = tl.exp(tl.where(causal, G[:, None] - G[None, :], 0.))
        R = k * tl.exp(last - G)[:, None]
        M = tl.where(causal, tl.dot(q, tl.trans(k), input_precision="tf32") * E, 0.)

        dU = tl.dot(tl.trans(M), dO, input_precision="tf32") + tl.dot(R, dS, input_precision="tf32")
        dM = tl.dot(dO, tl.trans(U), input_precision="tf32")
        dR = tl.dot(U, tl.trans(dS), input_precision="tf32")
        dr = tl.dot(tl.trans(aa), dU, input_precision="tf32")
        dA = -tl.dot(dr, tl.trans(U), input_precision="tf32")

        pred0 = tl.dot(k, S, input_precision="tf32")
        dx = -be[:, None] * dr
        dval = be[:, None] * dr
        dbe = tl.sum(dr * (val - D[:, None] * pred0), axis=1)
        dk = tl.dot(dx * D[:, None], tl.trans(S), input_precision="tf32")
        dS0 = tl.dot(tl.trans(k), dx * D[:, None], input_precision="tf32")
        dG = tl.sum(dx * (D[:, None] * pred0), axis=1)

        dq = tl.dot(dO * D[:, None], tl.trans(S), input_precision="tf32")
        dS0 += tl.dot(tl.trans(q), dO * D[:, None], input_precision="tf32")
        dG += tl.sum(dO * (D[:, None] * tl.dot(q, S, input_precision="tf32")), axis=1)

        lower = ii[:, None] > ii[None, :]
        gram = tl.dot(k, tl.trans(k), input_precision="tf32")
        dL = tl.where(lower, dA, 0.)
        wA = dL * be[:, None] * E
        dk += tl.dot(wA + tl.trans(wA), k, input_precision="tf32")
        dbe += tl.sum(dL * E * gram, axis=1)
        zA = dL * be[:, None] * E * gram
        dG += tl.sum(zA, axis=1) - tl.sum(zA, axis=0)

        incl = ii[:, None] >= ii[None, :]
        wM = tl.where(incl, dM * E, 0.)
        dq += tl.dot(wM, k, input_precision="tf32")
        dk += tl.dot(tl.trans(wM), q, input_precision="tf32")
        zM = tl.where(incl, dM * E * tl.dot(q, tl.trans(k), input_precision="tf32"), 0.)
        dG += tl.sum(zM, axis=1) - tl.sum(zM, axis=0)

        er = tl.exp(last - G)
        dk += dR * er[:, None]
        zR = tl.sum(dR * R, axis=1)
        dG -= zR
        dlast = tl.sum(zR, axis=0)
        dS0 += dS * tl.exp(last)
        dlast += tl.sum(tl.sum(dS * (S * tl.exp(last)), axis=1), axis=0)
        dG += tl.where(ii == last_i, dlast, 0.)
        dS = dS0

        dg = tl.sum(dG, axis=0) - tl.cumsum(dG, axis=0) + dG
        part = ((b * HV + h) * nv + iv) * T + t
        tl.store(dgp + part, dg, mask=tm)
        tl.store(dbp + part, dbe, mask=tm)
        tl.store(dvp + vo, dval, mask=tm[:, None] & (vv[None, :] < V))
        qpart = (((b * HV + h) * nv + iv) * T + t[:, None]) * K + kk[None, :]
        tl.store(dqp + qpart, dq * (K ** -0.5), mask=tm[:, None] & (kk[None, :] < K))
        tl.store(dkp + qpart, dk, mask=tm[:, None] & (kk[None, :] < K))


@triton.jit
def _sum_v_parts(xp, yp, N: tl.constexpr, T: tl.constexpr,
                 HV: tl.constexpr, NV: tl.constexpr, BS: tl.constexpr):
    i = tl.program_id(0) * BS + tl.arange(0, BS)
    b = i // (T * HV)
    rem = i % (T * HV)
    t = rem // HV
    h = rem % HV
    acc = tl.zeros((BS,), tl.float32)
    for j in range(0, NV):
        off = (((b * HV + h) * NV + j) * T + t).to(tl.int64)
        acc += tl.load(xp + off, mask=i < N, other=0.)
    tl.store(yp + i, acc, mask=i < N)


@triton.jit
def _sum_v_parts_head(xp, yp, N: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
                      H: tl.constexpr, HV: tl.constexpr, NV: tl.constexpr,
                      BS: tl.constexpr):
    i = tl.program_id(0) * BS + tl.arange(0, BS)
    b = i // (T * H * K)
    rem = i % (T * H * K)
    t = rem // (H * K)
    h = (rem // K) % H
    k = rem % K
    acc = tl.zeros((BS,), tl.float32)
    rep = HV // H
    for r in range(0, rep):
        vh = h * rep + r
        for j in range(0, NV):
            off = (((b * HV + vh) * NV + j) * T * K + t * K + k).to(tl.int64)
            acc += tl.load(xp + off, mask=i < N, other=0.)
    tl.store(yp + i, acc, mask=i < N)


class _GatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        NC = triton.cdiv(T, CHUNK)
        BK = triton.next_power_of_2(K)
        NV = triton.cdiv(V, VALUE_BLOCK)
        a = torch.empty((B, HV, NC, CHUNK, CHUNK), device=q.device, dtype=torch.float32)
        states = torch.empty((B, HV, NC, K, V), device=q.device, dtype=torch.float32)
        u = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _wy_build[(B * HV * NC,)](k, g, beta, a, T, K, H, HV, NC, BK=BK, BC=CHUNK, num_warps=4)
        _state_scan[(B * HV * NV,)](k, v, g, beta, a, states, u, T, K, V, H, HV, NC,
                                    BK=BK, BV=VALUE_BLOCK, BC=CHUNK, num_warps=4)
        _output_chunks[(B * HV * NC * NV,)](q, k, g, states, u, out, T, K, V, H, HV, NC,
                                            BK=BK, BV=VALUE_BLOCK, BC=CHUNK, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, a, states, u)
        ctx.meta = (B, T, H, K, HV, V, NC, BK, NV)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, a, states, u = ctx.saved_tensors
        dout = dout.contiguous()
        B, T, H, K, HV, V, NC, BK, NV = ctx.meta
        pq = torch.empty((B, HV, NV, T, K), device=q.device, dtype=torch.float32)
        pk = torch.empty_like(pq)
        pg = torch.empty((B, HV, NV, T), device=q.device, dtype=torch.float32)
        pb = torch.empty_like(pg)
        dv = torch.empty_like(v)
        _chunk_reverse[(B * HV * NV,)](q, k, v, g, beta, dout, a, states, u,
                                       pq, pk, dv, pg, pb, T, K, V, H, HV, NC,
                                       BK=BK, BV=VALUE_BLOCK, BC=CHUNK, num_warps=4)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dg = torch.empty_like(g)
        db = torch.empty_like(beta)
        n_q = B * H * T * K
        _sum_v_parts_head[(triton.cdiv(n_q, 256),)](pq, dq, n_q, T, K, H, HV, NV, BS=256)
        _sum_v_parts_head[(triton.cdiv(n_q, 256),)](pk, dk, n_q, T, K, H, HV, NV, BS=256)
        n_s = B * HV * T
        _sum_v_parts[(triton.cdiv(n_s, 256),)](pg, dg, n_s, T, HV, NV, BS=256)
        _sum_v_parts[(triton.cdiv(n_s, 256),)](pb, db, n_s, T, HV, NV, BS=256)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDelta.apply(q, k, v, g, beta)
