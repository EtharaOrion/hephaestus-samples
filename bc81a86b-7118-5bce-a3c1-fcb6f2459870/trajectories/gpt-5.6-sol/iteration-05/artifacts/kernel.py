import math

import torch
import triton
import triton.language as tl

CHUNK = 16
VALUE_BLOCK = 32


@triton.jit
def _safe_exp(x):
    return tl.exp(tl.minimum(x, 80.0))


@triton.jit
def _prepare_chunks(k, v, g, beta, A, W, U, T: tl.constexpr,
                    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                    V: tl.constexpr, C: tl.constexpr, BK: tl.constexpr,
                    BV: tl.constexpr):
    pc = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // HV
    hv = bh - b * HV
    h = hv // (HV // H)
    t = pc * C + tl.arange(0, C)
    r = tl.arange(0, C)
    mt = t < T
    base_k = ((b * T + t) * H + h) * K
    kk = tl.load(k + base_k[:, None] + tl.arange(0, BK)[None, :],
                 mask=mt[:, None] & (tl.arange(0, BK)[None, :] < K), other=0.0)
    gg = tl.load(g + (b * T + t) * HV + hv, mask=mt, other=0.0).to(tl.float32)
    bb = tl.load(beta + (b * T + t) * HV + hv, mask=mt, other=0.0).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    gram = tl.dot(kk, tl.trans(kk), input_precision="ieee")
    valid_l = mt[:, None] & mt[None, :] & (r[:, None] > r[None, :])
    decay = _safe_exp(cs[:, None] - cs[None, :])
    L = tl.where(valid_l, gram * decay * bb[:, None], 0.0)

    inv = -L
    for i in range(2, C):
        row = tl.sum(tl.where((r == i)[:, None], -L, 0.0), axis=0)
        row = tl.where(r < i, row, 0.0)
        row += tl.sum(row[:, None] * inv, axis=0)
        inv = tl.where((r == i)[:, None], row, inv)
    inv += tl.where(r[:, None] == r[None, :], 1.0, 0.0)

    aoff = ((b * triton.cdiv(T, C) + pc) * HV + hv) * C * C
    tl.store(A + aoff + r[:, None] * C + r[None, :], inv,
             mask=mt[:, None] & mt[None, :])

    kb = kk * (bb * _safe_exp(cs))[:, None]
    ww = tl.dot(inv, kb.to(tl.float32), input_precision="ieee")
    woff = ((b * T + t) * HV + hv) * K
    tl.store(W + woff[:, None] + tl.arange(0, BK)[None, :], ww,
             mask=mt[:, None] & (tl.arange(0, BK)[None, :] < K))

    for iv in range(tl.cdiv(V, BV)):
        ov = iv * BV + tl.arange(0, BV)
        vv = tl.load(v + ((b * T + t) * HV + hv)[:, None] * V + ov[None, :],
                     mask=mt[:, None] & (ov[None, :] < V), other=0.0)
        uu = tl.dot(inv, vv.to(tl.float32) * bb[:, None], input_precision="ieee")
        tl.store(U + ((b * T + t) * HV + hv)[:, None] * V + ov[None, :], uu,
                 mask=mt[:, None] & (ov[None, :] < V))


@triton.jit
def _state_forward(k, g, W, U, states, Z, T: tl.constexpr,
                   H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                   V: tl.constexpr, C: tl.constexpr, BK: tl.constexpr,
                   BV: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    iv = tl.program_id(1).to(tl.int64)
    b = bh // HV
    hv = bh - b * HV
    h = hv // (HV // H)
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mkv = (ok[:, None] < K) & (ov[None, :] < V)
    state = tl.zeros((BK, BV), tl.float32)
    for pc in range(triton.cdiv(T, C)):
        t = pc * C + tl.arange(0, C)
        mt = t < T
        soff = (((b * triton.cdiv(T, C) + pc) * HV + hv) * K)
        tl.store(states + (soff + ok)[:, None] * V + ov[None, :], state, mask=mkv)
        ww = tl.load(W + ((b * T + t) * HV + hv)[:, None] * K + ok[None, :],
                     mask=mt[:, None] & (ok[None, :] < K), other=0.0)
        uu = tl.load(U + ((b * T + t) * HV + hv)[:, None] * V + ov[None, :],
                     mask=mt[:, None] & (ov[None, :] < V), other=0.0)
        zz = uu - tl.dot(ww.to(tl.float32), state, input_precision="ieee")
        tl.store(Z + ((b * T + t) * HV + hv)[:, None] * V + ov[None, :], zz,
                 mask=mt[:, None] & (ov[None, :] < V))
        gg = tl.load(g + (b * T + t) * HV + hv, mask=mt, other=0.0).to(tl.float32)
        cs = tl.cumsum(gg, axis=0)
        total = tl.sum(gg, axis=0)
        kk = tl.load(k + ((b * T + t) * H + h)[:, None] * K + ok[None, :],
                     mask=mt[:, None] & (ok[None, :] < K), other=0.0)
        rr = kk * _safe_exp(total - cs)[:, None]
        state = state * _safe_exp(total) + tl.dot(tl.trans(rr).to(tl.float32), zz, input_precision="ieee")


@triton.jit
def _chunk_output(q, k, g, states, Z, out, T: tl.constexpr,
                  H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                  V: tl.constexpr, C: tl.constexpr, BK: tl.constexpr,
                  BV: tl.constexpr, SCALE: tl.constexpr):
    pc = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    iv = tl.program_id(2).to(tl.int64)
    b = bh // HV
    hv = bh - b * HV
    h = hv // (HV // H)
    t = pc * C + tl.arange(0, C)
    r = tl.arange(0, C)
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mt = t < T
    qq = tl.load(q + ((b * T + t) * H + h)[:, None] * K + ok[None, :],
                 mask=mt[:, None] & (ok[None, :] < K), other=0.0) * SCALE
    kk = tl.load(k + ((b * T + t) * H + h)[:, None] * K + ok[None, :],
                 mask=mt[:, None] & (ok[None, :] < K), other=0.0)
    state = tl.load(states + ((((b * triton.cdiv(T, C) + pc) * HV + hv) * K + ok)[:, None] * V + ov[None, :]),
                    mask=(ok[:, None] < K) & (ov[None, :] < V), other=0.0)
    zz = tl.load(Z + ((b * T + t) * HV + hv)[:, None] * V + ov[None, :],
                 mask=mt[:, None] & (ov[None, :] < V), other=0.0)
    gg = tl.load(g + (b * T + t) * HV + hv, mask=mt, other=0.0).to(tl.float32)
    cs = tl.cumsum(gg, axis=0)
    p = qq * _safe_exp(cs)[:, None]
    tri = mt[:, None] & mt[None, :] & (r[:, None] >= r[None, :])
    qm = tl.dot(qq.to(tl.float32), tl.trans(kk).to(tl.float32), input_precision="ieee")
    qm = tl.where(tri, qm * _safe_exp(cs[:, None] - cs[None, :]), 0.0)
    oo = tl.dot(p, state, input_precision="ieee") + tl.dot(qm, zz, input_precision="ieee")
    tl.store(out + ((b * T + t) * HV + hv)[:, None] * V + ov[None, :], oo,
             mask=mt[:, None] & (ov[None, :] < V))


@triton.jit
def _chunk_backward(q, k, v, g, beta, states, Z, dout,
                    dq_part, dk_part, dv, dg_part, db_part,
                    T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                    K: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr,
                    SCALE: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    iv = tl.program_id(1).to(tl.int64)
    b = bh // HV
    hv = bh - b * HV
    h = hv // (HV // H)
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mkv = (ok[:, None] < K) & (ov[None, :] < V)
    adj = tl.zeros((BK, BV), tl.float32)
    rr = tl.arange(0, C)
    for pc0 in range(triton.cdiv(T, C)):
        pc = triton.cdiv(T, C) - 1 - pc0
        tt = pc * C + rr
        mt = tt < T
        start = tl.load(states + ((((b * triton.cdiv(T, C) + pc) * HV + hv) * K + ok)[:, None] * V + ov[None, :]), mask=mkv, other=0.0)
        kc = tl.load(k + ((b * T + tt) * H + h)[:, None] * K + ok[None, :], mask=mt[:, None] & (ok[None, :] < K), other=0.0)
        zc = tl.load(Z + ((b * T + tt) * HV + hv)[:, None] * V + ov[None, :], mask=mt[:, None] & (ov[None, :] < V), other=0.0)
        gc = tl.load(g + (b * T + tt) * HV + hv, mask=mt, other=0.0).to(tl.float32)
        cs = tl.cumsum(gc, axis=0)
        for j0 in range(C):
            j = C - 1 - j0
            t = pc * C + j
            valid = t < T
            kj = tl.sum(tl.where((rr == j)[:, None], kc, 0.0), axis=0)
            zj = tl.sum(tl.where((rr == j)[:, None], zc, 0.0), axis=0)
            gj = tl.sum(tl.where(rr == j, gc, 0.0), axis=0)
            cj = tl.sum(tl.where(rr == j, cs, 0.0), axis=0)
            weights = tl.where((rr < j) & mt, _safe_exp(cj - cs), 0.0)
            sprime = start * _safe_exp(cj) + tl.dot(tl.trans(kc * weights[:, None]), zc, input_precision="ieee")
            state = sprime + kj[:, None] * zj[None, :]
            vv = tl.load(v + ((b * T + t) * HV + hv) * V + ov, mask=valid & (ov < V), other=0.0).to(tl.float32)
            bb = tl.load(beta + (b * T + t) * HV + hv, mask=valid, other=0.0).to(tl.float32)
            pred = tl.sum(sprime * kj[:, None], axis=0)
            residual = vv - pred
            do = tl.load(dout + ((b * T + t) * HV + hv) * V + ov, mask=valid & (ov < V), other=0.0).to(tl.float32)
            qq = tl.load(q + ((b * T + t) * H + h) * K + ok, mask=valid & (ok < K), other=0.0).to(tl.float32) * SCALE
            adj += qq[:, None] * do[None, :]
            dqv = tl.sum(state * do[None, :], axis=1) * SCALE
            ddelta = tl.sum(adj * kj[:, None], axis=0)
            dvv = ddelta * bb
            dp = -dvv
            dkv = tl.sum(adj * zj[None, :] + sprime * dp[None, :], axis=1)
            dbv = tl.sum(ddelta * residual, axis=0)
            dadj = adj + kj[:, None] * dp[None, :]
            dgv = tl.sum(tl.sum(dadj * sprime, axis=1), axis=0)
            adj = dadj * _safe_exp(gj)
            po = ((((b * T + t) * HV + hv) * NV + iv) * K) + ok
            tl.store(dq_part + po, dqv, mask=valid & (ok < K))
            tl.store(dk_part + po, dkv, mask=valid & (ok < K))
            tl.store(dv + ((b * T + t) * HV + hv) * V + ov, dvv, mask=valid & (ov < V))
            ps = ((b * T + t) * HV + hv) * NV + iv
            tl.store(dg_part + ps, dgv, mask=valid)
            tl.store(db_part + ps, dbv, mask=valid)


@triton.jit
def _reduce_qk(dqp, dkp, dq, dk, T: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, NV: tl.constexpr):
    bt = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    ok = tl.arange(0, BK)
    aq = tl.zeros((BK,), tl.float32)
    ak = tl.zeros((BK,), tl.float32)
    groups = HV // H
    for j in range(groups):
        hv = h * groups + j
        for iv in range(NV):
            p = (((bt * HV + hv) * NV + iv) * K) + ok
            aq += tl.load(dqp + p, mask=ok < K, other=0.0)
            ak += tl.load(dkp + p, mask=ok < K, other=0.0)
    tl.store(dq + (bt * H + h) * K + ok, aq, mask=ok < K)
    tl.store(dk + (bt * H + h) * K + ok, ak, mask=ok < K)


@triton.jit
def _reduce_scalars(dgp, dbp, dg, db, N: tl.constexpr, NV: tl.constexpr, BN: tl.constexpr):
    x = tl.program_id(0) * BN + tl.arange(0, BN)
    m = x < N
    ag = tl.zeros((BN,), tl.float32)
    ab = tl.zeros((BN,), tl.float32)
    for iv in range(NV):
        ag += tl.load(dgp + x * NV + iv, mask=m, other=0.0)
        ab += tl.load(dbp + x * NV + iv, mask=m, other=0.0)
    tl.store(dg + x, ag, mask=m)
    tl.store(db + x, ab, mask=m)


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        C = CHUNK
        NC = triton.cdiv(T, C)
        BK = triton.next_power_of_2(K)
        BV = VALUE_BLOCK
        NV = triton.cdiv(V, BV)
        A = torch.empty((B, NC, HV, C, C), device=q.device, dtype=torch.float32)
        W = torch.empty((B, T, HV, K), device=q.device, dtype=q.dtype)
        U = torch.empty((B, T, HV, V), device=q.device, dtype=v.dtype)
        Z = torch.empty((B, T, HV, V), device=q.device, dtype=torch.float32)
        states = torch.empty((B, NC, HV, K, V), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        _prepare_chunks[(NC, B * HV)](k, v, g, beta, A, W, U, T=T, H=H, HV=HV, K=K, V=V, C=C, BK=BK, BV=BV, num_warps=8)
        _state_forward[(B * HV, NV)](k, g, W, U, states, Z, T=T, H=H, HV=HV, K=K, V=V, C=C, BK=BK, BV=BV, num_warps=8)
        _chunk_output[(NC, B * HV, NV)](q, k, g, states, Z, out, T=T, H=H, HV=HV, K=K, V=V, C=C, BK=BK, BV=BV, SCALE=K ** -0.5, num_warps=8)
        ctx.save_for_backward(q, k, v, g, beta, states, Z)
        ctx.meta = (B, T, H, HV, K, V, C, BK, BV, NV)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, g, beta, states, Z = ctx.saved_tensors
        B, T, H, HV, K, V, C, BK, BV, NV = ctx.meta
        dout = dout.contiguous()
        dqp = torch.empty((B, T, HV, NV, K), device=q.device, dtype=q.dtype)
        dkp = torch.empty((B, T, HV, NV, K), device=q.device, dtype=k.dtype)
        dv = torch.empty_like(v)
        dgp = torch.empty((B, T, HV, NV), device=q.device, dtype=torch.float32)
        dbp = torch.empty((B, T, HV, NV), device=q.device, dtype=torch.float32)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dg = torch.empty_like(g, dtype=torch.float32)
        db = torch.empty_like(beta)
        _chunk_backward[(B * HV, NV)](q, k, v, g, beta, states, Z, dout, dqp, dkp, dv, dgp, dbp, T=T, H=H, HV=HV, K=K, V=V, C=C, BK=BK, BV=BV, NV=NV, SCALE=K ** -0.5, num_warps=8)
        _reduce_qk[(B * T, H)](dqp, dkp, dq, dk, T=T, H=H, HV=HV, K=K, BK=BK, NV=NV, num_warps=4)
        n = B * T * HV
        _reduce_scalars[(triton.cdiv(n, 256),)](dgp, dbp, dg, db, N=n, NV=NV, BN=256, num_warps=4)
        return dq, dk, dv, dg, db


def gated_delta_rule(q, k, v, g, beta):
    return _GatedDeltaRule.apply(q, k, v, g, beta)

