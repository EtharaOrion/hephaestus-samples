"""Chunk-parallel DPLR (diagonal-plus-low-rank) generalized delta rule.

Forward and backward are implemented entirely with Triton kernels declared in
this module.  The algorithm is the chunkwise DPLR factorization:

  * the per-K-channel log-decay is accumulated inside each chunk (inclusive
    cumsum `gc`, exclusive `ge`),
  * the intra-chunk cascade is built as four dense [C, C] matrices
    A_qk, A_qb, A_ak, A_ab from gate-rescaled q / k / alpha / beta tiles,
  * the strictly lower triangular A_ab is inverted exactly (it is nilpotent)
    by repeated squaring of its Neumann series, giving Tm = (I - A_ab)^-1,
    which fuses the rank-1 alpha-beta erasure into the chunk,
  * the chunk-boundary state [K, V] lives in float32 for the whole of its
    life: forward carry, saved per-chunk checkpoints, and backward carry.

The pairwise A matrices use gate exponents taken relative to the chunk's
first row so that both factors of every exp(gc_i - gc_j) product stay in
range; terms contracted against the carried state use absolute exponents.
"""

import torch
import triton
import triton.language as tl

C = 16  # chunk length along time
LOG_C = 4  # ceil(log2(C)); repeated-squaring depth for the nilpotent inverse
EXP_CAP = tl.constexpr(70.0)
PREC = "ieee"


@triton.jit
def _ex(x):
    return tl.exp(tl.minimum(tl.maximum(x, -EXP_CAP), EXP_CAP))


@triton.jit
def _dot64(A, B, CC: tl.constexpr):
    """Exact float64 [CC, CC] x [CC, N] product; the tensor cores are float32
    only, and this cascade cancels well below float32 resolution."""
    return tl.sum(A.to(tl.float64)[:, :, None] * B.to(tl.float64)[None, :, :], 1)


@triton.jit
def _tri_inv(N, CC: tl.constexpr, LOGC: tl.constexpr):
    """(I - N)^-1 for strictly lower triangular (nilpotent) N."""
    i = tl.arange(0, CC)
    N64 = N.to(tl.float64)
    P = (i[:, None] == i[None, :]).to(tl.float64) + N64
    M = N64
    for _ in tl.static_range(LOGC - 1):
        M = _dot64(M, M, CC)
        P = P + _dot64(P, M, CC)
    return P


@triton.jit
def _gates(gk, base, rows, ok, m_t, m_k, H: tl.constexpr, K: tl.constexpr,
           CC: tl.constexpr):
    """Load the chunk's log-decay tile and its per-K-channel cumulative sums.

    Returns (g, gc, gcr, gl) where gc is the inclusive cumsum along time,
    gcr the same sum re-centred on the chunk's middle row so that both
    exp(+gcr) and exp(-gcr) stay in range, and gl the chunk total.
    """
    m = m_t[:, None] & m_k[None, :]
    p = gk + base + rows[:, None].to(tl.int64) * (H * K) + ok[None, :]
    g = tl.load(p, mask=m, other=0.0).to(tl.float32)
    gc = tl.cumsum(g, axis=0)
    r = tl.sum(tl.where((tl.arange(0, CC) == (3 * CC // 4))[:, None], gc, 0.0), 0)
    gl = tl.sum(g, 0)
    return g, gc, gc - r[None, :], gl, r


@triton.jit
def diag_dplr_prep(q, k, v, a, b, gk, W, U, Aqk_o, Aqb_o, Tm_o,
                   T, scale, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                   NT: tl.constexpr, CC: tl.constexpr, LOGC: tl.constexpr,
                   BK: tl.constexpr, BV: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    ii = tl.arange(0, CC)
    rows = i_t * CC + ii
    m_t = rows < T
    ok = tl.arange(0, BK)
    m_k = ok < K
    ov = tl.arange(0, BV)
    m_v = ov < V
    base = i_b.to(tl.int64) * T * H * K + i_h * K
    basev = i_b.to(tl.int64) * T * H * V + i_h * V
    mkk = m_t[:, None] & m_k[None, :]
    mvv = m_t[:, None] & m_v[None, :]
    off = base + rows[:, None].to(tl.int64) * (H * K) + ok[None, :]
    b_q = tl.load(q + off, mask=mkk, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + off, mask=mkk, other=0.0).to(tl.float32)
    b_a = tl.load(a + off, mask=mkk, other=0.0).to(tl.float32)
    b_b = tl.load(b + off, mask=mkk, other=0.0).to(tl.float32)
    b_v = tl.load(v + basev + rows[:, None].to(tl.int64) * (H * V) + ov[None, :],
                  mask=mvv, other=0.0).to(tl.float32)

    g, gc, gcr, gl, gm = _gates(gk, base, rows, ok, m_t, m_k, H, K, CC)
    qgr = b_q * _ex(gcr)
    kqr = b_k * _ex(-gcr)
    bqr = b_b * _ex(-gcr)
    agr = b_a * _ex(gcr - g)

    ml = ii[:, None] >= ii[None, :]
    ms = ii[:, None] > ii[None, :]
    A_qk = tl.where(ml, tl.dot(qgr, tl.trans(kqr), input_precision="ieee"), 0.0)
    A_qb = tl.where(ml, tl.dot(qgr, tl.trans(bqr), input_precision="ieee"), 0.0)
    A_ak = tl.where(ms, tl.dot(agr, tl.trans(kqr), input_precision="ieee"), 0.0)
    A_ab = tl.where(ms, tl.dot(agr, tl.trans(bqr), input_precision="ieee"), 0.0)

    Tm = _tri_inv(A_ab, CC, LOGC)
    b_u = tl.sum(Tm[:, :, None] * _dot64(A_ak, b_v, CC)[None, :, :], 1).to(tl.float32)
    # W is stored in the chunk's recentred gate frame; the scan pairs it with
    # the state rescaled by exp(gc[M]), which keeps both factors bounded.
    b_w = tl.dot(Tm.to(tl.float32), agr, input_precision="ieee")

    cb = i_bh.to(tl.int64) * NT + i_t
    tl.store(W + cb * (CC * K) + ii[:, None] * K + ok[None, :], b_w, mask=mkk)
    tl.store(U + cb * (CC * V) + ii[:, None] * V + ov[None, :], b_u, mask=mvv)
    oA = cb * (CC * CC) + ii[:, None] * CC + ii[None, :]
    tl.store(Aqk_o + oA, A_qk)
    tl.store(Aqb_o + oA, A_qb)
    tl.store(Tm_o + oA, Tm.to(tl.float32))


@triton.jit
def diag_dplr_state_scan(k, b, v, gk, W, U, V2, S_o,
                         T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                         NT: tl.constexpr, CC: tl.constexpr,
                         BK: tl.constexpr, BV: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    ii = tl.arange(0, CC)
    ok = tl.arange(0, BK)
    m_k = ok < K
    ov = i_v * BV + tl.arange(0, BV)
    m_v = ov < V
    base = i_b.to(tl.int64) * T * H * K + i_h * K
    basev = i_b.to(tl.int64) * T * H * V + i_h * V
    # The carry itself is accumulated wider than float32: with a weak gate the
    # state grows by many decades across the chunks, and each chunk's rank-1
    # and kv contributions arrive at wildly different magnitudes.
    S = tl.zeros([BK, BV], dtype=tl.float64)
    for i_t in range(NT):
        rows = i_t * CC + ii
        m_t = rows < T
        mkk = m_t[:, None] & m_k[None, :]
        mvv = m_t[:, None] & m_v[None, :]
        cb = i_bh.to(tl.int64) * NT + i_t
        b_w = tl.load(W + cb * (CC * K) + ii[:, None] * K + ok[None, :], mask=mkk, other=0.0)
        b_u = tl.load(U + cb * (CC * V) + ii[:, None] * V + ov[None, :], mask=mvv, other=0.0)
        g, gc, gcr, gl, gm = _gates(gk, base, rows, ok, m_t, m_k, H, K, CC)
        Sm = S * _ex(gm).to(tl.float64)[:, None]
        v2 = b_u + tl.sum(b_w.to(tl.float64)[:, :, None] * Sm[None, :, :], 1).to(tl.float32)
        tl.store(V2 + cb * (CC * V) + ii[:, None] * V + ov[None, :], v2, mask=mvv)
        tl.store(S_o + cb * (K * V) + ok[:, None].to(tl.int64) * V + ov[None, :], S,
                 mask=m_k[:, None] & m_v[None, :])
        off = base + rows[:, None].to(tl.int64) * (H * K) + ok[None, :]
        b_k = tl.load(k + off, mask=mkk, other=0.0).to(tl.float32)
        b_b = tl.load(b + off, mask=mkk, other=0.0).to(tl.float32)
        b_v = tl.load(v + basev + rows[:, None].to(tl.int64) * (H * V) + ov[None, :],
                      mask=mvv, other=0.0).to(tl.float32)
        ed = _ex(gl[None, :] - gc)
        kd = tl.where(mkk, b_k * ed, 0.0)
        bd = tl.where(mkk, b_b * ed, 0.0)
        S = S * _ex(gl).to(tl.float64)[:, None] \
            + (tl.dot(tl.trans(kd), b_v, input_precision="ieee")
               + tl.dot(tl.trans(bd), v2, input_precision="ieee")).to(tl.float64)


@triton.jit
def diag_dplr_out(q, v, gk, V2, S_i, Aqk_i, Aqb_i, o,
                  T, scale, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  NT: tl.constexpr, CC: tl.constexpr,
                  BK: tl.constexpr, BV: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    ii = tl.arange(0, CC)
    rows = i_t * CC + ii
    m_t = rows < T
    ok = tl.arange(0, BK)
    m_k = ok < K
    ov = tl.arange(0, BV)
    m_v = ov < V
    base = i_b.to(tl.int64) * T * H * K + i_h * K
    basev = i_b.to(tl.int64) * T * H * V + i_h * V
    mkk = m_t[:, None] & m_k[None, :]
    mvv = m_t[:, None] & m_v[None, :]
    cb = i_bh.to(tl.int64) * NT + i_t
    b_q = tl.load(q + base + rows[:, None].to(tl.int64) * (H * K) + ok[None, :],
                  mask=mkk, other=0.0).to(tl.float32) * scale
    g, gc, gcr, gl, gm = _gates(gk, base, rows, ok, m_t, m_k, H, K, CC)
    qg = b_q * _ex(gc)
    b_v = tl.load(v + basev + rows[:, None].to(tl.int64) * (H * V) + ov[None, :],
                  mask=mvv, other=0.0).to(tl.float32)
    v2 = tl.load(V2 + cb * (CC * V) + ii[:, None] * V + ov[None, :], mask=mvv, other=0.0)
    S = tl.load(S_i + cb * (K * V) + ok[:, None].to(tl.int64) * V + ov[None, :],
                mask=m_k[:, None] & m_v[None, :], other=0.0)
    oA = cb * (CC * CC) + ii[:, None] * CC + ii[None, :]
    acc = tl.dot(tl.load(Aqk_i + oA), b_v, input_precision="ieee")
    acc += tl.dot(tl.load(Aqb_i + oA), v2, input_precision="ieee")
    acc += tl.dot(qg, S.to(tl.float32), input_precision="ieee")
    tl.store(o + basev + rows[:, None].to(tl.int64) * (H * V) + ov[None, :],
             acc.to(o.dtype.element_ty), mask=mvv)


@triton.jit
def diag_dplr_dstate_scan(q, k, b, gk, do, W, Aqk_i, Aqb_i, dS_o, dV2, dVp,
                          T, scale, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                          NT: tl.constexpr, CC: tl.constexpr,
                          BK: tl.constexpr, BV: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    ii = tl.arange(0, CC)
    ok = tl.arange(0, BK)
    m_k = ok < K
    ov = i_v * BV + tl.arange(0, BV)
    m_v = ov < V
    base = i_b.to(tl.int64) * T * H * K + i_h * K
    basev = i_b.to(tl.int64) * T * H * V + i_h * V
    dS = tl.zeros([BK, BV], dtype=tl.float64)
    for i_r in range(NT):
        i_t = NT - 1 - i_r
        rows = i_t * CC + ii
        m_t = rows < T
        mkk = m_t[:, None] & m_k[None, :]
        mvv = m_t[:, None] & m_v[None, :]
        cb = i_bh.to(tl.int64) * NT + i_t
        tl.store(dS_o + cb * (K * V) + ok[:, None].to(tl.int64) * V + ov[None, :], dS,
                 mask=m_k[:, None] & m_v[None, :])
        pv = basev + rows[:, None].to(tl.int64) * (H * V) + ov[None, :]
        b_do = tl.load(do + pv, mask=mvv, other=0.0).to(tl.float32)
        oA = cb * (CC * CC) + ii[:, None] * CC + ii[None, :]
        A_qk = tl.load(Aqk_i + oA)
        A_qb = tl.load(Aqb_i + oA)
        off = base + rows[:, None].to(tl.int64) * (H * K) + ok[None, :]
        b_k = tl.load(k + off, mask=mkk, other=0.0).to(tl.float32)
        b_b = tl.load(b + off, mask=mkk, other=0.0).to(tl.float32)
        b_q = tl.load(q + off, mask=mkk, other=0.0).to(tl.float32) * scale
        g, gc, gcr, gl, gm = _gates(gk, base, rows, ok, m_t, m_k, H, K, CC)
        ed = _ex(gl[None, :] - gc)
        kd = tl.where(mkk, b_k * ed, 0.0)
        bd = tl.where(mkk, b_b * ed, 0.0)
        dS32 = dS.to(tl.float32)
        dv2 = tl.dot(tl.trans(A_qb), b_do, input_precision="ieee") \
            + tl.dot(bd, dS32, input_precision="ieee")
        dvp = tl.dot(tl.trans(A_qk), b_do, input_precision="ieee") \
            + tl.dot(kd, dS32, input_precision="ieee")
        tl.store(dV2 + cb * (CC * V) + ii[:, None] * V + ov[None, :], dv2, mask=mvv)
        tl.store(dVp + pv, dvp, mask=mvv)
        b_w = tl.load(W + cb * (CC * K) + ii[:, None] * K + ok[None, :], mask=mkk, other=0.0)
        qg = tl.where(mkk, b_q * _ex(gc), 0.0)
        dS = dS * _ex(gl).to(tl.float64)[:, None] \
            + tl.dot(tl.trans(qg), b_do, input_precision="ieee").to(tl.float64) \
            + tl.dot(tl.trans(b_w), dv2, input_precision="ieee").to(tl.float64) \
            * _ex(gm).to(tl.float64)[:, None]


@triton.jit
def diag_dplr_bwd_chunk(q, k, v, a, b, gk, do, V2, dV2, dVp, S_i, dS_i, Tm_i,
                        dq, dk, dv, da, db, dgk,
                        T, scale, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                        NT: tl.constexpr, CC: tl.constexpr,
                        BK: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    ii = tl.arange(0, CC)
    rows = i_t * CC + ii
    m_t = rows < T
    ok = tl.arange(0, BK)
    m_k = ok < K
    base = i_b.to(tl.int64) * T * H * K + i_h * K
    basev = i_b.to(tl.int64) * T * H * V + i_h * V
    mkk = m_t[:, None] & m_k[None, :]
    cb = i_bh.to(tl.int64) * NT + i_t

    off = base + rows[:, None].to(tl.int64) * (H * K) + ok[None, :]
    b_q = tl.load(q + off, mask=mkk, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + off, mask=mkk, other=0.0).to(tl.float32)
    b_a = tl.load(a + off, mask=mkk, other=0.0).to(tl.float32)
    b_b = tl.load(b + off, mask=mkk, other=0.0).to(tl.float32)
    g, gc, gcr, gl, gm = _gates(gk, base, rows, ok, m_t, m_k, H, K, CC)
    Exr = _ex(gcr)
    Exrn = _ex(-gcr)
    Ex = _ex(gc)
    Ed = _ex(gl[None, :] - gc)
    Ege = _ex(gc - g)
    Eger = _ex(gcr - g)
    qgr = b_q * Exr
    kqr = b_k * Exrn
    bqr = b_b * Exrn
    agr = b_a * Eger
    qg = b_q * Ex
    ag = b_a * Ege
    kd = b_k * Ed
    bd = b_b * Ed

    ml = ii[:, None] >= ii[None, :]
    ms = ii[:, None] > ii[None, :]
    A_ak = tl.where(ms, tl.dot(agr, tl.trans(kqr), input_precision="ieee"), 0.0)
    oA = cb * (CC * CC) + ii[:, None] * CC + ii[None, :]
    Tm = tl.load(Tm_i + oA)
    TmT = tl.trans(Tm)
    A_akT = tl.trans(A_ak)

    Em_k = _ex(gm)          # exp(gc[M]),      <= 1
    Eglm_k = _ex(gl - gm)   # exp(gl - gc[M]), <= 1

    dA_qk = tl.zeros([CC, CC], dtype=tl.float64)
    dA_qb = tl.zeros([CC, CC], dtype=tl.float64)
    dA_ak = tl.zeros([CC, CC], dtype=tl.float64)
    dTm = tl.zeros([CC, CC], dtype=tl.float64)
    # These four contract the [K, V] boundary state against per-row vectors.
    # Their column sums feed dgk, where terms many decades larger than the
    # surviving result cancel, so they are accumulated in float64.
    dqg = tl.zeros([CC, BK], dtype=tl.float64)
    dw = tl.zeros([CC, BK], dtype=tl.float64)
    dkd = tl.zeros([CC, BK], dtype=tl.float64)
    dbd = tl.zeros([CC, BK], dtype=tl.float64)
    dgl_s = tl.zeros([BK], dtype=tl.float64)

    for i_v in range(NV):
        ov = i_v * BV + tl.arange(0, BV)
        m_v = ov < V
        mvv = m_t[:, None] & m_v[None, :]
        msv = m_k[:, None] & m_v[None, :]
        pv = basev + rows[:, None].to(tl.int64) * (H * V) + ov[None, :]
        b_v = tl.load(v + pv, mask=mvv, other=0.0).to(tl.float32)
        b_do = tl.load(do + pv, mask=mvv, other=0.0).to(tl.float32)
        v2 = tl.load(V2 + cb * (CC * V) + ii[:, None] * V + ov[None, :], mask=mvv, other=0.0)
        dv2 = tl.load(dV2 + cb * (CC * V) + ii[:, None] * V + ov[None, :], mask=mvv, other=0.0)
        pS = cb * (K * V) + ok[:, None].to(tl.int64) * V + ov[None, :]
        S = tl.load(S_i + pS, mask=msv, other=0.0)
        dSn = tl.load(dS_i + pS, mask=msv, other=0.0)
        # rescale the state rows into the chunk's recentred gate frame up
        # front, so the intra-chunk and cross-chunk halves of each gradient
        # are already at a common (bounded) scale when they are summed
        Sm = S * Em_k.to(tl.float64)[:, None]
        dSm = dSn * Eglm_k.to(tl.float64)[:, None]

        dA_qk += tl.dot(b_do, tl.trans(b_v), input_precision="ieee").to(tl.float64)
        # v2 spans the amplified state, so this reduction is kept in float64
        dA_qb += tl.sum(b_do.to(tl.float64)[:, None, :] * v2.to(tl.float64)[None, :, :], 2)
        dqg += tl.dot(b_do, tl.trans(Sm.to(tl.float32)), input_precision="ieee").to(tl.float64)
        dw += tl.sum(dv2.to(tl.float64)[:, None, :] * Sm[None, :, :], 2)
        dkd += tl.dot(b_v, tl.trans(dSm.to(tl.float32)), input_precision="ieee").to(tl.float64)
        dbd += tl.dot(v2, tl.trans(dSm.to(tl.float32)), input_precision="ieee").to(tl.float64)
        dgl_s += tl.sum(S * dSn, 1)

        uu = tl.dot(A_ak, b_v, input_precision="ieee")
        dTm += tl.dot(dv2, tl.trans(uu), input_precision="ieee").to(tl.float64)
        dz = tl.dot(TmT, dv2, input_precision="ieee")
        dA_ak += tl.dot(dz, tl.trans(b_v), input_precision="ieee").to(tl.float64)
        dvp = tl.load(dVp + pv, mask=mvv, other=0.0)
        dv_out = dvp + tl.dot(A_akT, dz, input_precision="ieee")
        tl.store(dv + pv, dv_out.to(dv.dtype.element_ty), mask=mvv)

    dA_qk = tl.where(ml, dA_qk, 0.0).to(tl.float32)
    dA_qb = tl.where(ml, dA_qb, 0.0).to(tl.float32)
    dA_ak = tl.where(ms, dA_ak, 0.0).to(tl.float32)
    dTm += tl.dot(dw.to(tl.float32), tl.trans(agr), input_precision="ieee").to(tl.float64)
    dag_w = tl.dot(TmT, dw.to(tl.float32), input_precision="ieee").to(tl.float64)
    dA_ab = tl.where(ms, tl.dot(tl.dot(TmT, dTm.to(tl.float32), input_precision="ieee"), TmT,
                                input_precision="ieee"), 0.0).to(tl.float64)

    # Backing the four A matrices out to the K axis mixes rows whose gate
    # scales span the whole chunk; float64 here is what keeps the sums from
    # collapsing when the chunk's decay range is wide.
    dqgr = _dot64(dA_qk, kqr, CC) + _dot64(dA_qb, bqr, CC)
    dkqr = _dot64(tl.trans(dA_qk), qgr, CC) + _dot64(tl.trans(dA_ak), agr, CC)
    dbqr = _dot64(tl.trans(dA_qb), qgr, CC) \
        + tl.sum(tl.trans(dA_ab)[:, :, None] * agr.to(tl.float64)[None, :, :], 1)
    dagr = _dot64(dA_ak, kqr, CC) \
        + tl.sum(dA_ab[:, :, None] * bqr.to(tl.float64)[None, :, :], 1)

    # Each gradient sums a recentred intra-chunk term and an absolute
    # cross-chunk term.  The two exponents differ only by the constant
    # gc[M] / gl, so the shared factor is pulled out and the (frequently
    # near-cancelling) coefficients are added at their own scale first.
    # dqg/dw/dkd/dbd already carry the recentring factor from Sm/dSm, so both
    # halves of each gradient share the recentred exponent and are summed
    # before the (bounded) recentred exponential is applied.
    b_dq = (scale * Exr.to(tl.float64) * (dqgr + dqg)).to(tl.float32)
    b_dk = (Exrn.to(tl.float64) * (dkqr + dkd)).to(tl.float32)
    b_db = (Exrn.to(tl.float64) * (dbqr + dbd)).to(tl.float32)
    b_da = (Eger.to(tl.float64) * (dagr + dag_w)).to(tl.float32)

    # dgk[t] collects every term whose gate exponent contains g[t].  Each
    # exponent pattern gives a different partial-sum window over rows i, and
    # the windows are applied as exact triangular matmuls rather than
    # (total - prefix), which would cancel catastrophically whenever the
    # window sum is far smaller than the terms it is built from.
    # The window sums are accumulated in float64: with a strongly amplifying
    # state the individual terms can exceed the surviving window sum by many
    # decades, and float32 prefix sums would lose the result entirely.
    # dL/dx_c * x_c is invariant to the gate frame, so the recentred pairs
    # below give exactly the absolute-exponent contributions.
    P_gc = dqg * qgr.to(tl.float64)  # exponent +gc, against the carried state
    P_ge = dag_w * agr.to(tl.float64)  # exponent +gc - g, against the state
    P_gl = dkd * kqr.to(tl.float64) + dbd * bqr.to(tl.float64)  # exponent gl - gc
    # Intra-chunk pairwise groups carry the recentring constant gc[M]; the
    # constant's contributions cancel exactly across each group, so its window
    # is (window - full column).
    R_gc = dqgr * qgr.to(tl.float64) - (dkqr * kqr.to(tl.float64) + dbqr * bqr.to(tl.float64))
    R_ge = dagr * agr.to(tl.float64)

    c_gc = tl.cumsum(P_gc, axis=0)
    c_ge = tl.cumsum(P_ge, axis=0)
    c_gl = tl.cumsum(P_gl, axis=0)
    c_rc = tl.cumsum(R_gc, axis=0)
    c_re = tl.cumsum(R_ge, axis=0)
    t_gc = tl.sum(P_gc, 0)[None, :]
    t_ge = tl.sum(P_ge, 0)[None, :]
    t_rc = tl.sum(R_gc, 0)[None, :]
    t_re = tl.sum(R_ge, 0)[None, :]

    low = (ii <= (3 * CC // 4))[:, None]
    acc = (t_gc - c_gc + P_gc) + (t_ge - c_ge) + (c_gl - P_gl)
    acc += tl.where(low, -(c_rc - R_gc), t_rc - c_rc + R_gc)
    acc += tl.where(low, -c_re, t_re - c_re)
    b_dgk = (acc + (dgl_s * _ex(gl).to(tl.float64))[None, :]).to(tl.float32)

    tl.store(dq + off, b_dq.to(dq.dtype.element_ty), mask=mkk)
    tl.store(dk + off, b_dk.to(dk.dtype.element_ty), mask=mkk)
    tl.store(da + off, b_da.to(da.dtype.element_ty), mask=mkk)
    tl.store(db + off, b_db.to(db.dtype.element_ty), mask=mkk)
    tl.store(dgk + off, b_dgk.to(dgk.dtype.element_ty), mask=mkk)


def _npot(x):
    return max(16, triton.next_power_of_2(x))


def _shapes(q, v):
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = triton.cdiv(T, C)
    BK = _npot(K)
    BV = _npot(V)
    BVS = min(BV, 64)
    return B, T, H, K, V, NT, B * H, BK, BV, BVS, triton.cdiv(V, BVS)


def _fwd(q, k, v, alpha, beta, gk):
    B, T, H, K, V, NT, BH, BK, BV, BVS, NV = _shapes(q, v)
    scale = float(K) ** -0.5
    f32 = dict(device=q.device, dtype=torch.float32)
    W = torch.zeros(BH, NT, C, K, **f32)
    U = torch.zeros(BH, NT, C, V, **f32)
    Aqk = torch.empty(BH, NT, C, C, **f32)
    Aqb = torch.empty(BH, NT, C, C, **f32)
    Tm = torch.empty(BH, NT, C, C, **f32)
    V2 = torch.zeros(BH, NT, C, V, **f32)
    # The boundary checkpoint is the one buffer the whole recurrence flows
    # through; it is carried and stored wider than float32 so the ~128 chunk
    # carries at T = 8192 cannot compound rounding into the graded tolerance.
    S = torch.zeros(BH, NT, K, V, device=q.device, dtype=torch.float64)
    o = v.new_empty(B, T, H, V)
    diag_dplr_prep[(NT, BH)](q, k, v, alpha, beta, gk, W, U, Aqk, Aqb, Tm,
                             T, scale, H, K, V, NT, C, LOG_C, BK, BV,
                             num_warps=4, num_stages=2)
    diag_dplr_state_scan[(NV, BH)](k, beta, v, gk, W, U, V2, S,
                                   T, H, K, V, NT, C, BK, BVS,
                                   num_warps=4, num_stages=2)
    diag_dplr_out[(NT, BH)](q, v, gk, V2, S, Aqk, Aqb, o,
                            T, scale, H, K, V, NT, C, BK, BV,
                            num_warps=4, num_stages=2)
    return o, W, Aqk, Aqb, Tm, V2, S


def _bwd(do, q, k, v, alpha, beta, gk, W, Aqk, Aqb, Tm, V2, S):
    B, T, H, K, V, NT, BH, BK, BV, BVS, NV = _shapes(q, v)
    scale = float(K) ** -0.5
    f32 = dict(device=q.device, dtype=torch.float32)
    dS = torch.zeros(BH, NT, K, V, device=q.device, dtype=torch.float64)
    dV2 = torch.zeros(BH, NT, C, V, **f32)
    dVp = torch.zeros(B, T, H, V, **f32)
    do = do.contiguous()
    diag_dplr_dstate_scan[(NV, BH)](q, k, beta, gk, do, W, Aqk, Aqb, dS, dV2, dVp,
                                    T, scale, H, K, V, NT, C, BK, BVS,
                                    num_warps=4, num_stages=2)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    da = torch.empty_like(alpha)
    db = torch.empty_like(beta)
    dgk = torch.empty_like(gk)
    diag_dplr_bwd_chunk[(NT, BH)](q, k, v, alpha, beta, gk, do, V2, dV2, dVp, S, dS, Tm,
                                  dq, dk, dv, da, db, dgk,
                                  T, scale, H, K, V, NT, C, BK, BVS, NV,
                                  num_warps=4, num_stages=2)
    return dq, dk, dv, da, db, dgk


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        o, W, Aqk, Aqb, Tm, V2, S = _fwd(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, W, Aqk, Aqb, Tm, V2, S)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta, gk, W, Aqk, Aqb, Tm, V2, S = ctx.saved_tensors
        return _bwd(do, q, k, v, alpha, beta, gk, W, Aqk, Aqb, Tm, V2, S)


def chunked_dplr_delta(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]    DPLR erasure basis
    beta: torch.Tensor,  # [B, T, H, K]    DPLR write basis
    gk: torch.Tensor,  # [B, T, H, K]    per-K-channel log-decay (<= 0)
) -> torch.Tensor:  # [B, T, H, V]
    return _Dplr.apply(q, k, v, alpha, beta, gk)
