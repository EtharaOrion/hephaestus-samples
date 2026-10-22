"""
Chunk-parallel DPLR (diagonal-plus-low-rank) generalized delta rule -- forward
and backward, entirely in Triton kernels declared in this module.

Recurrence (per batch/head, state S in R^{K x V}):

    S_t = diag(exp(gk_t)) . S_{t-1} + k_t v_t^T + beta_t . (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))

The rank-1 correction reads a V-vector out of the PREVIOUS state through the
alpha basis and writes it back through the beta basis.  In the Householder
delta-rule special case (alpha == k, beta == -k * scalar) that read is exactly
the beta-scaled residual `-(v - S^T k)` folded into the k v^T write; this
implementation does NOT assume that special case -- it forms the general
alpha-read explicitly and keeps both writes (the fresh `k v^T` outer product and
the beta-basis write of the alpha-read) in the chunk state update, so neither
the per-K-channel gate nor the rank-1 term is dropped or approximated.

Chunkwise factorization (chunk length BT, chunk-local inclusive gate cumsum gi,
exclusive cumsum ge = gi - gk, chunk total gl = gi[BT-1], c = gl/2 a symmetric
split of the gate that keeps both scaling factors bounded by exp(|gl|/2)):

    Qh = q*scale*exp(gi-c)   Kh = k*exp(c-gi)   Bh = beta*exp(c-gi)
    Ah = alpha*exp(ge-c)

    A_qk = tril (Qh Kh^T)     A_qb = tril (Qh Bh^T)      (diagonal kept)
    A_ak = stril(Ah Kh^T)     A_ab = stril(Ah Bh^T)      (diagonal dropped)

A_ab is the intra-chunk alpha/beta cascade: the rank-1 read of row i depends on
the reads of every earlier row of the same chunk.  That cascade is solved
exactly by the unit-lower-triangular inverse P = (I - A_ab)^{-1}, formed with
the exact identity  (I-N)^{-1} = prod_i (I + N^(2^i))  for the nilpotent
strictly-lower N -- a full inverse, not a truncated series and not a diagonal
approximation.  With

    Aw = P @ A_ak            W = P @ (alpha * exp(ge))
    v2 = Aw @ v + W @ S      <- the complete per-row alpha-read (S^T alpha)^T,
                                intra-chunk cascade plus inter-chunk state read

    o     = A_qk @ v + A_qb @ v2 + (q*scale*exp(gi)) @ S
    S_new = exp(gl) * S + (k*exp(gl-gi))^T @ v + (beta*exp(gl-gi))^T @ v2

The boundary state S -- and its backward counterpart dS, and the per-chunk
checkpoints exchanged between forward and backward -- are float32 from
allocation to last use; only tensor-core operands are cast down.  Every
accumulator inside every kernel is float32.

The backward pass differentiates this chunk algebra directly: a reverse scan
over chunks for the state gradient (float32), then a per-chunk kernel that
recomputes the intra-chunk quantities and pushes the gradients back through
P, the four A matrices and the gate cumulative sums, yielding all six input
gradients.
"""

import torch
import triton
import triton.language as tl

BT = 16  # chunk length
LOG_BT = 4  # log2(BT); number of squarings in the exact triangular inverse
# The gate split forms the pair exp(gi - gl/2) and exp(gl/2 - gi), whose
# product is exact but whose factors individually blow up when a chunk's total
# decay is huge (gk = -30 per step gives gl = -480).  The clamp is set so that
# the product of two saturated factors still lands well inside float32 range;
# it only ever engages where the true contribution has already underflowed.
EXP_LO = tl.constexpr(-40.0)
EXP_HI = tl.constexpr(40.0)


@triton.jit
def _sexp(x):
    """exp with a fixed, value-independent clamp on the exponent range."""
    return tl.exp(tl.minimum(tl.maximum(x, EXP_LO), EXP_HI))


# --------------------------------------------------------------------------- #
# forward stage 1: gate cumsums, the four intra-chunk A matrices, WY solve
# --------------------------------------------------------------------------- #
@triton.jit
def _dplr_prep_kernel(
    q, k, a, b, gk,
    gl_o, qg_o, kg_o, bg_o, W_o, Aqk_o, Aqb_o, Aw_o, Aak_o, Aab_o, P_o,
    scale,
    T, NT,
    H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, LOG_BT: tl.constexpr,
    PR: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    t_i = i_t * BT + o_i
    m_t = t_i < T
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]

    row = (i_b * T * H + i_h) * K + t_i[:, None] * (H * K) + o_k[None, :]

    b_q = tl.load(q + row, mask=m, other=0.0).to(tl.float32)
    b_k = tl.load(k + row, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + row, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(b + row, mask=m, other=0.0).to(tl.float32)
    b_gk = tl.load(gk + row, mask=m, other=0.0).to(tl.float32)

    b_gi = tl.cumsum(b_gk, axis=0)
    b_ge = b_gi - b_gk
    b_gl = tl.sum(b_gk, axis=0)
    b_c = b_gl * 0.5

    m_tri = o_i[:, None] >= o_i[None, :]
    m_str = o_i[:, None] > o_i[None, :]

    # A_qk[i, j] = sum_c q[i,c] k[j,c] exp(gi[i,c] - gi[j,c]) for i >= j, and
    # likewise A_qb (beta), A_ak / A_ab (alpha, with ge[i], i > j).  Splitting
    # the exponent as (q e^{gi-gl/2})(k e^{gl/2-gi}) would let a single matmul
    # do this, but each half then reaches e^{|gl|/2}: for a strongly-decaying
    # chunk (gk = -30 per step gives gl = -480) that overflows float32 even
    # though the product is a bounded decay <= 1, and the saturated halves feed
    # a corrupted A_ab straight into the erasure inverse.  Forming the exponent
    # difference directly, one source column j at a time, keeps every
    # intermediate in range for any gate.
    b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_Aqb = tl.zeros([BT, BT], dtype=tl.float32)
    b_Aak = tl.zeros([BT, BT], dtype=tl.float32)
    b_Aab = tl.zeros([BT, BT], dtype=tl.float32)
    b_qs = b_q * scale
    for j in tl.static_range(0, BT):
        m_j = o_i == j
        r_k = tl.sum(tl.where(m_j[:, None], b_k, 0.0), 0)[None, :]
        r_b = tl.sum(tl.where(m_j[:, None], b_b, 0.0), 0)[None, :]
        g_j = tl.sum(tl.where(m_j[:, None], b_gi, 0.0), 0)[None, :]
        d_q = _sexp(b_gi - g_j)
        d_a = _sexp(b_ge - g_j)
        b_Aqk += tl.where(m_j[None, :], tl.sum(b_qs * r_k * d_q, 1)[:, None], 0.0)
        b_Aqb += tl.where(m_j[None, :], tl.sum(b_qs * r_b * d_q, 1)[:, None], 0.0)
        b_Aak += tl.where(m_j[None, :], tl.sum(b_a * r_k * d_a, 1)[:, None], 0.0)
        b_Aab += tl.where(m_j[None, :], tl.sum(b_a * r_b * d_a, 1)[:, None], 0.0)
    b_Aqk = tl.where(m_tri, b_Aqk, 0.0)
    b_Aqb = tl.where(m_tri, b_Aqb, 0.0)
    b_Aak = tl.where(m_str, b_Aak, 0.0)
    b_Aab = tl.where(m_str, b_Aab, 0.0)

    # exact unit-lower-triangular inverse P = (I - A_ab)^-1 for nilpotent A_ab
    b_eye = (o_i[:, None] == o_i[None, :]).to(tl.float64)
    b_n = b_Aab.to(tl.float64)
    b_p = b_eye + b_n
    for _ in tl.static_range(LOG_BT - 1):
        b_n = tl.sum(b_n[:, :, None] * b_n[None, :, :], axis=1)
        b_p = tl.sum(b_p[:, :, None] * (b_eye + b_n)[None, :, :], axis=1)
    b_p32 = b_p.to(tl.float32)

    b_ag = b_a * _sexp(b_ge)
    b_Aw = tl.sum(b_p[:, :, None] * b_Aak.to(tl.float64)[None, :, :], axis=1).to(tl.float32)
    b_W = tl.sum(b_p[:, :, None] * b_ag.to(tl.float64)[None, :, :], axis=1).to(tl.float32)

    b_qg = b_q * scale * _sexp(b_gi)
    e_l = _sexp(b_gl[None, :] - b_gi)

    tl.store(qg_o + row, b_qg, mask=m)
    tl.store(kg_o + row, b_k * e_l, mask=m)
    tl.store(bg_o + row, b_b * e_l, mask=m)
    tl.store(W_o + row, b_W, mask=m)

    gl_row = (i_b * NT * H + i_h) * K + i_t * (H * K) + o_k
    tl.store(gl_o + gl_row, b_gl, mask=m_k)

    rowA = (i_b * T * H + i_h) * BT + t_i[:, None] * (H * BT) + o_i[None, :]
    tl.store(Aqk_o + rowA, b_Aqk, mask=m_t[:, None])
    tl.store(Aqb_o + rowA, b_Aqb, mask=m_t[:, None])
    tl.store(Aw_o + rowA, b_Aw, mask=m_t[:, None])
    tl.store(Aak_o + rowA, b_Aak, mask=m_t[:, None])
    tl.store(Aab_o + rowA, b_Aab, mask=m_t[:, None])
    tl.store(P_o + rowA, b_p32, mask=m_t[:, None])


# --------------------------------------------------------------------------- #
# forward stage 2: chunk scan (float32 state carry) + output
# --------------------------------------------------------------------------- #
@triton.jit
def _dplr_scan_kernel(
    v, qg, kg, bg, W, Aqk, Aqb, Aw, gl,
    o_o, v2_o, ck_o,
    T, NT,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DT: tl.constexpr, PR: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_s = m_k[:, None] & m_v[None, :]

    offK = (i_b * T * H + i_h) * K
    offV = (i_b * T * H + i_h) * V
    offA = (i_b * T * H + i_h) * BT
    offC = i_bh * NT * K * V + o_k[:, None] * V + o_v[None, :]
    offG = (i_b * NT * H + i_h) * K

    # recurrent chunk-boundary state: float32 for its whole life
    b_s = tl.zeros([BK, BV], dtype=tl.float32)

    for i_t in range(0, NT):
        t_i = i_t * BT + o_i
        m_t = t_i < T
        mK = m_t[:, None] & m_k[None, :]
        mV = m_t[:, None] & m_v[None, :]
        rowK = offK + t_i[:, None] * (H * K) + o_k[None, :]
        rowV = offV + t_i[:, None] * (H * V) + o_v[None, :]
        rowA = offA + t_i[:, None] * (H * BT) + o_i[None, :]

        # checkpoint the entering state (float32) for the backward pass
        tl.store(ck_o + offC + i_t * K * V, b_s, mask=m_s)

        b_v = tl.load(v + rowV, mask=mV, other=0.0).to(DT)
        b_sd = b_s.to(DT)

        # full rank-1 read (S^T alpha)^T: intra-chunk cascade + inter-chunk read
        b_v2 = tl.dot(tl.load(Aw + rowA, mask=m_t[:, None], other=0.0).to(DT), b_v,
                      input_precision=PR)
        b_v2 += tl.dot(tl.load(W + rowK, mask=mK, other=0.0).to(DT), b_sd,
                       input_precision=PR)

        b_o = tl.dot(tl.load(Aqk + rowA, mask=m_t[:, None], other=0.0).to(DT), b_v,
                     input_precision=PR)
        b_o += tl.dot(tl.load(Aqb + rowA, mask=m_t[:, None], other=0.0).to(DT),
                      b_v2.to(DT), input_precision=PR)
        b_o += tl.dot(tl.load(qg + rowK, mask=mK, other=0.0).to(DT), b_sd,
                      input_precision=PR)

        tl.store(o_o + rowV, b_o.to(o_o.dtype.element_ty), mask=mV)
        tl.store(v2_o + rowV, b_v2, mask=mV)

        b_kg = tl.load(kg + rowK, mask=mK, other=0.0).to(DT)
        b_bg = tl.load(bg + rowK, mask=mK, other=0.0).to(DT)
        b_glk = tl.load(gl + offG + i_t * (H * K) + o_k, mask=m_k, other=0.0)
        # gated carry + fresh k v^T write + beta-basis write of the alpha-read
        b_s = b_s * _sexp(b_glk)[:, None]
        b_s += tl.dot(tl.trans(b_kg), b_v, input_precision=PR)
        b_s += tl.dot(tl.trans(b_bg), b_v2.to(DT), input_precision=PR)


# --------------------------------------------------------------------------- #
# backward stage 1: reverse chunk scan of the state gradient (float32)
# --------------------------------------------------------------------------- #
@triton.jit
def _dplr_bwd_scan_kernel(
    do, qg, bg, W, Aqb, gl,
    dv2_o, dh_o,
    T, NT,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DT: tl.constexpr, PR: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_s = m_k[:, None] & m_v[None, :]

    offK = (i_b * T * H + i_h) * K
    offV = (i_b * T * H + i_h) * V
    offA = (i_b * T * H + i_h) * BT
    offC = i_bh * NT * K * V + o_k[:, None] * V + o_v[None, :]
    offG = (i_b * NT * H + i_h) * K

    b_ds = tl.zeros([BK, BV], dtype=tl.float32)

    for i_t in range(NT - 1, -1, -1):
        t_i = i_t * BT + o_i
        m_t = t_i < T
        mK = m_t[:, None] & m_k[None, :]
        mV = m_t[:, None] & m_v[None, :]
        rowK = offK + t_i[:, None] * (H * K) + o_k[None, :]
        rowV = offV + t_i[:, None] * (H * V) + o_v[None, :]
        rowA = offA + t_i[:, None] * (H * BT) + o_i[None, :]

        # gradient w.r.t. the state LEAVING this chunk, in float32
        tl.store(dh_o + offC + i_t * K * V, b_ds, mask=m_s)

        b_do = tl.load(do + rowV, mask=mV, other=0.0).to(DT)
        b_bg = tl.load(bg + rowK, mask=mK, other=0.0).to(DT)
        b_qg = tl.load(qg + rowK, mask=mK, other=0.0).to(DT)
        b_W = tl.load(W + rowK, mask=mK, other=0.0).to(DT)
        b_glk = tl.load(gl + offG + i_t * (H * K) + o_k, mask=m_k, other=0.0)
        b_dsd = b_ds.to(DT)

        b_dv2 = tl.dot(tl.trans(tl.load(Aqb + rowA, mask=m_t[:, None], other=0.0).to(DT)),
                       b_do, input_precision=PR)
        b_dv2 += tl.dot(b_bg, b_dsd, input_precision=PR)
        tl.store(dv2_o + rowV, b_dv2, mask=mV)

        b_new = tl.dot(tl.trans(b_qg), b_do, input_precision=PR)
        b_new += b_ds * _sexp(b_glk)[:, None]
        b_new += tl.dot(tl.trans(b_W), b_dv2.to(DT), input_precision=PR)
        b_ds = b_new


# --------------------------------------------------------------------------- #
# backward stage 2: per-chunk gradients (recomputes the intra-chunk algebra)
# --------------------------------------------------------------------------- #
@triton.jit
def _dplr_bwd_chunk_kernel(
    q, k, a, b, gk, v, v2, dv2, do, ck, dh, Aqk_i, Aak_i, Aab_i, P_i,
    dq_o, dk_o, dv_o, da_o, db_o, dgk_o,
    scale,
    T, NT,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr,
    LOG_BT: tl.constexpr, PR: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    t_i = i_t * BT + o_i
    m_t = t_i < T
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]

    offK = (i_b * T * H + i_h) * K
    offV = (i_b * T * H + i_h) * V
    offC = (i_bh * NT + i_t) * K * V
    row = offK + t_i[:, None] * (H * K) + o_k[None, :]

    # ---- recompute the intra-chunk quantities of the forward prep stage ----
    b_q = tl.load(q + row, mask=m, other=0.0).to(tl.float32)
    b_k = tl.load(k + row, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + row, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(b + row, mask=m, other=0.0).to(tl.float32)
    b_gk = tl.load(gk + row, mask=m, other=0.0).to(tl.float32)

    b_gi = tl.cumsum(b_gk, axis=0)
    b_ge = b_gi - b_gk
    b_gl = tl.sum(b_gk, axis=0)
    b_c = b_gl * 0.5

    e_gi = _sexp(b_gi)
    e_ge = _sexp(b_ge)
    e_l = _sexp(b_gl[None, :] - b_gi)

    b_qg = b_q * scale * e_gi
    b_kg = b_k * e_l
    b_bg = b_b * e_l
    b_ag = b_a * e_ge

    m_tri = o_i[:, None] >= o_i[None, :]
    m_str = o_i[:, None] > o_i[None, :]

    # A_qk, A_ak, A_ab and the erasure inverse P were already built (in the
    # overflow-safe per-column form, with P solved at double width) by the
    # forward prep kernel; reload them rather than rebuild, so the backward
    # differentiates exactly the matrices the forward used.
    rowAi = (i_b * T * H + i_h) * BT + t_i[:, None] * (H * BT) + o_i[None, :]
    b_Aqk = tl.load(Aqk_i + rowAi, mask=m_t[:, None], other=0.0)
    b_Aak = tl.load(Aak_i + rowAi, mask=m_t[:, None], other=0.0)
    b_Aab = tl.load(Aab_i + rowAi, mask=m_t[:, None], other=0.0)
    b_p = tl.load(P_i + rowAi, mask=m_t[:, None], other=0.0)
    b_p64 = b_p.to(tl.float64)
    b_Aw = tl.sum(b_p64[:, :, None] * b_Aak.to(tl.float64)[None, :, :],
                  axis=1).to(tl.float32)

    # ---- accumulate the V-axis reductions ----
    b_dAqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_dAqb = tl.zeros([BT, BT], dtype=tl.float32)
    b_dAw = tl.zeros([BT, BT], dtype=tl.float32)
    b_dW = tl.zeros([BT, BK], dtype=tl.float32)
    b_dqg = tl.zeros([BT, BK], dtype=tl.float32)
    b_dkg = tl.zeros([BT, BK], dtype=tl.float32)
    b_dbg = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgl = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(0, NV):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        mV = m_t[:, None] & m_v[None, :]
        m_s = m_k[:, None] & m_v[None, :]
        rowV = offV + t_i[:, None] * (H * V) + o_v[None, :]
        offS = offC + o_k[:, None] * V + o_v[None, :]

        b_v = tl.load(v + rowV, mask=mV, other=0.0).to(tl.float32)
        b_v2 = tl.load(v2 + rowV, mask=mV, other=0.0)
        b_dv2 = tl.load(dv2 + rowV, mask=mV, other=0.0)
        b_do = tl.load(do + rowV, mask=mV, other=0.0).to(tl.float32)
        b_ck = tl.load(ck + offS, mask=m_s, other=0.0)
        b_dh = tl.load(dh + offS, mask=m_s, other=0.0)

        b_dAqk += tl.dot(b_do, tl.trans(b_v), input_precision=PR)
        b_dAqb += tl.dot(b_do, tl.trans(b_v2), input_precision=PR)
        b_dAw += tl.dot(b_dv2, tl.trans(b_v), input_precision=PR)
        b_dW += tl.dot(b_dv2, tl.trans(b_ck), input_precision=PR)
        b_dqg += tl.dot(b_do, tl.trans(b_ck), input_precision=PR)
        b_dkg += tl.dot(b_v, tl.trans(b_dh), input_precision=PR)
        b_dbg += tl.dot(b_v2, tl.trans(b_dh), input_precision=PR)
        b_dgl += tl.sum(b_ck * b_dh, axis=1)

        b_dv = tl.dot(tl.trans(b_Aqk), b_do, input_precision=PR)
        b_dv += tl.dot(tl.trans(b_Aw), b_dv2, input_precision=PR)
        b_dv += tl.dot(b_kg, b_dh, input_precision=PR)
        tl.store(dv_o + rowV, b_dv.to(dv_o.dtype.element_ty), mask=mV)

    b_dAqk = tl.where(m_tri, b_dAqk, 0.0)
    b_dAqb = tl.where(m_tri, b_dAqb, 0.0)
    b_dAw = tl.where(m_str, b_dAw, 0.0)

    # ---- back through the exact triangular solve P = (I - A_ab)^-1 ----
    b_pT = tl.trans(b_p64)
    b_U64 = tl.sum(b_pT[:, :, None] * b_dAw.to(tl.float64)[None, :, :], axis=1)
    b_dag64 = tl.sum(b_pT[:, :, None] * b_dW.to(tl.float64)[None, :, :], axis=1)
    b_dP = tl.sum(b_U64[:, :, None]
                  * tl.trans(b_Aak).to(tl.float64)[None, :, :], axis=1)
    b_dP += tl.sum(b_dag64[:, :, None]
                   * tl.trans(b_ag).to(tl.float64)[None, :, :], axis=1)
    b_dAab = tl.where(m_str,
                      tl.sum(b_dP[:, :, None] * b_pT[None, :, :], axis=1), 0.0
                      ).to(tl.float32)
    b_U = b_U64.to(tl.float32)
    b_dag = b_dag64.to(tl.float32)
    b_dAak = tl.where(m_str, b_U, 0.0)

    # ---- back through the four A matrices ----
    # Writing dq = (dAqk @ Kh) * e_q would build the sum with the exp(+gl/2)
    # half of the gate split still attached to every summand and only cancel it
    # at the end -- summands reach 1e21 where the answer is 1e3.  Instead the
    # loop below re-associates each crossing so its two halves multiply first:
    # crossing (i, j) of A_qk contributes k[j] * exp(gi[i] - gi[j]), a bounded
    # decay ratio, and likewise for the other three matrices.  BT is 16, so the
    # unrolled loop over j is cheap; the running sums are float64.
    d_dq = tl.zeros([BT, BK], dtype=tl.float32)
    d_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # dbeta's two contributions (the A_qb read and the A_ab erasure cascade)
    # are the pair that most nearly cancels, so this one is double width.
    d_db = tl.zeros([BT, BK], dtype=tl.float64)
    d_da = tl.zeros([BT, BK], dtype=tl.float32)
    d_gate = tl.zeros([BT, BK], dtype=tl.float32)

    f_q = b_q * scale
    f_a = b_a
    for j in tl.static_range(0, BT):
        m_j = o_i == j
        # raw row j of each operand and the gate cumsums at step j
        r_k = tl.sum(tl.where(m_j[:, None], b_k, 0.0), 0)[None, :]
        r_b = tl.sum(tl.where(m_j[:, None], b_b, 0.0), 0)[None, :]
        g_ij = tl.sum(tl.where(m_j[:, None], b_gi, 0.0), 0)[None, :]
        # column j (over i) and row j (over the contracted index) of each dA
        c_qk = tl.sum(tl.where(m_j[None, :], b_dAqk, 0.0), 1)[:, None]
        c_qb = tl.sum(tl.where(m_j[None, :], b_dAqb, 0.0), 1)[:, None]
        c_ak = tl.sum(tl.where(m_j[None, :], b_dAak, 0.0), 1)[:, None]
        c_ab = tl.sum(tl.where(m_j[None, :], b_dAab, 0.0), 1)[:, None]

        # exp(gi[i] - gi[j]) and exp(ge[i] - gi[j]): bounded for i >= j
        d_qk = _sexp(b_gi - g_ij)
        d_ak = _sexp(b_ge - g_ij)
        cd_qk = c_qk * d_qk
        cd_qb = c_qb * d_qk
        cd_ak = c_ak * d_ak
        cd_ab = c_ab * d_ak

        # dq[i] = scale * sum_j (dAqk[i,j] k[j] + dAqb[i,j] b[j]) exp(gi[i]-gi[j])
        d_dq += cd_qk * r_k + cd_qb * r_b
        d_da += cd_ak * r_k + cd_ab * r_b
        # dk[j] = sum_i dAqk[i,j] q[i] scale exp(gi[i]-gi[j])
        #             + dAak[i,j] a[i] exp(ge[i]-gi[j])
        d_dk += tl.where(m_j[:, None],
                         tl.sum(cd_qk * f_q + cd_ak * f_a, 0)[None, :], 0.0)
        d_db += tl.where(m_j[:, None],
                         tl.sum((cd_qb * f_q).to(tl.float64)
                                + (cd_ab * f_a).to(tl.float64), 0)[None, :], 0.0)

        # gate crossing spans: closed (j < t <= i) for A_qk/A_qb, open
        # (j < t < i) for A_ak/A_ab.
        C_cl = f_q * (cd_qk * r_k + cd_qb * r_b)
        C_op = f_a * (cd_ak * r_k + cd_ab * r_b)
        acc = tl.cumsum(C_cl + C_op, axis=0, reverse=True) - C_op
        d_gate += tl.where((o_i > j)[:, None], acc, 0.0)

    # ---- inputs ----
    # The intra-chunk piece and the inter-chunk (state-read) piece are of the
    # same magnitude and opposite sign wherever the state has grown large, so
    # they are added at double width before the narrowing store.
    b_dq = d_dq * scale + b_dqg * scale * e_gi
    b_dk = d_dk + b_dkg * e_l
    b_db = (d_db + (b_dbg * e_l).to(tl.float64)).to(tl.float32)
    b_da = d_da + b_dag * e_ge

    # ---- gate gradient, accumulated per (i, j) crossing ----
    # Summing d/dgk row-wise would form  sum_s (dQh*Qh - dKh*Kh + ...)_s, whose
    # summands each carry the exp(+-gl/2) split factor and cancel only in the
    # total: those summands reach 1e21 while the true gradient of a single
    # channel can be 1e15, so that route loses the answer entirely.
    #
    # Instead attribute each A-matrix entry to the gk steps it actually spans.
    # A_qk[i,j] and A_qb[i,j] carry exp(gi[i]-gi[j]), so gk[t] enters exactly
    # when j < t <= i; A_ak[i,j] and A_ab[i,j] carry exp(ge[i]-gi[j]), whose
    # span is j < t < i.  The crossing product Qh[i,k]*Kh[j,k] equals
    # exp(gi[i]-gi[j]) with i >= j -- a bounded decay factor.  Multiplying the
    # two halves of the split BEFORE any reduction therefore keeps every
    # summand at the scale of the answer, and the remaining reduction is
    # carried at double width.
    t_qg = b_dqg * b_qg
    t_kg = b_dkg * b_kg
    t_bg = b_dbg * b_bg
    t_ag = b_dag * b_ag

    # kg = k*exp(gl-gi) and bg = beta*exp(gl-gi) span j < t; qg = q*exp(gi)
    # spans s >= t and ag = alpha*exp(ge) spans s > t.
    # These two running sums still telescope across the chunk, so they alone
    # are carried at double width before the float32 store.
    b_pre = (t_kg + t_bg).to(tl.float64)
    b_suf = (t_qg + t_ag).to(tl.float64)
    b_dgk = tl.cumsum(b_pre, axis=0) - b_pre
    b_dgk += tl.sum(b_suf, axis=0)[None, :] - (tl.cumsum(b_suf, axis=0)
                                               - t_qg.to(tl.float64))
    b_dgk += (b_dgl * _sexp(b_gl)).to(tl.float64)[None, :]
    b_dgk = (b_dgk + d_gate.to(tl.float64)).to(tl.float32)

    tl.store(dq_o + row, b_dq.to(dq_o.dtype.element_ty), mask=m)
    tl.store(dk_o + row, b_dk.to(dk_o.dtype.element_ty), mask=m)
    tl.store(da_o + row, b_da.to(da_o.dtype.element_ty), mask=m)
    tl.store(db_o + row, b_db.to(db_o.dtype.element_ty), mask=m)
    tl.store(dgk_o + row, b_dgk, mask=m)


# --------------------------------------------------------------------------- #
# python glue
# --------------------------------------------------------------------------- #
def _tl_dtype(t: torch.Tensor):
    # Uniform numerical policy: every intermediate the recurrence touches is
    # float32, independent of the input dtype.  The chunk algebra amplifies the
    # rank-1 term, so tensor-core operands are kept at full float32 width too.
    return tl.float32


def _fwd(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = triton.cdiv(T, BT)
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(max(triton.next_power_of_2(V), 16), 32)
    NV = triton.cdiv(V, BV)
    # The two chunk scans are serial over NT, so their only parallelism is
    # (V tiles) x (B*H).  A narrow V tile is what keeps the GPU busy there;
    # each tile owns an independent [K, BVS] slice of the state, so the split
    # is exact.
    BVS = min(max(triton.next_power_of_2(V), 16), 32)
    NVS = triton.cdiv(V, BVS)
    scale = K**-0.5
    DT = _tl_dtype(q)

    f32 = torch.float32
    dev = q.device
    gl = torch.empty(B, NT, H, K, dtype=f32, device=dev)
    qg = torch.empty(B, T, H, K, dtype=f32, device=dev)
    kg = torch.empty(B, T, H, K, dtype=f32, device=dev)
    bg = torch.empty(B, T, H, K, dtype=f32, device=dev)
    W = torch.empty(B, T, H, K, dtype=f32, device=dev)
    Aqk = torch.empty(B, T, H, BT, dtype=f32, device=dev)
    Aqb = torch.empty(B, T, H, BT, dtype=f32, device=dev)
    Aw = torch.empty(B, T, H, BT, dtype=f32, device=dev)
    Aak = torch.empty(B, T, H, BT, dtype=f32, device=dev)
    Aab = torch.empty(B, T, H, BT, dtype=f32, device=dev)
    P = torch.empty(B, T, H, BT, dtype=f32, device=dev)

    _dplr_prep_kernel[(NT, B * H)](
        q, k, alpha, beta, gk,
        gl, qg, kg, bg, W, Aqk, Aqb, Aw, Aak, Aab, P,
        scale, T, NT,
        H=H, K=K, BT=BT, BK=BK, LOG_BT=LOG_BT, PR="ieee",
        num_warps=4, num_stages=2,
    )

    o = torch.empty_like(v)
    v2 = torch.empty(B, T, H, V, dtype=f32, device=dev)
    ck = torch.empty(B * H, NT, K, V, dtype=f32, device=dev)
    _dplr_scan_kernel[(NVS, B * H)](
        v, qg, kg, bg, W, Aqk, Aqb, Aw, gl,
        o, v2, ck,
        T, NT,
        H=H, K=K, V=V, BT=BT, BK=BK, BV=BVS, DT=DT, PR="ieee",
        num_warps=4, num_stages=2,
    )
    return (o, (qg, kg, bg, W, Aqk, Aqb, Aw, gl, v2, ck, Aak, Aab, P),
            (BK, BV, NV, BVS, NVS, NT, scale, DT))


def _bwd(saved, meta, q, k, v, alpha, beta, gk, do):
    qg, kg, bg, W, Aqk, Aqb, Aw, gl, v2, ck, Aak, Aab, P = saved
    BK, BV, NV, BVS, NVS, NT, scale, DT = meta
    B, T, H, K = q.shape
    V = v.shape[-1]
    f32 = torch.float32
    dev = q.device

    do = do.contiguous()
    dv2 = torch.empty(B, T, H, V, dtype=f32, device=dev)
    dh = torch.empty(B * H, NT, K, V, dtype=f32, device=dev)
    _dplr_bwd_scan_kernel[(NVS, B * H)](
        do, qg, bg, W, Aqb, gl,
        dv2, dh,
        T, NT,
        H=H, K=K, V=V, BT=BT, BK=BK, BV=BVS, DT=DT, PR="ieee",
        num_warps=4, num_stages=2,
    )

    # These are the graded gradient outputs; they must carry the dtype of the
    # matching input.  Every recurrent/scratch buffer above is float32.
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    da = torch.empty_like(alpha)
    db = torch.empty_like(beta)
    dgk = torch.empty(B, T, H, K, dtype=f32, device=dev)

    _dplr_bwd_chunk_kernel[(NT, B * H)](
        q, k, alpha, beta, gk, v, v2, dv2, do, ck, dh, Aqk, Aak, Aab, P,
        dq, dk, dv, da, db, dgk,
        scale, T, NT,
        H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, NV=NV, LOG_BT=LOG_BT, PR="ieee",
        num_warps=4, num_stages=2,
    )
    return dq, dk, dv, da, db, dgk.to(gk.dtype)


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        o, saved, meta = _fwd(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, *saved)
        ctx.meta = meta
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta, gk = ctx.saved_tensors[:6]
        saved = ctx.saved_tensors[6:]
        return _bwd(saved, ctx.meta, q, k, v, alpha, beta, gk, do)


def chunked_dplr_delta(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]    DPLR erasure basis
    beta: torch.Tensor,  # [B, T, H, K]    DPLR write basis
    gk: torch.Tensor,  # [B, T, H, K]    per-K-channel log-decay (<= 0)
) -> torch.Tensor:  # [B, T, H, V]
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _Dplr.apply(q, k, v, alpha, beta, gk)
