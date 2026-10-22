"""Chunk-parallel DPLR (diagonal-plus-low-rank) generalized delta rule.

Forward and backward are Triton kernels working over time chunks of BT steps.
Per chunk the per-K-channel log-decay is accumulated (`gi` inclusive, `ge`
exclusive, held in log2 units), the four intra-chunk interaction matrices

    A_qk[i,j] = <q_i, k_j> exp2(gi_i - gi_j)        i >= j
    A_qb[i,j] = <q_i, b_j> exp2(gi_i - gi_j)        i >= j
    A_ak[i,j] = <a_i, k_j> exp2(ge_i - gi_j)        i >  j
    A_ab[i,j] = <a_i, b_j> exp2(ge_i - gi_j)        i >  j

are formed with tensor cores, the strictly-lower A_ab is inverted exactly by
forward substitution to P = (I - A_ab)^-1, and the resulting (w, u) pair drives
a chunk-level recurrence whose state is carried in float32 for its whole life.
All exponent arguments are clamped to a bounded range before exp2.
"""

import torch
import triton
import triton.language as tl

RCP_LN2 = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)
EXP_CLAMP = tl.constexpr(60.0)

BT = 8


@triton.jit
def _clamp(x):
    return tl.minimum(tl.maximum(x, -EXP_CLAMP), EXP_CLAMP)


@triton.jit
def _exp2d(x):
    """exp2 of a bounded argument, evaluated at float64 width.

    The chunk-state and intra-chunk halves of every output cancel to several
    digits, and they reach the same gate through different exponent expressions;
    matching them to float64 keeps that cancellation from dominating.
    """
    return tl.math.exp2(_clamp(x).to(tl.float64))


@triton.jit
def _mask_lo(x, strict: tl.constexpr):
    i = tl.arange(0, x.shape[0])[:, None]
    j = tl.arange(0, x.shape[1])[None, :]
    if strict:
        return tl.where(i > j, x, 0.0)
    return tl.where(i >= j, x, 0.0)


@triton.jit
def dplr_fwd_intra_kernel(
    q, k, a, b, gk, v,
    qg, kg, ag, bg, gi_o, Aqk, Aqb, Aak, Ainv, w, u,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    scale: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DOT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NT = tl.num_programs(0)
    i_b = i_bh // H
    i_h = i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_c = tl.arange(0, BT)
    m_t = o_t < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]

    base_k = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]

    b_g = tl.load(gk + base_k, mask=m_tk, other=0.0).to(tl.float32) * RCP_LN2
    b_gi = tl.cumsum(b_g, axis=0)
    b_ge = b_gi - b_g
    b_gl = tl.sum(tl.where((o_c == BT - 1)[:, None], b_gi, 0.0), 0)

    # gk <= 0, so gi and ge are non-increasing and gl <= gi: every gate factor
    # below is a bounded exp2 of a non-positive, clamped argument.
    f_i = _exp2d(b_gi)
    f_e = _exp2d(b_ge)
    f_d = _exp2d(b_gl[None, :] - b_gi)

    b_q = tl.load(q + base_k, mask=m_tk, other=0.0).to(tl.float64) * scale
    b_k = tl.load(k + base_k, mask=m_tk, other=0.0).to(tl.float64)
    b_a = tl.load(a + base_k, mask=m_tk, other=0.0).to(tl.float64)
    b_b = tl.load(b + base_k, mask=m_tk, other=0.0).to(tl.float64)

    q_ops = b_q * f_i
    a_ops = b_a * f_e

    tl.store(qg + base_k, q_ops, mask=m_tk)
    tl.store(ag + base_k, a_ops, mask=m_tk)
    tl.store(kg + base_k, b_k * f_d, mask=m_tk)
    tl.store(bg + base_k, b_b * f_d, mask=m_tk)
    tl.store(gi_o + base_k, b_gi, mask=m_tk)

    # The intra-chunk interaction matrices are built one source column j at a time
    # so that the gate factor enters as the bounded difference exp2(g_i - gi_j),
    # never as a product of two separately huge exp2 terms.
    b_Aqk = tl.zeros([BT, BT], dtype=tl.float64)
    b_Aqb = tl.zeros([BT, BT], dtype=tl.float64)
    b_Aak = tl.zeros([BT, BT], dtype=tl.float64)
    b_Aab = tl.zeros([BT, BT], dtype=tl.float64)
    q64 = b_q
    a64 = b_a
    for j in range(BT):
        m_j = o_c == j
        gi_j = tl.sum(tl.where(m_j[:, None], b_gi, 0.0), 0)
        k_j = tl.sum(tl.where(m_j[:, None], b_k, 0.0), 0)
        bb_j = tl.sum(tl.where(m_j[:, None], b_b, 0.0), 0)
        e1 = _exp2d(b_gi - gi_j[None, :])
        e2 = _exp2d(b_ge - gi_j[None, :])
        col_qk = tl.sum(q64 * k_j[None, :] * e1, 1) * (o_c >= j)
        col_qb = tl.sum(q64 * bb_j[None, :] * e1, 1) * (o_c >= j)
        col_ak = tl.sum(a64 * k_j[None, :] * e2, 1) * (o_c > j)
        col_ab = tl.sum(a64 * bb_j[None, :] * e2, 1) * (o_c > j)
        b_Aqk = tl.where(m_j[None, :], col_qk[:, None], b_Aqk)
        b_Aqb = tl.where(m_j[None, :], col_qb[:, None], b_Aqb)
        b_Aak = tl.where(m_j[None, :], col_ak[:, None], b_Aak)
        b_Aab = tl.where(m_j[None, :], col_ab[:, None], b_Aab)

    o_A = ((i_b * NT + i_t) * H + i_h) * BT * BT + o_c[:, None] * BT + o_c[None, :]
    m_A = m_t[:, None]
    tl.store(Aqk + o_A, b_Aqk, mask=m_A)
    tl.store(Aqb + o_A, b_Aqb, mask=m_A)
    tl.store(Aak + o_A, b_Aak, mask=m_A)

    # exact inverse of the unit lower triangular (I - A_ab) by forward substitution
    b_P = b_Aab
    for i in range(1, BT):
        m_r = o_c == i
        row = tl.sum(tl.where(m_r[:, None], b_P, 0.0), 0)
        row = row + tl.sum(row[:, None] * b_P, 0) * (o_c < i)
        b_P = tl.where(m_r[:, None], row, b_P)
    b_P = b_P + (o_c[:, None] == o_c[None, :])
    tl.store(Ainv + o_A, b_P, mask=m_A)

    # w and u seed the chunk-state recurrence; the WY inverse can be badly
    # conditioned, so these two reductions are carried at float64
    b_P64 = b_P
    b_w = tl.sum(b_P64[:, :, None] * a_ops.to(tl.float64)[None, :, :], 1)
    tl.store(w + base_k, b_w, mask=m_tk)

    b_Aak2 = tl.sum(b_P64[:, :, None] * b_Aak[None, :, :], 1)
    o_v = tl.arange(0, BV)
    m_tv = m_t[:, None] & (o_v < V)[None, :]
    base_v = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    b_v = tl.load(v + base_v, mask=m_tv, other=0.0).to(tl.float64)
    b_u = tl.sum(b_Aak2[:, :, None] * b_v[None, :, :], 1)
    tl.store(u + base_v, b_u, mask=m_tv)


@triton.jit
def dplr_fwd_h_kernel(
    kg, bg, v, w, u, gi, h, v_new,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
    BT: tl.constexpr, BTP: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DOT: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float64)
    for i_t in range(NT):
        # rows past the chunk are padding that keeps the dot at the tensor-core
        # minimum tile height; they are masked to zero on load and on store
        o_c = tl.arange(0, BTP)
        o_t = i_t * BT + o_c
        m_t = (o_c < BT) & (o_t < T)
        m_tk = m_t[:, None] & m_k[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        base_k = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        base_v = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_h = ((i_b * NT + i_t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h32 = b_h.to(tl.float32)
        tl.store(h + p_h, b_h, mask=m_h)

        b_w = tl.load(w + base_k, mask=m_tk, other=0.0).to(tl.float32)
        b_u = tl.load(u + base_v, mask=m_tv, other=0.0)
        b_v2 = tl.dot(b_w, b_h32, input_precision="ieee").to(tl.float64) + b_u
        tl.store(v_new + base_v, b_v2, mask=m_tv)

        b_kg = tl.load(kg + base_k, mask=m_tk, other=0.0).to(tl.float32)
        b_bg = tl.load(bg + base_k, mask=m_tk, other=0.0).to(tl.float32)
        b_v = tl.load(v + base_v, mask=m_tv, other=0.0).to(tl.float32)
        b_hc = tl.dot(tl.trans(b_kg), b_v, input_precision="ieee")
        b_hc += tl.dot(tl.trans(b_bg), b_v2.to(tl.float32), input_precision="ieee")

        last = tl.minimum((i_t + 1) * BT, T) - 1
        b_gl = tl.load(gi + (i_b * T * H + i_h) * K + last * (H * K) + o_k, mask=m_k, other=0.0)
        b_h = b_h * _exp2d(b_gl)[:, None] + b_hc.to(tl.float64)


@triton.jit
def dplr_fwd_o_kernel(
    qg, v, v_new, Aqk, Aqb, h, o,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DOT: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    NT = tl.num_programs(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, BT)
    m_t = o_t < T
    m_tk = m_t[:, None] & (o_k < K)[None, :]
    m_tv = m_t[:, None] & (o_v < V)[None, :]
    m_h = (o_k < K)[:, None] & (o_v < V)[None, :]

    base_k = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    base_v = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    p_h = ((i_b * NT + i_t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
    o_A = ((i_b * NT + i_t) * H + i_h) * BT * BT + o_c[:, None] * BT + o_c[None, :]

    b_qg = tl.load(qg + base_k, mask=m_tk, other=0.0)
    b_hs = tl.load(h + p_h, mask=m_h, other=0.0)
    b_Aqk = tl.load(Aqk + o_A, mask=m_t[:, None], other=0.0)
    b_Aqb = tl.load(Aqb + o_A, mask=m_t[:, None], other=0.0)
    b_v = tl.load(v + base_v, mask=m_tv, other=0.0).to(tl.float64)
    b_vn = tl.load(v_new + base_v, mask=m_tv, other=0.0)
    # the state-carry and intra-chunk halves of o cancel heavily once the state
    # has grown, so the reduction is accumulated in float64
    b_o = tl.sum(b_qg[:, :, None] * b_hs[None, :, :], 1)
    b_o += tl.sum(b_Aqk[:, :, None] * b_v[None, :, :], 1)
    b_o += tl.sum(b_Aqb[:, :, None] * b_vn[None, :, :], 1)
    tl.store(o + base_v, b_o.to(o.dtype.element_ty), mask=m_tv)


@triton.jit
def dplr_bwd_dA_kernel(
    do, v, v_new, Aqk, Aqb, dAqk, dAqb, dv2i, dv0,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
    DOT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NT = tl.num_programs(0)
    i_b = i_bh // H
    i_h = i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    o_c = tl.arange(0, BT)
    m_t = o_t < T
    m_tv = m_t[:, None] & (o_v < V)[None, :]
    base_v = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    o_A = ((i_b * NT + i_t) * H + i_h) * BT * BT + o_c[:, None] * BT + o_c[None, :]

    b_do = tl.load(do + base_v, mask=m_tv, other=0.0).to(tl.float64)
    b_v = tl.load(v + base_v, mask=m_tv, other=0.0).to(tl.float64)
    b_vn = tl.load(v_new + base_v, mask=m_tv, other=0.0)
    # dA_qb multiplies the (large) recomputed v_new and later cancels against the
    # state-side terms, so these reductions are accumulated in float64
    b_dAqk = _mask_lo(tl.sum(b_do[:, :, None] * tl.trans(b_v)[None, :, :], 1), False)
    b_dAqb = _mask_lo(tl.sum(b_do[:, :, None] * tl.trans(b_vn)[None, :, :], 1), False)
    tl.store(dAqk + o_A, b_dAqk, mask=m_t[:, None])
    tl.store(dAqb + o_A, b_dAqb, mask=m_t[:, None])

    b_Aqk = tl.load(Aqk + o_A, mask=m_t[:, None], other=0.0)
    b_Aqb = tl.load(Aqb + o_A, mask=m_t[:, None], other=0.0)
    tl.store(dv2i + base_v, tl.sum(tl.trans(b_Aqb)[:, :, None] * b_do[None, :, :], 1), mask=m_tv)
    tl.store(dv0 + base_v, tl.sum(tl.trans(b_Aqk)[:, :, None] * b_do[None, :, :], 1), mask=m_tv)


@triton.jit
def dplr_bwd_dh_kernel(
    qg, bg, w, gi, do, dv2i, dh, dv2,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
    BT: tl.constexpr, BTP: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DOT: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    # the reverse state carry mirrors the forward one and spans the same dynamic
    # range, so it is accumulated in float64
    b_dh = tl.zeros([BK, BV], dtype=tl.float64)
    for i_s in range(NT):
        i_t = NT - 1 - i_s
        o_c = tl.arange(0, BTP)
        o_t = i_t * BT + o_c
        m_t = (o_c < BT) & (o_t < T)
        m_tk = m_t[:, None] & m_k[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        base_k = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        base_v = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        p_dh = ((i_b * NT + i_t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(dh + p_dh, b_dh, mask=m_h)

        b_dh32 = b_dh.to(tl.float32)
        b_bg = tl.load(bg + base_k, mask=m_tk, other=0.0).to(tl.float32)
        b_dv2 = tl.load(dv2i + base_v, mask=m_tv, other=0.0)
        b_dv2 += tl.dot(b_bg, b_dh32, input_precision="ieee").to(tl.float64)
        tl.store(dv2 + base_v, b_dv2, mask=m_tv)

        b_qg = tl.load(qg + base_k, mask=m_tk, other=0.0).to(tl.float32)
        b_do = tl.load(do + base_v, mask=m_tv, other=0.0).to(tl.float32)
        b_w = tl.load(w + base_k, mask=m_tk, other=0.0).to(tl.float32)
        b_acc = tl.dot(tl.trans(b_qg), b_do, input_precision="ieee")
        b_acc += tl.dot(tl.trans(b_w), b_dv2.to(tl.float32), input_precision="ieee")

        last = tl.minimum((i_t + 1) * BT, T) - 1
        b_gl = tl.load(gi + (i_b * T * H + i_h) * K + last * (H * K) + o_k, mask=m_k, other=0.0)
        b_dh = b_dh * _exp2d(b_gl)[:, None] + b_acc.to(tl.float64)


@triton.jit
def dplr_bwd_state_kernel(
    h, dh, do, v, v_new, dv2, kg, gi,
    dqg, dkg, dbg, dw, dv0, dglast,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BTP: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DOT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NT = tl.num_programs(0)
    i_b = i_bh // H
    i_h = i_bh % H

    o_c = tl.arange(0, BTP)
    o_t = i_t * BT + o_c
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_t = (o_c < BT) & (o_t < T)
    m_k = o_k < K
    m_v = o_v < V
    m_tk = m_t[:, None] & m_k[None, :]
    m_tv = m_t[:, None] & m_v[None, :]
    m_h = m_k[:, None] & m_v[None, :]

    base_k = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    base_v = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    p_h = ((i_b * NT + i_t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]

    b_h = tl.load(h + p_h, mask=m_h, other=0.0).to(tl.float32)
    b_dh = tl.load(dh + p_h, mask=m_h, other=0.0).to(tl.float32)
    b_hT = tl.trans(b_h)
    b_dhT = tl.trans(b_dh)

    # dqg and dw reduce against the (large) chunk state and later cancel against
    # the intra-chunk terms, so they use the extended-precision dot
    b_hT64 = b_hT.to(tl.float64)
    b_do = tl.load(do + base_v, mask=m_tv, other=0.0).to(tl.float64)
    tl.store(dqg + base_k, tl.sum(b_do[:, :, None] * b_hT64[None, :, :], 1), mask=m_tk)

    b_dv2 = tl.load(dv2 + base_v, mask=m_tv, other=0.0)
    tl.store(dw + base_k, tl.sum(b_dv2[:, :, None] * b_hT64[None, :, :], 1), mask=m_tk)

    b_v = tl.load(v + base_v, mask=m_tv, other=0.0).to(tl.float32)
    b_vn = tl.load(v_new + base_v, mask=m_tv, other=0.0).to(tl.float32)
    tl.store(dkg + base_k, tl.dot(b_v, b_dhT, input_precision="ieee"), mask=m_tk)
    tl.store(dbg + base_k, tl.dot(b_vn, b_dhT, input_precision="ieee"), mask=m_tk)

    b_kg = tl.load(kg + base_k, mask=m_tk, other=0.0).to(tl.float32)
    b_dv = tl.load(dv0 + base_v, mask=m_tv, other=0.0)
    b_dv += tl.dot(b_kg, b_dh, input_precision="ieee").to(tl.float64)
    tl.store(dv0 + base_v, b_dv, mask=m_tv)

    last = tl.minimum((i_t + 1) * BT, T) - 1
    b_gl = tl.load(gi + (i_b * T * H + i_h) * K + last * (H * K) + o_k, mask=m_k, other=0.0)
    b_dgl = LN2 * _exp2d(b_gl) * tl.sum(b_h.to(tl.float64) * b_dh.to(tl.float64), 1)
    tl.store(dglast + ((i_b * NT + i_t) * H + i_h) * K + o_k, b_dgl, mask=m_k)


@triton.jit
def dplr_bwd_wy_kernel(
    Ainv, Aak, v, ag, dw, dv2, dv0, dAak, dAab, dag,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DOT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NT = tl.num_programs(0)
    i_b = i_bh // H
    i_h = i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    o_c = tl.arange(0, BT)
    m_t = o_t < T
    m_tk = m_t[:, None] & (o_k < K)[None, :]
    m_tv = m_t[:, None] & (o_v < V)[None, :]
    base_k = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    base_v = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    o_A = ((i_b * NT + i_t) * H + i_h) * BT * BT + o_c[:, None] * BT + o_c[None, :]

    # every reduction here runs through the (possibly ill-conditioned) WY inverse
    # and cancels against the state-side terms, so all of it is carried in float64
    b_P = _mask_lo(tl.load(Ainv + o_A, mask=m_t[:, None], other=0.0), False)
    b_Aak = _mask_lo(tl.load(Aak + o_A, mask=m_t[:, None], other=0.0), True)
    b_Pt = tl.trans(b_P)

    b_du = tl.load(dv2 + base_v, mask=m_tv, other=0.0)
    b_v = tl.load(v + base_v, mask=m_tv, other=0.0).to(tl.float64)
    b_dw = tl.load(dw + base_k, mask=m_tk, other=0.0)
    b_ag = tl.load(ag + base_k, mask=m_tk, other=0.0)

    b_dM = tl.sum(b_Pt[:, :, None] * b_du[None, :, :], 1)
    b_dv = tl.load(dv0 + base_v, mask=m_tv, other=0.0)
    b_dv += tl.sum(tl.trans(b_Aak)[:, :, None] * b_dM[None, :, :], 1)
    tl.store(dv0 + base_v, b_dv, mask=m_tv)
    tl.store(dAak + o_A,
             _mask_lo(tl.sum(b_dM[:, :, None] * tl.trans(b_v)[None, :, :], 1), True),
             mask=m_t[:, None])

    tl.store(dag + base_k, tl.sum(b_Pt[:, :, None] * b_dw[None, :, :], 1), mask=m_tk)

    b_M = tl.sum(b_Aak[:, :, None] * b_v[None, :, :], 1)
    b_dP = tl.sum(b_du[:, :, None] * tl.trans(b_M)[None, :, :], 1)
    b_dP += tl.sum(b_dw[:, :, None] * tl.trans(b_ag)[None, :, :], 1)
    b_dAab = tl.sum(tl.sum(b_Pt[:, :, None] * b_dP[None, :, :], 1)[:, :, None] * b_Pt[None, :, :], 1)
    tl.store(dAab + o_A, _mask_lo(b_dAab, True), mask=m_t[:, None])


@triton.jit
def dplr_bwd_intra_kernel(
    q, k, a, b, gk, dAqk, dAqb, dAak, dAab, dqg, dkg, dag, dbg, dglast,
    dq, dk, da, db, dgk,
    T, H: tl.constexpr, K: tl.constexpr,
    scale: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    DOT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    NT = tl.num_programs(0)
    i_b = i_bh // H
    i_h = i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_c = tl.arange(0, BT)
    m_t = o_t < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]
    base_k = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    o_A = ((i_b * NT + i_t) * H + i_h) * BT * BT + o_c[:, None] * BT + o_c[None, :]

    b_g = tl.load(gk + base_k, mask=m_tk, other=0.0).to(tl.float32) * RCP_LN2
    b_gi = tl.cumsum(b_g, axis=0)
    b_ge = b_gi - b_g
    b_gl = tl.sum(tl.where((o_c == BT - 1)[:, None], b_gi, 0.0), 0)

    f_i = _exp2d(b_gi)
    f_e = _exp2d(b_ge)
    f_d = _exp2d(b_gl[None, :] - b_gi)

    b_q = tl.load(q + base_k, mask=m_tk, other=0.0).to(tl.float64) * scale
    b_k = tl.load(k + base_k, mask=m_tk, other=0.0).to(tl.float64)
    b_a = tl.load(a + base_k, mask=m_tk, other=0.0).to(tl.float64)
    b_b = tl.load(b + base_k, mask=m_tk, other=0.0).to(tl.float64)
    q_ops = b_q * f_i
    a_ops = b_a * f_e

    b_dAqk = _mask_lo(tl.load(dAqk + o_A, mask=m_t[:, None], other=0.0), False)
    b_dAqb = _mask_lo(tl.load(dAqb + o_A, mask=m_t[:, None], other=0.0), False)
    b_dAak = _mask_lo(tl.load(dAak + o_A, mask=m_t[:, None], other=0.0), True)
    b_dAab = _mask_lo(tl.load(dAab + o_A, mask=m_t[:, None], other=0.0), True)

    b_dqg = tl.load(dqg + base_k, mask=m_tk, other=0.0)
    b_dag = tl.load(dag + base_k, mask=m_tk, other=0.0)
    b_dkg = tl.load(dkg + base_k, mask=m_tk, other=0.0)
    b_dbg = tl.load(dbg + base_k, mask=m_tk, other=0.0)

    # Pull the four A matrices back column by column.  Each entry carries a factor
    # exp2(gi_i - gi_j) or exp2(ge_i - gi_j), which is <= 1 and is evaluated as a
    # single bounded exponent instead of a ratio of two separately-scaled terms.
    # the intra-chunk and cross-chunk halves of dq / dalpha cancel heavily when the
    # state has grown large, so these accumulators are carried in float64
    d_q = tl.zeros([BT, BK], dtype=tl.float64)
    d_a = tl.zeros([BT, BK], dtype=tl.float64)
    d_k = tl.zeros([BT, BK], dtype=tl.float64)
    d_b = tl.zeros([BT, BK], dtype=tl.float64)
    # gate-pair contribution to dgk, telescoped over the pair indicator j < t <= i
    # the row and column halves of the pair sum cancel to many digits, so this
    # accumulator is carried in float64
    acc = tl.zeros([BT, BK], dtype=tl.float64)
    row2 = tl.zeros([BT, BK], dtype=tl.float64)
    for j in range(BT):
        m_j = o_c == j
        gi_j = tl.sum(tl.where(m_j[:, None], b_gi, 0.0), 0)
        k_j = tl.sum(tl.where(m_j[:, None], b_k, 0.0), 0)
        bb_j = tl.sum(tl.where(m_j[:, None], b_b, 0.0), 0)
        e1d = _exp2d(b_gi - gi_j[None, :]) * (o_c >= j)[:, None]
        e2d = _exp2d(b_ge - gi_j[None, :]) * (o_c > j)[:, None]
        c_qk = tl.sum(tl.where(m_j[None, :], b_dAqk, 0.0), 1).to(tl.float64)[:, None]
        c_qb = tl.sum(tl.where(m_j[None, :], b_dAqb, 0.0), 1).to(tl.float64)[:, None]
        c_ak = tl.sum(tl.where(m_j[None, :], b_dAak, 0.0), 1).to(tl.float64)[:, None]
        c_ab = tl.sum(tl.where(m_j[None, :], b_dAab, 0.0), 1).to(tl.float64)[:, None]

        P1 = (c_qk * k_j[None, :] + c_qb * bb_j[None, :]) * e1d
        P2 = (c_ak * k_j[None, :] + c_ab * bb_j[None, :]) * e2d
        d_q += P1
        d_a += P2
        q64 = b_q
        a64 = b_a
        colk = tl.sum((c_qk * q64) * e1d + (c_ak * a64) * e2d, 0)
        colb = tl.sum((c_qb * q64) * e1d + (c_ab * a64) * e2d, 0)
        d_k += tl.where(m_j[:, None], colk[None, :], tl.zeros_like(d_k))
        d_b += tl.where(m_j[:, None], colb[None, :], tl.zeros_like(d_b))

        Pq = q64 * P1
        Pa = a64 * P2
        P = Pq + Pa
        acc += P - tl.where(m_j[:, None], tl.sum(P, 0)[None, :], tl.zeros_like(P))
        row2 += Pa

    d_q += b_dqg * f_i
    d_a += b_dag * f_e
    d_k += b_dkg.to(tl.float64) * f_d
    d_b += b_dbg.to(tl.float64) * f_d

    tl.store(dq + base_k, (d_q.to(tl.float32) * scale).to(dq.dtype.element_ty), mask=m_tk)
    tl.store(dk + base_k, d_k.to(tl.float32).to(dk.dtype.element_ty), mask=m_tk)
    tl.store(da + base_k, d_a.to(tl.float32).to(da.dtype.element_ty), mask=m_tk)
    tl.store(db + base_k, d_b.to(tl.float32).to(db.dtype.element_ty), mask=m_tk)

    x = tl.where(m_t[:, None], acc, tl.zeros_like(acc))
    p2 = tl.where(m_t[:, None], row2, tl.zeros_like(row2))
    f_pair = -(tl.cumsum(x, axis=0) - x) - p2

    # qg = q exp2(gi_i) is active for t <= i;  ag = a exp2(ge_i) for t < i
    t3 = tl.where(m_t[:, None], b_dqg * q_ops, tl.zeros_like(b_dqg))
    t4 = tl.where(m_t[:, None], b_dag * a_ops, tl.zeros_like(b_dag))
    f_row = tl.cumsum(t3 + t4, axis=0, reverse=True) - t4
    # kg, bg carry exp2(gl - gi_j), whose derivative wrt g_t is active for t > j
    t5 = tl.where(m_t[:, None], (b_dkg.to(tl.float64) * b_k + b_dbg.to(tl.float64) * b_b) * f_d,
                  tl.zeros_like(f_d))
    f_col = tl.cumsum(t5, axis=0) - t5

    b_dgl = tl.load(dglast + ((i_b * NT + i_t) * H + i_h) * K + o_k, mask=m_k, other=0.0)
    d_g = (f_pair + f_row + f_col) * LN2 + b_dgl[None, :]
    tl.store(dgk + base_k, (d_g * RCP_LN2).to(dgk.dtype.element_ty), mask=m_tk)


def _launch_fwd(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    dt = q.dtype
    gdt = gk.dtype
    dev = q.device
    NT = triton.cdiv(T, BT)
    BK = max(triton.next_power_of_2(K), 16)
    BVf = max(triton.next_power_of_2(V), 16)
    DOT = "ieee"  # all intermediates are wider than the inputs; keep dots exact
    BTP = max(BT, 16)  # tensor cores need a reduction tile of at least 16
    scale = K ** -0.5

    ek = (B, T, H, K)
    ev = (B, T, H, V)
    eA = (B, NT, H, BT, BT)
    qg = torch.empty(ek, dtype=torch.float64, device=dev)
    kg = torch.empty(ek, dtype=torch.float64, device=dev)
    ag = torch.empty(ek, dtype=torch.float64, device=dev)
    bg = torch.empty(ek, dtype=torch.float64, device=dev)
    w = torch.empty(ek, dtype=torch.float64, device=dev)
    gi = torch.empty(ek, dtype=torch.float32, device=dev)
    u = torch.empty(ev, dtype=torch.float64, device=dev)
    Aqk = torch.empty(eA, dtype=torch.float64, device=dev)
    Aqb = torch.empty(eA, dtype=torch.float64, device=dev)
    Aak = torch.empty(eA, dtype=torch.float64, device=dev)
    Ainv = torch.empty(eA, dtype=torch.float64, device=dev)

    dplr_fwd_intra_kernel[(NT, B * H)](
        q, k, alpha, beta, gk, v,
        qg, kg, ag, bg, gi, Aqk, Aqb, Aak, Ainv, w, u,
        T, H, K, V, scale, BT, BK, BVf, DOT, num_warps=4, num_stages=2,
    )

    h = torch.empty((B, NT, H, K, V), dtype=torch.float64, device=dev)
    v_new = torch.empty(ev, dtype=torch.float64, device=dev)
    BV = 64 if V > 64 else BVf
    NV = triton.cdiv(V, BV)
    dplr_fwd_h_kernel[(NV, B * H)](
        kg, bg, v, w, u, gi, h, v_new,
        T, H, K, V, NT, BT, BTP, BK, BV, DOT, num_warps=4, num_stages=4,
    )

    o = torch.empty(ev, dtype=dt, device=dev)
    dplr_fwd_o_kernel[(NV, NT, B * H)](
        qg, v, v_new, Aqk, Aqb, h, o,
        T, H, K, V, BT, BK, BV, DOT, num_warps=4, num_stages=2,
    )
    cache = (qg, kg, ag, bg, w, gi, Aqk, Aqb, Aak, Ainv, h, v_new)
    return o, cache, (NT, BK, BVf, BV, NV, DOT, scale, gdt, BTP)


def _launch_bwd(do, q, k, v, alpha, beta, gk, cache, meta):
    B, T, H, K = q.shape
    V = v.shape[-1]
    dt = q.dtype
    dev = q.device
    qg, kg, ag, bg, w, gi, Aqk, Aqb, Aak, Ainv, h, v_new = cache
    NT, BK, BVf, BV, NV, DOT, scale, gdt, BTP = meta

    f32 = torch.float32
    ek32 = (B, T, H, K)
    ev32 = (B, T, H, V)
    eA = (B, NT, H, BT, BT)
    dAqk = torch.empty(eA, dtype=torch.float64, device=dev)
    dAqb = torch.empty(eA, dtype=torch.float64, device=dev)
    dv2i = torch.empty(ev32, dtype=torch.float64, device=dev)
    dv0 = torch.empty(ev32, dtype=torch.float64, device=dev)

    dplr_bwd_dA_kernel[(NT, B * H)](
        do, v, v_new, Aqk, Aqb, dAqk, dAqb, dv2i, dv0,
        T, H, K, V, BT, BVf, DOT, num_warps=4, num_stages=2,
    )

    dh = torch.empty((B, NT, H, K, V), dtype=torch.float64, device=dev)
    dv2 = torch.empty(ev32, dtype=torch.float64, device=dev)
    dplr_bwd_dh_kernel[(NV, B * H)](
        qg, bg, w, gi, do, dv2i, dh, dv2,
        T, H, K, V, NT, BT, BTP, BK, BV, DOT, num_warps=4, num_stages=4,
    )

    dqg = torch.empty(ek32, dtype=torch.float64, device=dev)
    dkg = torch.empty(ek32, dtype=f32, device=dev)
    dbg = torch.empty(ek32, dtype=f32, device=dev)
    dw = torch.empty(ek32, dtype=torch.float64, device=dev)
    dglast = torch.empty((B, NT, H, K), dtype=torch.float64, device=dev)
    dplr_bwd_state_kernel[(NT, B * H)](
        h, dh, do, v, v_new, dv2, kg, gi,
        dqg, dkg, dbg, dw, dv0, dglast,
        T, H, K, V, BT, BTP, BK, BVf, DOT, num_warps=16, num_stages=2,
    )

    dAak = torch.empty(eA, dtype=torch.float64, device=dev)
    dAab = torch.empty(eA, dtype=torch.float64, device=dev)
    dag = torch.empty(ek32, dtype=torch.float64, device=dev)
    dplr_bwd_wy_kernel[(NT, B * H)](
        Ainv, Aak, v, ag, dw, dv2, dv0, dAak, dAab, dag,
        T, H, K, V, BT, BK, BVf, DOT, num_warps=4, num_stages=2,
    )

    dq = torch.empty(ek32, dtype=dt, device=dev)
    dk = torch.empty(ek32, dtype=dt, device=dev)
    da = torch.empty(ek32, dtype=dt, device=dev)
    db = torch.empty(ek32, dtype=dt, device=dev)
    dgkk = torch.empty(ek32, dtype=gdt, device=dev)
    dplr_bwd_intra_kernel[(NT, B * H)](
        q, k, alpha, beta, gk, dAqk, dAqb, dAak, dAab, dqg, dkg, dag, dbg, dglast,
        dq, dk, da, db, dgkk,
        T, H, K, scale, BT, BK, DOT, num_warps=4, num_stages=2,
    )
    return dq, dk, dv0.to(v.dtype), da, db, dgkk


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        o, cache, meta = _launch_fwd(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, *cache)
        ctx.meta = meta
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta, gk = ctx.saved_tensors[:6]
        cache = ctx.saved_tensors[6:]
        return _launch_bwd(do.contiguous(), q, k, v, alpha, beta, gk, cache, ctx.meta)


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
