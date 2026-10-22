"""Chunk-parallel gated delta rule, forward and backward, in Triton.

Per chunk of length BT, with a_i the inclusive cumulative log-decay inside the
chunk and S the state entering the chunk:

    A_ij  = beta_i exp(a_i - a_j) (k_i . k_j)      for j < i
    Tm    = (I + A)^{-1}
    W     = Tm (beta_i exp(a_i) k_i)
    Um    = Tm (beta_i v_i)
    U     = Um - W S
    S_out = exp(a_last) S + sum_j exp(a_last - a_j) k_j U_j^T
    O     = (exp(a_i) q_i) S + tril(exp(a_i - a_j) (q_i . k_j)) U

The gate is carried in log2 units so the decay is an exp2; gradients with
respect to it are accumulated in natural units by contracting each decayed
term with its own upstream gradient.
"""

import torch
import triton
import triton.language as tl

BT = 64
NLEV = tl.constexpr(6)
LOG2E = 1.4426950408889634


def _pow2(n):
    return 1 << (n - 1).bit_length()


@triton.jit
def diag_gate_cumsum(g, aa, T, HV: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p = (i_b * T) * HV + i_hv + o_t.to(tl.int64) * HV
    b_g = tl.load(g + p, mask=m_t, other=0.0).to(tl.float32)
    tl.store(aa + p, tl.cumsum(b_g, axis=0) * 1.4426950408889634, mask=m_t)


@triton.jit
def diag_wy_form(k, v, beta, aa, tm, w, um, T,
                 H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                 BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    NT = tl.cdiv(T, BT)

    rows = tl.arange(0, BT)
    o_t = i_t * BT + rows
    m_t = o_t < T
    o_k = tl.arange(0, BK)
    m_k = o_k < K

    p_bg = (i_b * T) * HV + i_hv + o_t.to(tl.int64) * HV
    b_beta = tl.load(beta + p_bg, mask=m_t, other=0.0).to(tl.float32)
    b_aa = tl.load(aa + p_bg, mask=m_t, other=0.0)

    p_k = ((i_b * T) * H + i_h) * K + o_t.to(tl.int64) * (H * K)
    b_k = tl.load(k + p_k[:, None] + o_k[None, :],
                  mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)

    b_kk = tl.dot(b_k, tl.trans(b_k), input_precision=PREC)
    lower = rows[:, None] > rows[None, :]
    b_dec = tl.exp2(tl.where(lower, b_aa[:, None] - b_aa[None, :], 0.0))
    b_A = tl.where(lower, b_beta[:, None] * b_dec * b_kk, 0.0)

    b_x = tl.where(rows[:, None] == rows[None, :], 1.0, 0.0)
    for lev in tl.static_range(0, NLEV):
        bs = 1 << lev
        blk = rows // bs
        sup = rows // (2 * bs)
        b_l = tl.where((sup[:, None] == sup[None, :]) & (blk[:, None] > blk[None, :]), b_A, 0.0)
        b_x = b_x - tl.dot(b_x, tl.dot(b_l, b_x, input_precision=PREC), input_precision=PREC)

    p_tm = tm + i_bh * (NT * BT * BT) + i_t.to(tl.int64) * (BT * BT)
    tl.store(p_tm + rows[:, None] * BT + rows[None, :], b_x)

    b_kb = b_k * (b_beta * tl.exp2(b_aa))[:, None]
    b_w = tl.dot(b_x, b_kb, input_precision=PREC)
    p_w = ((i_b * T) * HV + i_hv) * K + o_t.to(tl.int64) * (HV * K)
    tl.store(w + p_w[:, None] + o_k[None, :], b_w, mask=m_t[:, None] & m_k[None, :])

    p_v = ((i_b * T) * HV + i_hv) * V + o_t.to(tl.int64) * (HV * V)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_v = tl.load(v + p_v[:, None] + o_v[None, :],
                      mask=m_t[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_um = tl.dot(b_x, b_v * b_beta[:, None], input_precision=PREC)
        tl.store(um + p_v[:, None] + o_v[None, :], b_um, mask=m_t[:, None] & m_v[None, :])


@triton.jit
def diag_state_scan(k, w, um, aa, h, u, T,
                    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    NT = tl.cdiv(T, BT)

    rows = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    b_s = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        o_t = i_t * BT + rows
        m_t = o_t < T
        p_h = h + (i_bh * NT + i_t.to(tl.int64)) * (K * V) + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h, b_s, mask=m_kv)

        p_bg = (i_b * T) * HV + i_hv + o_t.to(tl.int64) * HV
        b_aa = tl.load(aa + p_bg, mask=m_t, other=0.0)
        last = tl.minimum(BT, T - i_t * BT) - 1
        atot = tl.sum(tl.where(rows == last, b_aa, 0.0), axis=0)

        p_w = ((i_b * T) * HV + i_hv) * K + o_t.to(tl.int64) * (HV * K)
        b_w = tl.load(w + p_w[:, None] + o_k[None, :],
                      mask=m_t[:, None] & m_k[None, :], other=0.0)
        p_u = ((i_b * T) * HV + i_hv) * V + o_t.to(tl.int64) * (HV * V)
        b_um = tl.load(um + p_u[:, None] + o_v[None, :],
                       mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_u = b_um - tl.dot(b_w, b_s, input_precision=PREC)
        tl.store(u + p_u[:, None] + o_v[None, :], b_u, mask=m_t[:, None] & m_v[None, :])

        p_k = ((i_b * T) * H + i_h) * K + o_t.to(tl.int64) * (H * K)
        b_k = tl.load(k + p_k[:, None] + o_k[None, :],
                      mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_kd = b_k * tl.exp2(atot - b_aa)[:, None]
        b_s = b_s * tl.exp2(atot) + tl.dot(tl.trans(b_kd), b_u, input_precision=PREC)


@triton.jit
def diag_chunked_matmul(q, k, aa, h, u, o, scale, T,
                        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                        BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr):
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    NT = tl.cdiv(T, BT)

    rows = tl.arange(0, BT)
    o_t = i_t * BT + rows
    m_t = o_t < T
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_tk = m_t[:, None] & m_k[None, :]
    m_tv = m_t[:, None] & m_v[None, :]

    p_bg = (i_b * T) * HV + i_hv + o_t.to(tl.int64) * HV
    b_aa = tl.load(aa + p_bg, mask=m_t, other=0.0)
    p_q = ((i_b * T) * H + i_h) * K + o_t.to(tl.int64) * (H * K)
    b_q = tl.load(q + p_q[:, None] + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + p_q[:, None] + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)

    p_h = h + (i_bh * NT + i_t.to(tl.int64)) * (K * V) + o_k[:, None] * V + o_v[None, :]
    b_s = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
    p_u = ((i_b * T) * HV + i_hv) * V + o_t.to(tl.int64) * (HV * V)
    b_u = tl.load(u + p_u[:, None] + o_v[None, :], mask=m_tv, other=0.0)

    b_o = tl.dot(b_q * tl.exp2(b_aa)[:, None], b_s, input_precision=PREC)
    causal = rows[:, None] >= rows[None, :]
    b_p = tl.where(causal, tl.exp2(tl.where(causal, b_aa[:, None] - b_aa[None, :], 0.0))
                   * tl.dot(b_q, tl.trans(b_k), input_precision=PREC), 0.0)
    b_o += tl.dot(b_p, b_u, input_precision=PREC)
    tl.store(o + p_u[:, None] + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_tv)


@triton.jit
def diag_dv_local(q, k, aa, do, dui, scale, T,
                  H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr):
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    rows = tl.arange(0, BT)
    o_t = i_t * BT + rows
    m_t = o_t < T
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_tk = m_t[:, None] & m_k[None, :]
    m_tv = m_t[:, None] & m_v[None, :]

    p_bg = (i_b * T) * HV + i_hv + o_t.to(tl.int64) * HV
    b_aa = tl.load(aa + p_bg, mask=m_t, other=0.0)
    p_q = ((i_b * T) * H + i_h) * K + o_t.to(tl.int64) * (H * K)
    b_q = tl.load(q + p_q[:, None] + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + p_q[:, None] + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
    causal = rows[:, None] >= rows[None, :]
    b_p = tl.where(causal, tl.exp2(tl.where(causal, b_aa[:, None] - b_aa[None, :], 0.0))
                   * tl.dot(b_q, tl.trans(b_k), input_precision=PREC), 0.0)

    p_v = ((i_b * T) * HV + i_hv) * V + o_t.to(tl.int64) * (HV * V)
    b_do = tl.load(do + p_v[:, None] + o_v[None, :], mask=m_tv, other=0.0).to(tl.float32)
    b_dui = tl.dot(tl.trans(b_p), b_do, input_precision=PREC)
    tl.store(dui + p_v[:, None] + o_v[None, :], b_dui, mask=m_tv)


@triton.jit
def diag_bwd_chunked(q, k, w, aa, do, dui, dhs, du, scale, T,
                     H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    NT = tl.cdiv(T, BT)

    rows = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    b_ds = tl.zeros([BK, BV], dtype=tl.float32)
    for j in range(NT):
        i_t = NT - 1 - j
        o_t = i_t * BT + rows
        m_t = o_t < T
        m_tk = m_t[:, None] & m_k[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        p_dh = dhs + (i_bh * NT + i_t.to(tl.int64)) * (K * V) + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh, b_ds, mask=m_kv)

        p_bg = (i_b * T) * HV + i_hv + o_t.to(tl.int64) * HV
        b_aa = tl.load(aa + p_bg, mask=m_t, other=0.0)
        last = tl.minimum(BT, T - i_t * BT) - 1
        atot = tl.sum(tl.where(rows == last, b_aa, 0.0), axis=0)

        p_q = ((i_b * T) * H + i_h) * K + o_t.to(tl.int64) * (H * K)
        b_k = tl.load(k + p_q[:, None] + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
        b_kd = b_k * tl.exp2(atot - b_aa)[:, None]

        p_v = ((i_b * T) * HV + i_hv) * V + o_t.to(tl.int64) * (HV * V)
        b_dui = tl.load(dui + p_v[:, None] + o_v[None, :], mask=m_tv, other=0.0)
        b_du = b_dui + tl.dot(b_kd, b_ds, input_precision=PREC)
        tl.store(du + p_v[:, None] + o_v[None, :], b_du, mask=m_tv)

        b_q = tl.load(q + p_q[:, None] + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
        b_qd = b_q * (scale * tl.exp2(b_aa))[:, None]
        b_do = tl.load(do + p_v[:, None] + o_v[None, :], mask=m_tv, other=0.0).to(tl.float32)
        p_w = ((i_b * T) * HV + i_hv) * K + o_t.to(tl.int64) * (HV * K)
        b_w = tl.load(w + p_w[:, None] + o_k[None, :], mask=m_tk, other=0.0)
        b_ds = (tl.dot(tl.trans(b_qd), b_do, input_precision=PREC)
                + b_ds * tl.exp2(atot)
                - tl.dot(tl.trans(b_w), b_du, input_precision=PREC))


@triton.jit
def diag_bwd_vloop(v, beta, tm, h, dhs, u, du, do,
                   dqd, dwo, dkd, dpo, dtmo, dvo, dbv, dal, T,
                   HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                   BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    NT = tl.cdiv(T, BT)

    rows = tl.arange(0, BT)
    o_t = i_t * BT + rows
    m_t = o_t < T
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]

    p_bg = (i_b * T) * HV + i_hv + o_t.to(tl.int64) * HV
    b_beta = tl.load(beta + p_bg, mask=m_t, other=0.0).to(tl.float32)

    p_tm = tm + i_bh * (NT * BT * BT) + i_t.to(tl.int64) * (BT * BT)
    b_tm = tl.load(p_tm + rows[:, None] * BT + rows[None, :])

    b_dqd = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_dkd = tl.zeros([BT, BK], dtype=tl.float32)
    b_dtm = tl.zeros([BT, BT], dtype=tl.float32)
    b_dp = tl.zeros([BT, BT], dtype=tl.float32)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    da_last = tl.zeros([1], dtype=tl.float32)

    p_v0 = ((i_b * T) * HV + i_hv) * V + o_t.to(tl.int64) * (HV * V)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        p_h = (i_bh * NT + i_t.to(tl.int64)) * (K * V) + o_k[:, None] * V + o_v[None, :]
        b_s = tl.load(h + p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_dh = tl.load(dhs + p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_u = tl.load(u + p_v0[:, None] + o_v[None, :], mask=m_tv, other=0.0)
        b_du = tl.load(du + p_v0[:, None] + o_v[None, :], mask=m_tv, other=0.0)
        b_do = tl.load(do + p_v0[:, None] + o_v[None, :], mask=m_tv, other=0.0).to(tl.float32)
        b_v = tl.load(v + p_v0[:, None] + o_v[None, :], mask=m_tv, other=0.0).to(tl.float32)

        b_dqd += tl.dot(b_do, tl.trans(b_s), input_precision=PREC)
        b_dw -= tl.dot(b_du, tl.trans(b_s), input_precision=PREC)
        b_dkd += tl.dot(b_u, tl.trans(b_dh), input_precision=PREC)
        b_dp += tl.dot(b_do, tl.trans(b_u), input_precision=PREC)
        b_dtm += tl.dot(b_du, tl.trans(b_v * b_beta[:, None]), input_precision=PREC)
        da_last += tl.sum(b_dh * b_s)

        b_dvb = tl.dot(tl.trans(b_tm), b_du, input_precision=PREC)
        b_dbeta += tl.sum(b_dvb * b_v, axis=1)
        tl.store(dvo + p_v0[:, None] + o_v[None, :],
                 (b_dvb * b_beta[:, None]).to(dvo.dtype.element_ty), mask=m_tv)

    p_o = ((i_b * T) * HV + i_hv) * K + o_t.to(tl.int64) * (HV * K)
    tl.store(dqd + p_o[:, None] + o_k[None, :], b_dqd, mask=m_tk)
    tl.store(dwo + p_o[:, None] + o_k[None, :], b_dw, mask=m_tk)
    tl.store(dkd + p_o[:, None] + o_k[None, :], b_dkd, mask=m_tk)
    p_m = (i_bh * NT + i_t.to(tl.int64)) * (BT * BT) + rows[:, None] * BT + rows[None, :]
    tl.store(dpo + p_m, b_dp)
    tl.store(dtmo + p_m, b_dtm)
    tl.store(dbv + p_bg, b_dbeta, mask=m_t)
    tl.store(dal + i_bh * NT + i_t.to(tl.int64), tl.sum(da_last, axis=0))


@triton.jit
def diag_bwd_inputs(q, k, beta, aa, tm, dqdi, dwi, dkdi, dpi, dtmi, dbv, dal,
                    dq, dk, dbeta, dan, scale, T,
                    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                    BT: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)
    NT = tl.cdiv(T, BT)
    NK: tl.constexpr = (K + BK - 1) // BK

    rows = tl.arange(0, BT)
    o_t = i_t * BT + rows
    m_t = o_t < T
    lower = rows[:, None] > rows[None, :]
    causal = rows[:, None] >= rows[None, :]

    p_bg = (i_b * T) * HV + i_hv + o_t.to(tl.int64) * HV
    b_beta = tl.load(beta + p_bg, mask=m_t, other=0.0).to(tl.float32)
    b_aa = tl.load(aa + p_bg, mask=m_t, other=0.0)
    last = tl.minimum(BT, T - i_t * BT) - 1
    atot = tl.sum(tl.where(rows == last, b_aa, 0.0), axis=0)
    b_ea = tl.exp2(b_aa)
    b_edk = tl.exp2(atot - b_aa)
    b_dec = tl.exp2(tl.where(causal, b_aa[:, None] - b_aa[None, :], 0.0))
    b_bea = b_beta * b_ea

    p_tm = tm + i_bh * (NT * BT * BT) + i_t.to(tl.int64) * (BT * BT)
    b_tm = tl.load(p_tm + rows[:, None] * BT + rows[None, :])
    p_m = (i_bh * NT + i_t.to(tl.int64)) * (BT * BT) + rows[:, None] * BT + rows[None, :]
    b_dp = tl.where(causal, tl.load(dpi + p_m), 0.0)
    b_dtm = tl.load(dtmi + p_m)

    b_qbase = ((i_b * T) * H + i_h) * K + o_t.to(tl.int64) * (H * K)
    b_ibase = ((i_b * T) * HV + i_hv) * K + o_t.to(tl.int64) * (HV * K)

    b_kk = tl.zeros([BT, BT], dtype=tl.float32)
    b_qk = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(NK):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k < K)[None, :]
        b_k = tl.load(k + b_qbase[:, None] + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
        b_q = tl.load(q + b_qbase[:, None] + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32) * scale
        b_dw = tl.load(dwi + b_ibase[:, None] + o_k[None, :], mask=m_tk, other=0.0)
        b_kk += tl.dot(b_k, tl.trans(b_k), input_precision=PREC)
        b_qk += tl.dot(b_q, tl.trans(b_k), input_precision=PREC)
        b_dtm += tl.dot(b_dw, tl.trans(b_k * b_bea[:, None]), input_precision=PREC)

    b_dam = -tl.dot(tl.dot(tl.trans(b_tm), b_dtm, input_precision=PREC),
                    tl.trans(b_tm), input_precision=PREC)
    b_dam = tl.where(lower, b_dam, 0.0)
    b_amat = tl.where(lower, b_beta[:, None] * b_dec * b_kk, 0.0)
    b_bm = b_dam * b_beta[:, None] * b_dec
    b_dbeta = tl.load(dbv + p_bg, mask=m_t, other=0.0) + tl.sum(b_dam * b_dec * b_kk * lower, axis=1)
    b_ga = b_dam * b_amat
    b_da = tl.sum(b_ga, axis=1) - tl.sum(b_ga, axis=0)

    b_gp = b_dp * tl.where(causal, b_dec * b_qk, 0.0)
    b_da += tl.sum(b_gp, axis=1) - tl.sum(b_gp, axis=0)
    b_cp = b_dp * b_dec
    b_bmt = tl.trans(b_bm)
    b_cpt = tl.trans(b_cp)
    b_tmt = tl.trans(b_tm)

    b_dl = tl.load(dal + i_bh * NT + i_t.to(tl.int64)) * tl.exp2(atot)
    for i_k in range(NK):
        o_k = i_k * BK + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k < K)[None, :]
        b_k = tl.load(k + b_qbase[:, None] + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32)
        b_q = tl.load(q + b_qbase[:, None] + o_k[None, :], mask=m_tk, other=0.0).to(tl.float32) * scale
        b_dqd = tl.load(dqdi + b_ibase[:, None] + o_k[None, :], mask=m_tk, other=0.0)
        b_dw = tl.load(dwi + b_ibase[:, None] + o_k[None, :], mask=m_tk, other=0.0)
        b_dkd = tl.load(dkdi + b_ibase[:, None] + o_k[None, :], mask=m_tk, other=0.0)
        b_dkb = tl.dot(b_tmt, b_dw, input_precision=PREC)

        b_dq = (tl.dot(b_cp, b_k, input_precision=PREC) + b_dqd * b_ea[:, None]) * scale
        b_dk = (tl.dot(b_bm, b_k, input_precision=PREC)
                + tl.dot(b_bmt, b_k, input_precision=PREC)
                + tl.dot(b_cpt, b_q, input_precision=PREC)
                + b_dkb * b_bea[:, None]
                + b_dkd * b_edk[:, None])
        tl.store(dq + b_ibase[:, None] + o_k[None, :], b_dq, mask=m_tk)
        tl.store(dk + b_ibase[:, None] + o_k[None, :], b_dk, mask=m_tk)

        b_ekd = tl.sum(b_dkd * (b_k * b_edk[:, None]), axis=1)
        b_dbeta += tl.sum(b_dkb * b_k, axis=1) * b_ea
        b_da += (tl.sum(b_dqd * (b_q * b_ea[:, None]), axis=1)
                 + tl.sum(b_dkb * (b_k * b_bea[:, None]), axis=1)
                 - b_ekd)
        b_dl += tl.sum(b_ekd, axis=0)

    b_da += tl.where(rows == last, b_dl, 0.0)
    tl.store(dbeta + p_bg, b_dbeta, mask=m_t)
    tl.store(dan + p_bg, b_da, mask=m_t)


@triton.jit
def diag_dg_revcumsum(dan, dg, T, HV: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    rows = tl.arange(0, BT)
    o_t = i_t * BT + rows
    m_t = o_t < T
    p = (i_b * T) * HV + i_hv + o_t.to(tl.int64) * HV
    b_da = tl.load(dan + p, mask=m_t, other=0.0)
    b_rev = tl.sum(tl.where(rows[None, :] >= rows[:, None], b_da[None, :], 0.0), axis=1)
    tl.store(dg + p, b_rev.to(dg.dtype.element_ty), mask=m_t)


@triton.jit
def diag_group_reduce(src, dst, T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                      REP: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]
    acc = tl.zeros([BT, BK], dtype=tl.float32)
    for r in tl.static_range(0, REP):
        p = ((i_b * T) * HV + i_h * REP + r) * K + o_t.to(tl.int64) * (HV * K)
        acc += tl.load(src + p[:, None] + o_k[None, :], mask=m, other=0.0)
    p = ((i_b * T) * H + i_h) * K + o_t.to(tl.int64) * (H * K)
    tl.store(dst + p[:, None] + o_k[None, :], acc.to(dst.dtype.element_ty), mask=m)


@triton.jit
def diag_narrow(src, dst, T, HV: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p = (i_b * T) * HV + i_hv + o_t.to(tl.int64) * HV
    tl.store(dst + p, tl.load(src + p, mask=m_t, other=0.0).to(dst.dtype.element_ty), mask=m_t)


def _tiles(K, V, dtype):
    BK = _pow2(K)
    if BK <= 64:
        BV = min(64, _pow2(V))
    elif BK <= 128:
        BV = min(64, _pow2(V))
    else:
        BV = min(16, _pow2(V))
    prec = "ieee" if dtype == torch.float32 else "tf32"
    # Deep pipelining multiplies the resident tile footprint; the wide-BK
    # scans only fit in shared memory single-staged.
    ns = 2 if BK <= 128 else 1
    return BK, BV, prec, ns


def _fwd(q, k, v, g, beta):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    NT = triton.cdiv(T, BT)
    dev, f32 = q.device, torch.float32
    BK, BV, prec, ns = _tiles(K, V, q.dtype)
    scale = K ** -0.5

    aa = torch.empty(B, T, HV, device=dev, dtype=f32)
    diag_gate_cumsum[(NT, B * HV)](g, aa, T, HV=HV, BT=BT, num_warps=1)

    tm = torch.empty(B * HV, NT, BT, BT, device=dev, dtype=f32)
    w = torch.empty(B, T, HV, K, device=dev, dtype=f32)
    um = torch.empty(B, T, HV, V, device=dev, dtype=f32)
    diag_wy_form[(NT, B * HV)](k, v, beta, aa, tm, w, um, T,
                               H=H, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
                               num_warps=4, num_stages=1)

    h = torch.empty(B * HV, NT, K, V, device=dev, dtype=f32)
    u = torch.empty(B, T, HV, V, device=dev, dtype=f32)
    diag_state_scan[(triton.cdiv(V, BV), B * HV)](
        k, w, um, aa, h, u, T, H=H, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
        num_warps=4, num_stages=ns)

    odt = v.dtype
    o = torch.empty(B, T, HV, V, device=dev, dtype=odt)
    diag_chunked_matmul[(NT, triton.cdiv(V, BV), B * HV)](
        q, k, aa, h, u, o, scale, T, H=H, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
        num_warps=4, num_stages=ns)
    return o, aa, tm, w, h, u


def _bwd(do, q, k, v, g, beta, aa, tm, w, h, u):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    NT = triton.cdiv(T, BT)
    dev, f32 = q.device, torch.float32
    BK, BV, prec, ns = _tiles(K, V, q.dtype)
    scale = K ** -0.5
    do = do.contiguous()

    dui = torch.empty(B, T, HV, V, device=dev, dtype=f32)
    diag_dv_local[(NT, triton.cdiv(V, BV), B * HV)](
        q, k, aa, do, dui, scale, T, H=H, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
        num_warps=4, num_stages=ns)

    dhs = torch.empty(B * HV, NT, K, V, device=dev, dtype=f32)
    du = torch.empty(B, T, HV, V, device=dev, dtype=f32)
    diag_bwd_chunked[(triton.cdiv(V, BV), B * HV)](
        q, k, w, aa, do, dui, dhs, du, scale, T, H=H, HV=HV, K=K, V=V,
        BT=BT, BK=BK, BV=BV, PREC=prec, num_warps=8, num_stages=1)

    dqd = torch.empty(B, T, HV, K, device=dev, dtype=f32)
    dwo = torch.empty(B, T, HV, K, device=dev, dtype=f32)
    dkd = torch.empty(B, T, HV, K, device=dev, dtype=f32)
    dpo = torch.empty(B * HV, NT, BT, BT, device=dev, dtype=f32)
    dtmo = torch.empty(B * HV, NT, BT, BT, device=dev, dtype=f32)
    vdt = v.dtype
    dv = torch.empty(B, T, HV, V, device=dev, dtype=vdt)
    dbv = torch.empty(B, T, HV, device=dev, dtype=f32)
    dal = torch.empty(B * HV, NT, device=dev, dtype=f32)
    diag_bwd_vloop[(NT, B * HV)](
        v, beta, tm, h, dhs, u, du, do,
        dqd, dwo, dkd, dpo, dtmo, dv, dbv, dal, T,
        HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec, num_warps=8, num_stages=ns)

    dqf = torch.empty(B, T, HV, K, device=dev, dtype=f32)
    dkf = torch.empty(B, T, HV, K, device=dev, dtype=f32)
    dbetaf = torch.empty(B, T, HV, device=dev, dtype=f32)
    dan = torch.empty(B, T, HV, device=dev, dtype=f32)
    diag_bwd_inputs[(NT, B * HV)](
        q, k, beta, aa, tm, dqd, dwo, dkd, dpo, dtmo, dbv, dal,
        dqf, dkf, dbetaf, dan, scale, T, H=H, HV=HV, K=K,
        BT=BT, BK=min(BK, 64), PREC=prec, num_warps=4, num_stages=1)

    gdt = g.dtype
    dg = torch.empty(B, T, HV, device=dev, dtype=gdt)
    diag_dg_revcumsum[(NT, B * HV)](dan, dg, T, HV=HV, BT=BT, num_warps=4)

    rep = HV // H
    qdt = q.dtype
    dq = torch.empty(B, T, H, K, device=dev, dtype=qdt)
    kdt = k.dtype
    dk = torch.empty(B, T, H, K, device=dev, dtype=kdt)
    diag_group_reduce[(NT, B * H)](dqf, dq, T, H=H, HV=HV, K=K, REP=rep,
                                   BT=BT, BK=BK, num_warps=4)
    diag_group_reduce[(NT, B * H)](dkf, dk, T, H=H, HV=HV, K=K, REP=rep,
                                   BT=BT, BK=BK, num_warps=4)
    bdt = beta.dtype
    dbeta = torch.empty(B, T, HV, device=dev, dtype=bdt)
    diag_narrow[(NT, B * HV)](dbetaf, dbeta, T, HV=HV, BT=BT, num_warps=1)
    return dq, dk, dv, dg, dbeta


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        g, beta = g.contiguous(), beta.contiguous()
        o, aa, tm, w, h, u = _fwd(q, k, v, g, beta)
        ctx.save_for_backward(q, k, v, g, beta, aa, tm, w, h, u)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, aa, tm, w, h, u = ctx.saved_tensors
        return _bwd(do, q, k, v, g, beta, aa, tm, w, h, u)


def gated_delta_rule(
    q: torch.Tensor,      # [B, T, H, K]
    k: torch.Tensor,      # [B, T, H, K]
    v: torch.Tensor,      # [B, T, HV, V]
    g: torch.Tensor,      # [B, T, HV]
    beta: torch.Tensor,   # [B, T, HV]
) -> torch.Tensor:        # [B, T, HV, V]
    return _GatedDeltaRule.apply(q, k, v, g, beta)
