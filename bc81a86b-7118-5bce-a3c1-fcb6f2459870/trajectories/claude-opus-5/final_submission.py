"""Chunk-parallel gated delta rule: Triton forward and backward.

Per chunk of BT timesteps the log-gate is accumulated once, the intra-chunk
delta corrections are collapsed into a unit lower triangular system that is
inverted exactly by repeated squaring of its nilpotent part, and the recurrent
state is carried across chunk boundaries in float32.

Chunk algebra, with a_t the inclusive cumulative log-gate inside the chunk,
A the chunk total, S the state entering the chunk and w the chunk write:

    z_t = beta_t v_t - beta_t exp(a_t) k_t^T S
    (I + T) w = z,   T_tj = beta_t exp(a_t - a_j) (k_t . k_j)  for j < t
    o_t       = exp(a_t) q_t^T S + sum_{j<=t} exp(a_t - a_j) (q_t . k_j) w_j
    S_next    = exp(A) S + sum_t exp(A - a_t) k_t w_t^T
"""

import torch
import triton
import triton.language as tl

BT = 64
LOG2E = tl.constexpr(1.4426950408889634)
EXP_CLAMP = tl.constexpr(64.0)


@triton.jit
def _cumsum_kernel(g, a, eAs, T, s_gb, s_gt, s_gh,
                   HV: tl.constexpr, NT: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HV
    i_h = i_bh % HV
    o_t = tl.arange(0, BT)
    idx = i_t * BT + o_t
    m = idx < T
    p = i_b.to(tl.int64) * s_gb + i_h.to(tl.int64) * s_gh + idx.to(tl.int64) * s_gt
    b_g = tl.load(g + p, mask=m, other=0.0).to(tl.float32)
    b_a = tl.cumsum(b_g, axis=0)
    tl.store(a + i_bh.to(tl.int64) * T + idx.to(tl.int64), b_a, mask=m)
    b_A = tl.sum(tl.where(m, b_a, 0.0) * tl.where(idx == tl.minimum((i_t + 1) * BT, T) - 1,
                                                  1.0, 0.0), axis=0)
    tl.store(eAs + i_bh.to(tl.int64) * NT + i_t, tl.exp2(tl.minimum(b_A * LOG2E, EXP_CLAMP)))


@triton.jit
def _wy_kernel(k, v, beta, a, M, w, w2, u0, kbo, kdo, T, K, V,
               s_kb, s_kt, s_kh, s_vb, s_vt, s_vh, s_bb, s_bt, s_bh,
               HV: tl.constexpr, REP: tl.constexpr, NT: tl.constexpr, NV: tl.constexpr,
               BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HV
    i_h = i_bh % HV
    i_hk = i_h // REP
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    idx = i_t * BT + o_t
    m_t = idx < T
    m_k = o_k < K

    p_k = k + i_b.to(tl.int64) * s_kb + i_hk.to(tl.int64) * s_kh \
        + idx.to(tl.int64)[:, None] * s_kt + o_k[None, :]
    b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)
    b_beta = tl.load(beta + i_b.to(tl.int64) * s_bb + i_h.to(tl.int64) * s_bh
                     + idx.to(tl.int64) * s_bt, mask=m_t, other=0.0).to(tl.float32)
    b_a = tl.load(a + i_bh.to(tl.int64) * T + idx.to(tl.int64), mask=m_t, other=0.0)

    b_kk = tl.dot(b_k, tl.trans(b_k), input_precision=PREC).to(tl.float32)
    gm = tl.exp2(tl.minimum((b_a[:, None] - b_a[None, :]) * LOG2E, EXP_CLAMP))
    low = o_t[:, None] > o_t[None, :]
    b_N = tl.where(low, -(b_beta[:, None] * gm * b_kk), 0.0)

    b_R = tl.where(o_t[:, None] == o_t[None, :], 1.0, 0.0) + b_N
    b_Y = b_N
    for _ in tl.static_range(5):
        b_Y = tl.dot(b_Y, b_Y, input_precision="tf32")
        b_R = b_R + tl.dot(b_R, b_Y, input_precision="tf32")

    tl.store(M + (i_bh.to(tl.int64) * NT + i_t) * (BT * BT) + o_t[:, None] * BT + o_t[None, :],
             b_R)

    # w = M (beta exp(a) k) and u0 = M (beta v) fold the triangular solve out of
    # the serial state scan, which then only multiplies against the state.
    last = tl.minimum((i_t + 1) * BT, T) - 1
    b_A = tl.load(a + i_bh.to(tl.int64) * T + last.to(tl.int64))
    ea = tl.exp2(tl.minimum(b_a * LOG2E, EXP_CLAMP))
    b_kb = (b_k.to(tl.float32) * (b_beta * ea)[:, None]).to(b_k.dtype)
    b_Rc = b_R.to(b_k.dtype)
    b_w = tl.dot(b_Rc, b_kb, input_precision=PREC)
    p_w = w + (i_bh.to(tl.int64) * T + idx.to(tl.int64))[:, None] * K + o_k[None, :]
    tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_t[:, None] & m_k[None, :])

    # w2 = M^T (exp(A - a) k) is the state-facing half of the reverse scan.
    b_kd = (b_k.to(tl.float32)
            * tl.exp2(tl.minimum((b_A - b_a) * LOG2E, EXP_CLAMP))[:, None]).to(b_k.dtype)
    b_w2 = tl.dot(tl.trans(b_Rc), b_kd, input_precision=PREC)
    p_w2 = w2 + (i_bh.to(tl.int64) * T + idx.to(tl.int64))[:, None] * K + o_k[None, :]
    tl.store(p_w2, b_w2.to(p_w2.dtype.element_ty), mask=m_t[:, None] & m_k[None, :])

    p_kbo = kbo + (i_bh.to(tl.int64) * T + idx.to(tl.int64))[:, None] * K + o_k[None, :]
    tl.store(p_kbo, b_kb, mask=m_t[:, None] & m_k[None, :])
    p_kdo = kdo + (i_bh.to(tl.int64) * T + idx.to(tl.int64))[:, None] * K + o_k[None, :]
    tl.store(p_kdo, b_kd, mask=m_t[:, None] & m_k[None, :])

    p_vb = v + i_b.to(tl.int64) * s_vb + i_h.to(tl.int64) * s_vh + idx.to(tl.int64)[:, None] * s_vt
    for i_v in range(NV):
        o_v = i_v * BV + tl.arange(0, BV)
        mv = m_t[:, None] & (o_v < V)[None, :]
        b_v = tl.load(p_vb + o_v[None, :], mask=mv, other=0.0).to(tl.float32)
        b_u0 = tl.dot(b_Rc, (b_beta[:, None] * b_v).to(b_k.dtype), input_precision=PREC)
        tl.store(u0 + (i_bh.to(tl.int64) * T + idx.to(tl.int64))[:, None] * V + o_v[None, :],
                 b_u0, mask=mv)


@triton.jit
def _fwd_scan_kernel(w, kdo, eAs, u0, u, h, T, K, V,
                     HV: tl.constexpr, NT: tl.constexpr,
                     BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                     PREC: tl.constexpr, CT: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V

    base_u = u + i_bh.to(tl.int64) * T * V
    base_u0 = u0 + i_bh.to(tl.int64) * T * V
    base_w = w + i_bh.to(tl.int64) * T * K
    base_kd = kdo + i_bh.to(tl.int64) * T * K
    base_h = h + i_bh.to(tl.int64) * NT * K * V

    b_S = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        tl.store(base_h + i_t.to(tl.int64) * K * V + o_k[:, None].to(tl.int64) * V + o_v[None, :],
                 b_S, mask=m_k[:, None] & m_v[None, :])

        idx = i_t * BT + o_t
        m_t = idx < T
        mk = m_t[:, None] & m_k[None, :]
        mv = m_t[:, None] & m_v[None, :]
        b_w = tl.load(base_w + idx.to(tl.int64)[:, None] * K + o_k[None, :], mask=mk, other=0.0)
        b_kd = tl.load(base_kd + idx.to(tl.int64)[:, None] * K + o_k[None, :], mask=mk, other=0.0)
        b_u0 = tl.load(base_u0 + idx.to(tl.int64)[:, None] * V + o_v[None, :], mask=mv, other=0.0)
        b_eA = tl.load(eAs + i_bh.to(tl.int64) * NT + i_t)

        b_u = b_u0 - tl.dot(b_w, b_S.to(CT), input_precision=PREC)
        tl.store(base_u + idx.to(tl.int64)[:, None] * V + o_v[None, :], b_u, mask=mv)

        b_S = b_S * b_eA + tl.dot(tl.trans(b_kd), b_u.to(CT), input_precision=PREC)


@triton.jit
def _out_kernel(q, k, a, u, h, o, T, K, V, scale,
                s_kb, s_kt, s_kh, s_vb, s_vt, s_vh,
                HV: tl.constexpr, REP: tl.constexpr, NT: tl.constexpr,
                BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                PREC: tl.constexpr):
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // HV
    i_h = i_bh % HV
    i_hk = i_h // REP
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    idx = i_t * BT + o_t
    m_t = idx < T
    m_k = o_k < K
    m_v = o_v < V
    mm = m_t[:, None] & m_k[None, :]

    off = i_b.to(tl.int64) * s_kb + i_hk.to(tl.int64) * s_kh \
        + idx.to(tl.int64)[:, None] * s_kt + o_k[None, :]
    b_q = tl.load(q + off, mask=mm, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + off, mask=mm, other=0.0)
    b_a = tl.load(a + i_bh.to(tl.int64) * T + idx.to(tl.int64), mask=m_t, other=0.0)
    b_u = tl.load(u + i_bh.to(tl.int64) * T * V + idx.to(tl.int64)[:, None] * V + o_v[None, :],
                  mask=m_t[:, None] & m_v[None, :], other=0.0)
    b_S = tl.load(h + i_bh.to(tl.int64) * NT * K * V + i_t.to(tl.int64) * K * V
                  + o_k[:, None].to(tl.int64) * V + o_v[None, :],
                  mask=m_k[:, None] & m_v[None, :], other=0.0)

    b_qk = tl.dot(b_q.to(b_k.dtype), tl.trans(b_k), input_precision=PREC).to(tl.float32)
    gm = tl.exp2(tl.minimum((b_a[:, None] - b_a[None, :]) * LOG2E, EXP_CLAMP))
    b_att = tl.where(o_t[:, None] >= o_t[None, :], b_qk * gm, 0.0)
    b_qd = (b_q * tl.exp2(tl.minimum(b_a * LOG2E, EXP_CLAMP))[:, None]).to(b_k.dtype)
    b_o = tl.dot(b_qd, b_S.to(b_k.dtype), input_precision=PREC)
    b_o += tl.dot(b_att.to(b_k.dtype), b_u.to(b_k.dtype), input_precision=PREC)

    p_o = o + i_b.to(tl.int64) * s_vb + i_h.to(tl.int64) * s_vh \
        + idx.to(tl.int64)[:, None] * s_vt + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


@triton.jit
def _bwd_prep_kernel(q, k, a, M, do, dz0, dsc, T, K, V, scale,
                     s_kb, s_kt, s_kh, s_vb, s_vt, s_vh,
                     HV: tl.constexpr, REP: tl.constexpr, NT: tl.constexpr,
                     BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                     PREC: tl.constexpr):
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // HV
    i_h = i_bh % HV
    i_hk = i_h // REP
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    idx = i_t * BT + o_t
    m_t = idx < T
    m_k = o_k < K
    m_v = o_v < V
    mm = m_t[:, None] & m_k[None, :]
    mv = m_t[:, None] & m_v[None, :]

    off = i_b.to(tl.int64) * s_kb + i_hk.to(tl.int64) * s_kh \
        + idx.to(tl.int64)[:, None] * s_kt + o_k[None, :]
    b_q = tl.load(q + off, mask=mm, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + off, mask=mm, other=0.0)
    b_do = tl.load(do + i_b.to(tl.int64) * s_vb + i_h.to(tl.int64) * s_vh
                   + idx.to(tl.int64)[:, None] * s_vt + o_v[None, :], mask=mv, other=0.0)
    b_a = tl.load(a + i_bh.to(tl.int64) * T + idx.to(tl.int64), mask=m_t, other=0.0)
    b_M = tl.load(M + (i_bh.to(tl.int64) * NT + i_t) * (BT * BT)
                  + o_t[:, None] * BT + o_t[None, :])

    b_qk = tl.dot(b_q.to(b_k.dtype), tl.trans(b_k), input_precision=PREC).to(tl.float32)
    gm = tl.exp2(tl.minimum((b_a[:, None] - b_a[None, :]) * LOG2E, EXP_CLAMP))
    b_att = tl.where(o_t[:, None] >= o_t[None, :], b_qk * gm, 0.0)
    b_aM = tl.dot(b_att.to(b_k.dtype), b_M.to(b_k.dtype), input_precision=PREC)
    b_dz0 = tl.dot(tl.trans(b_aM).to(b_k.dtype), b_do, input_precision=PREC)
    tl.store(dz0 + i_bh.to(tl.int64) * T * V + idx.to(tl.int64)[:, None] * V + o_v[None, :],
             b_dz0, mask=mv)

    b_qd = (b_q * tl.exp2(tl.minimum(b_a * LOG2E, EXP_CLAMP))[:, None]).to(b_k.dtype)
    b_dsc = tl.dot(tl.trans(b_qd), b_do, input_precision=PREC)
    tl.store(dsc + (i_bh.to(tl.int64) * NT + i_t) * K * V + o_k[:, None].to(tl.int64) * V
             + o_v[None, :], b_dsc, mask=m_k[:, None] & m_v[None, :])


@triton.jit
def _bwd_scan_kernel(w2, kbo, eAs, dz0, dsc, dz, dh, T, K, V,
                     HV: tl.constexpr, NT: tl.constexpr,
                     BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                     PREC: tl.constexpr, CT: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V

    base_dz = dz + i_bh.to(tl.int64) * T * V
    base_dz0 = dz0 + i_bh.to(tl.int64) * T * V
    base_w2 = w2 + i_bh.to(tl.int64) * T * K
    base_kb = kbo + i_bh.to(tl.int64) * T * K
    base_dh = dh + i_bh.to(tl.int64) * NT * K * V
    base_dsc = dsc + i_bh.to(tl.int64) * NT * K * V

    b_dS = tl.zeros([BK, BV], dtype=tl.float32)
    for step in range(NT):
        i_t = NT - 1 - step
        tl.store(base_dh + i_t.to(tl.int64) * K * V + o_k[:, None].to(tl.int64) * V + o_v[None, :],
                 b_dS, mask=m_k[:, None] & m_v[None, :])

        idx = i_t * BT + o_t
        m_t = idx < T
        mk = m_t[:, None] & m_k[None, :]
        mv = m_t[:, None] & m_v[None, :]
        b_w2 = tl.load(base_w2 + idx.to(tl.int64)[:, None] * K + o_k[None, :], mask=mk, other=0.0)
        b_kb = tl.load(base_kb + idx.to(tl.int64)[:, None] * K + o_k[None, :], mask=mk, other=0.0)
        b_dz0 = tl.load(base_dz0 + idx.to(tl.int64)[:, None] * V + o_v[None, :], mask=mv, other=0.0)
        b_dsc = tl.load(base_dsc + i_t.to(tl.int64) * K * V + o_k[:, None].to(tl.int64) * V
                        + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_eA = tl.load(eAs + i_bh.to(tl.int64) * NT + i_t)

        b_dz = b_dz0 + tl.dot(b_w2, b_dS.to(CT), input_precision=PREC)
        tl.store(base_dz + idx.to(tl.int64)[:, None] * V + o_v[None, :], b_dz, mask=mv)

        b_dS = b_dsc + b_dS * b_eA - tl.dot(tl.trans(b_kb), b_dz.to(CT), input_precision=PREC)


@triton.jit
def _bwd_grads_kernel(q, k, v, beta, a, u, dz, h, dh, do,
                      dq, dk, dv, dg, dbeta, T, K, V, scale,
                      s_kb, s_kt, s_kh, s_vb, s_vt, s_vh, s_bb, s_bt, s_bh,
                      s_gb, s_gt, s_gh, s_ob, s_ot, s_oh,
                      HV: tl.constexpr, REP: tl.constexpr, NT: tl.constexpr, NV: tl.constexpr,
                      BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                      PREC: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HV
    i_h = i_bh % HV
    i_hk = i_h // REP
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    idx = i_t * BT + o_t
    m_t = idx < T
    m_k = o_k < K
    mm = m_t[:, None] & m_k[None, :]

    off_qk = i_b.to(tl.int64) * s_kb + i_hk.to(tl.int64) * s_kh \
        + idx.to(tl.int64)[:, None] * s_kt + o_k[None, :]
    b_q = tl.load(q + off_qk, mask=mm, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + off_qk, mask=mm, other=0.0)
    b_beta = tl.load(beta + i_b.to(tl.int64) * s_bb + i_h.to(tl.int64) * s_bh
                     + idx.to(tl.int64) * s_bt, mask=m_t, other=0.0).to(tl.float32)
    b_a = tl.load(a + i_bh.to(tl.int64) * T + idx.to(tl.int64), mask=m_t, other=0.0)
    last = tl.minimum((i_t + 1) * BT, T) - 1
    b_A = tl.load(a + i_bh.to(tl.int64) * T + last.to(tl.int64))
    ea = tl.exp2(tl.minimum(b_a * LOG2E, EXP_CLAMP))
    eA = tl.exp2(tl.minimum(b_A * LOG2E, EXP_CLAMP))
    eda = tl.exp2(tl.minimum((b_A - b_a) * LOG2E, EXP_CLAMP))

    acc_N = tl.zeros([BT, BT], dtype=tl.float32)
    acc_D = tl.zeros([BT, BT], dtype=tl.float32)
    acc_E = tl.zeros([BT, BK], dtype=tl.float32)
    acc_KD = tl.zeros([BT, BK], dtype=tl.float32)
    acc_G = tl.zeros([BT, BK], dtype=tl.float32)
    acc_sS = tl.zeros([BK], dtype=tl.float32)
    acc_bv = tl.zeros([BT], dtype=tl.float32)

    base_u = u + i_bh.to(tl.int64) * T * V
    base_dz = dz + i_bh.to(tl.int64) * T * V
    base_h = h + i_bh.to(tl.int64) * NT * K * V + i_t.to(tl.int64) * K * V
    base_dh = dh + i_bh.to(tl.int64) * NT * K * V + i_t.to(tl.int64) * K * V
    off_v = i_b.to(tl.int64) * s_vb + i_h.to(tl.int64) * s_vh + idx.to(tl.int64)[:, None] * s_vt

    for i_v in range(NV):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        mv = m_t[:, None] & m_v[None, :]
        mkv = m_k[:, None] & m_v[None, :]
        b_u = tl.load(base_u + idx.to(tl.int64)[:, None] * V + o_v[None, :], mask=mv, other=0.0)
        b_dz = tl.load(base_dz + idx.to(tl.int64)[:, None] * V + o_v[None, :], mask=mv, other=0.0)
        b_do = tl.load(do + off_v + o_v[None, :], mask=mv, other=0.0)
        b_v = tl.load(v + off_v + o_v[None, :], mask=mv, other=0.0).to(tl.float32)
        b_S = tl.load(base_h + o_k[:, None].to(tl.int64) * V + o_v[None, :], mask=mkv, other=0.0)
        b_dS = tl.load(base_dh + o_k[:, None].to(tl.int64) * V + o_v[None, :], mask=mkv, other=0.0)

        ct = b_k.dtype
        b_uc = b_u.to(ct)
        b_dzc = b_dz.to(ct)
        b_Sc = tl.trans(b_S).to(ct)
        acc_N += tl.dot(b_dzc, tl.trans(b_uc), input_precision=PREC)
        acc_D += tl.dot(b_do, tl.trans(b_uc), input_precision=PREC)
        acc_E += tl.dot(b_do, b_Sc, input_precision=PREC)
        acc_KD += tl.dot(b_uc, tl.trans(b_dS).to(ct), input_precision=PREC)
        acc_G -= tl.dot(b_dzc, b_Sc, input_precision=PREC)
        acc_sS += tl.sum(b_S * b_dS, axis=1)
        acc_bv += tl.sum(b_v * b_dz, axis=1)

        p_dv = dv + off_v + o_v[None, :]
        tl.store(p_dv, (b_beta[:, None] * b_dz).to(p_dv.dtype.element_ty), mask=mv)

    gm = tl.exp2(tl.minimum((b_a[:, None] - b_a[None, :]) * LOG2E, EXP_CLAMP))
    b_kk = tl.dot(b_k, tl.trans(b_k), input_precision=PREC).to(tl.float32)
    b_qk = tl.dot(b_q.to(b_k.dtype), tl.trans(b_k), input_precision=PREC).to(tl.float32)

    b_P = tl.where(o_t[:, None] > o_t[None, :], -acc_N, 0.0)
    b_B = b_P * gm * b_kk
    b_Pg = b_P * b_beta[:, None] * gm
    b_Q = b_beta[:, None] * b_B

    b_dAtt = tl.where(o_t[:, None] >= o_t[None, :], acc_D, 0.0)
    b_Ag = b_dAtt * gm
    b_R = b_Ag * b_qk

    da = tl.sum(b_Q, axis=1) - tl.sum(b_Q, axis=0) + tl.sum(b_R, axis=1) - tl.sum(b_R, axis=0)
    da += tl.sum(b_q * ea[:, None] * acc_E, axis=1)
    kdg = tl.sum(b_k.to(tl.float32) * eda[:, None] * acc_KD, axis=1)
    b_kb = b_k.to(tl.float32) * (b_beta * ea)[:, None]
    da += tl.sum(b_kb * acc_G, axis=1) - kdg
    dA = tl.sum(tl.where(m_t, kdg, 0.0), axis=0) + eA * tl.sum(tl.where(m_k, acc_sS, 0.0), axis=0)
    da += tl.where(idx == last, dA, 0.0)
    da = tl.where(m_t, da, 0.0)
    b_dg = tl.sum(tl.where(o_t[:, None] <= o_t[None, :], da[None, :], 0.0), axis=1)
    tl.store(dg + i_b.to(tl.int64) * s_gb + i_h.to(tl.int64) * s_gh + idx.to(tl.int64) * s_gt,
             b_dg, mask=m_t)

    b_dbeta = tl.sum(b_B, axis=1) + ea * tl.sum(b_k.to(tl.float32) * acc_G, axis=1) + acc_bv
    p_db = dbeta + i_b.to(tl.int64) * s_bb + i_h.to(tl.int64) * s_bh + idx.to(tl.int64) * s_bt
    tl.store(p_db, b_dbeta.to(p_db.dtype.element_ty), mask=m_t)

    ct = b_k.dtype
    b_dq = tl.dot(b_Ag.to(ct), b_k, input_precision=PREC) + ea[:, None] * acc_E
    b_dk = tl.dot(tl.trans(b_Ag).to(ct), b_q.to(ct), input_precision=PREC)
    b_dk += tl.dot(b_Pg.to(ct), b_k, input_precision=PREC)
    b_dk += tl.dot(tl.trans(b_Pg).to(ct), b_k, input_precision=PREC)
    b_dk += eda[:, None] * acc_KD + (b_beta * ea)[:, None] * acc_G

    p_out = i_b.to(tl.int64) * s_ob + i_h.to(tl.int64) * s_oh \
        + idx.to(tl.int64)[:, None] * s_ot + o_k[None, :]
    tl.store(dq + p_out, (b_dq * scale).to(dq.dtype.element_ty), mask=mm)
    tl.store(dk + p_out, b_dk.to(dk.dtype.element_ty), mask=mm)


@triton.jit
def _reduce_heads_kernel(src, dst, T, K, s_ob, s_ot, s_oh,
                         H: tl.constexpr, HV: tl.constexpr, REP: tl.constexpr,
                         BT: tl.constexpr, BK: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_hk = i_bh % H
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    idx = i_t * BT + o_t
    m_t = idx < T
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]
    acc = tl.zeros([BT, BK], dtype=tl.float32)
    for r in range(REP):
        i_h = i_hk * REP + r
        p = ((i_b.to(tl.int64) * HV + i_h) * T + idx.to(tl.int64))[:, None] * K + o_k[None, :]
        acc += tl.load(src + p, mask=m, other=0.0)
    p_o = dst + i_b.to(tl.int64) * s_ob + i_hk.to(tl.int64) * s_oh \
        + idx.to(tl.int64)[:, None] * s_ot + o_k[None, :]
    tl.store(p_o, acc.to(p_o.dtype.element_ty), mask=m)


def _tlty(dtype):
    return {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16,
            torch.float32: tl.float32}[dtype]


def _cfg(q, v):
    K, V = q.shape[-1], v.shape[-1]
    BK = triton.next_power_of_2(K)
    BV = min(64, triton.next_power_of_2(V))
    # The chunk scans are serial in NT, so their only parallelism is
    # (batch*head) x V-blocks; narrow the V block until the grid fills the device.
    nbh = q.shape[0] * v.shape[2]
    BVS = BV
    while BVS > 16 and nbh * triton.cdiv(V, BVS) < 128:
        BVS //= 2
    prec = "tf32x3" if q.dtype == torch.float32 else "tf32"
    return BK, BV, triton.cdiv(V, BV), prec, BVS, triton.cdiv(V, BVS)


def _launch_fwd(q, k, v, g, beta):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    REP = HV // H
    NT = triton.cdiv(T, BT)
    BK, BV, NV, prec, BVS, NVS = _cfg(q, v)
    NBH = B * HV
    dev = q.device

    ct = q.dtype
    a = torch.empty(B, HV, T, dtype=torch.float32, device=dev)
    eAs = torch.empty(B, HV, NT, dtype=torch.float32, device=dev)
    M = torch.empty(B, HV, NT, BT, BT, dtype=torch.float32, device=dev)
    u = torch.empty(B, HV, T, V, dtype=torch.float32, device=dev)
    h = torch.empty(B, HV, NT, K, V, dtype=torch.float32, device=dev)
    u0 = torch.empty(B, HV, T, V, dtype=torch.float32, device=dev)
    w = torch.empty(B, HV, T, K, dtype=ct, device=dev)
    w2 = torch.empty(B, HV, T, K, dtype=ct, device=dev)
    kbo = torch.empty(B, HV, T, K, dtype=ct, device=dev)
    kdo = torch.empty(B, HV, T, K, dtype=ct, device=dev)
    o = torch.empty_like(v)

    _cumsum_kernel[(NT, NBH)](g, a, eAs, T, g.stride(0), g.stride(1), g.stride(2),
                              HV=HV, NT=NT, BT=BT, num_warps=1)
    _wy_kernel[(NT, NBH)](k, v, beta, a, M, w, w2, u0, kbo, kdo, T, K, V,
                          k.stride(0), k.stride(1), k.stride(2),
                          v.stride(0), v.stride(1), v.stride(2),
                          beta.stride(0), beta.stride(1), beta.stride(2),
                          HV=HV, REP=REP, NT=NT, NV=NV, BT=BT, BK=BK, BV=BV, PREC=prec,
                          num_warps=4, num_stages=2)
    _fwd_scan_kernel[(NVS, NBH)](w, kdo, eAs, u0, u, h, T, K, V,
                                HV=HV, NT=NT, BT=BT, BK=BK, BV=BVS, PREC=prec,
                                CT=_tlty(ct), num_warps=4, num_stages=2)
    _out_kernel[(NT, NV, NBH)](q, k, a, u, h, o, T, K, V, K ** -0.5,
                               k.stride(0), k.stride(1), k.stride(2),
                               v.stride(0), v.stride(1), v.stride(2),
                               HV=HV, REP=REP, NT=NT, BT=BT, BK=BK, BV=BV, PREC=prec,
                               num_warps=4, num_stages=2)
    return o, a, eAs, M, u, h, w2, kbo


def _launch_bwd(q, k, v, g, beta, a, eAs, M, u, h, w2, kbo, do):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    REP = HV // H
    NT = triton.cdiv(T, BT)
    BK, BV, NV, prec, BVS, NVS = _cfg(q, v)
    NBH = B * HV
    dev = q.device

    dz = torch.empty(B, HV, T, V, dtype=torch.float32, device=dev)
    dh = torch.empty(B, HV, NT, K, V, dtype=torch.float32, device=dev)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float32)
    dbeta = torch.empty_like(beta, dtype=torch.float32)

    if REP == 1:
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        s_ob, s_ot, s_oh = q.stride(0), q.stride(1), q.stride(2)
        dq_t, dk_t = dq, dk
    else:
        dq_t = torch.empty(B, HV, T, K, dtype=torch.float32, device=dev)
        dk_t = torch.empty(B, HV, T, K, dtype=torch.float32, device=dev)
        s_ob, s_ot, s_oh = HV * T * K, K, T * K

    dz0 = torch.empty(B, HV, T, V, dtype=torch.float32, device=dev)
    dsc = torch.empty(B, HV, NT, K, V, dtype=torch.float32, device=dev)
    _bwd_prep_kernel[(NT, NV, NBH)](q, k, a, M, do, dz0, dsc, T, K, V, K ** -0.5,
                                    k.stride(0), k.stride(1), k.stride(2),
                                    v.stride(0), v.stride(1), v.stride(2),
                                    HV=HV, REP=REP, NT=NT, BT=BT, BK=BK, BV=BV, PREC=prec,
                                    num_warps=4, num_stages=2)
    _bwd_scan_kernel[(NVS, NBH)](w2, kbo, eAs, dz0, dsc, dz, dh, T, K, V,
                                HV=HV, NT=NT, BT=BT, BK=BK, BV=BVS, PREC=prec,
                                CT=_tlty(q.dtype), num_warps=4, num_stages=2)
    _bwd_grads_kernel[(NT, NBH)](q, k, v, beta, a, u, dz, h, dh, do,
                                 dq_t, dk_t, dv, dg, dbeta, T, K, V, K ** -0.5,
                                 k.stride(0), k.stride(1), k.stride(2),
                                 v.stride(0), v.stride(1), v.stride(2),
                                 beta.stride(0), beta.stride(1), beta.stride(2),
                                 g.stride(0), g.stride(1), g.stride(2),
                                 s_ob, s_ot, s_oh,
                                 HV=HV, REP=REP, NT=NT, NV=NV, BT=BT, BK=BK, BV=BV,
                                 PREC=prec, num_warps=8, num_stages=1)
    if REP != 1:
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        for src, dst in ((dq_t, dq), (dk_t, dk)):
            _reduce_heads_kernel[(NT, B * H)](src, dst, T, K,
                                              q.stride(0), q.stride(1), q.stride(2),
                                              H=H, HV=HV, REP=REP, BT=BT, BK=BK, num_warps=4)
    return dq, dk, dv, dg.to(g.dtype), dbeta.to(beta.dtype)


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        g, beta = g.contiguous(), beta.contiguous()
        o, a, eAs, M, u, h, w2, kbo = _launch_fwd(q, k, v, g, beta)
        ctx.save_for_backward(q, k, v, g, beta, a, eAs, M, u, h, w2, kbo)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, a, eAs, M, u, h, w2, kbo = ctx.saved_tensors
        return _launch_bwd(q, k, v, g, beta, a, eAs, M, u, h, w2, kbo, do.contiguous())


def gated_delta_rule(
    q: torch.Tensor,      # [B, T, H, K]
    k: torch.Tensor,      # [B, T, H, K]
    v: torch.Tensor,      # [B, T, H, V]
    g: torch.Tensor,      # [B, T, H]
    beta: torch.Tensor,   # [B, T, H]
) -> torch.Tensor:        # [B, T, H, V]
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _GatedDeltaRule.apply(q, k, v, g, beta)
