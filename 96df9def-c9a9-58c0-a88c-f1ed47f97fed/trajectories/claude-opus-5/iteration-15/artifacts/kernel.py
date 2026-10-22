"""Chunk-parallel DPLR (diagonal-plus-low-rank) generalized delta rule.

Forward and backward are computed entirely by the Triton kernels declared in
this module.  The algorithm is the chunkwise DPLR factorization:

  * the per-K-channel log-decay is accumulated inside every chunk (``gi``
    inclusive, ``ge`` exclusive),
  * the intra-chunk cascade builds the four coupling matrices ``A_qk``,
    ``A_qb``, ``A_ab``, ``A_ak`` as dense tile matmuls over the chunk,
  * the strictly lower-triangular nilpotent ``A_ab`` is inverted exactly by
    repeated squaring, giving the WY-style transform ``Tinv = (I - A_ab)^-1``,
  * the chunk-boundary state is carried in float32 through a sequential scan
    over chunks, fused with the output contraction,
  * the backward pass is a reverse float32 state scan plus one per-chunk local
    kernel that produces all six input gradients.

Nothing is cached between calls; every call recomputes from its arguments.
"""

import torch
import triton
import triton.language as tl

CHUNK = 16
EXP_CAP = tl.constexpr(40.0)




@triton.jit
def _tri_inv(b_N, BC: tl.constexpr, LOG_BC: tl.constexpr):
    """(I - N)^-1 for strictly lower triangular N, via repeated squaring."""
    o = tl.arange(0, BC)
    b_R = (o[:, None] == o[None, :]).to(tl.float32) + b_N
    b_P = b_N
    for _ in tl.static_range(LOG_BC - 1):
        b_P = tl.dot(b_P, b_P, input_precision="ieee")
        b_R = b_R + tl.dot(b_R, b_P, input_precision="ieee")
    return b_R


@triton.jit
def _fwd_prep(
    q, k, a, bt, gk, v,
    qg, kg, bg, kgl, bgl, W, U, Aqk, Aqb, Aak, Ainv, GL,
    T, NT, scale,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BC: tl.constexpr, LOG_BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DT: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_t = (i_t * BC + o_c) < T
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]

    base_in = (i_b * T + i_t * BC) * (H * K) + i_h * K
    rows = o_c[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(q + base_in + rows, mask=m, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + base_in + rows, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + base_in + rows, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(bt + base_in + rows, mask=m, other=0.0).to(tl.float32)
    b_gk = tl.load(gk + base_in + rows, mask=m, other=0.0).to(tl.float32)

    b_gi = tl.cumsum(b_gk, axis=0)
    b_ge = b_gi - b_gk
    b_gl = tl.sum(tl.where(o_c[:, None] == (BC - 1), b_gi, 0.0), 0)

    # Column-by-column construction of the four intra-chunk coupling matrices.
    # Referencing every decay to the source row keeps each exponential bounded,
    # which a single chunk-wide offset cannot do once the chunk decay is deep.
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqb = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aak = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aab = tl.zeros([BC, BC], dtype=tl.float32)
    for j in range(0, BC):
        m_j = o_c == j
        g_j = tl.sum(tl.where(m_j[:, None], b_gi, 0.0), 0)
        k_j = tl.sum(tl.where(m_j[:, None], b_k, 0.0), 0)
        b_j = tl.sum(tl.where(m_j[:, None], b_b, 0.0), 0)
        e1 = tl.exp(tl.minimum(b_gi - g_j[None, :], EXP_CAP))
        e2 = tl.exp(tl.minimum(b_ge - g_j[None, :], EXP_CAP))
        w_incl = (o_c >= j).to(tl.float32)
        w_strict = (o_c > j).to(tl.float32)
        c_qk = tl.sum(b_q * k_j[None, :] * e1, 1) * w_incl
        c_qb = tl.sum(b_q * b_j[None, :] * e1, 1) * w_incl
        c_ak = tl.sum(b_a * k_j[None, :] * e2, 1) * w_strict
        c_ab = tl.sum(b_a * b_j[None, :] * e2, 1) * w_strict
        b_Aqk += tl.where(m_j[None, :], c_qk[:, None], 0.0)
        b_Aqb += tl.where(m_j[None, :], c_qb[:, None], 0.0)
        b_Aak += tl.where(m_j[None, :], c_ak[:, None], 0.0)
        b_Aab += tl.where(m_j[None, :], c_ab[:, None], 0.0)

    b_T = _tri_inv(b_Aab, BC, LOG_BC)
    b_ag = b_a * tl.exp(tl.minimum(b_ge, EXP_CAP))
    b_w = tl.dot(b_T, b_ag, input_precision="ieee")

    base_c = (i_bh * T + i_t * BC) * K
    rows_c = o_c[:, None] * K + o_k[None, :]
    # The gated key / write factors are stored as an unevaluated (hi, lo) fp32
    # pair.  Their contraction against v is the whole intra-chunk contribution
    # to the boundary state, and rounding it to a single fp32 factor is what
    # sets the floor on the long-context carry.
    e_end = tl.exp(tl.minimum(b_gl[None, :] - b_gi, EXP_CAP).to(tl.float64))
    kf = b_k.to(tl.float64) * e_end
    bf = b_b.to(tl.float64) * e_end
    kh = kf.to(tl.float32)
    bh = bf.to(tl.float32)
    tl.store(qg + base_c + rows_c,
             (b_q * tl.exp(tl.minimum(b_gi, EXP_CAP))).to(DT), mask=m)
    tl.store(kg + base_c + rows_c, kh, mask=m)
    tl.store(bg + base_c + rows_c, bh, mask=m)
    tl.store(kgl + base_c + rows_c, (kf - kh.to(tl.float64)).to(tl.float32), mask=m)
    tl.store(bgl + base_c + rows_c, (bf - bh.to(tl.float64)).to(tl.float32), mask=m)
    tl.store(W + base_c + rows_c, b_w.to(DT), mask=m)
    tl.store(GL + (i_bh * NT + i_t) * K + o_k, b_gl, mask=m_k)

    base_a = (i_bh * NT + i_t) * BC * BC
    rows_a = o_c[:, None] * BC + o_c[None, :]
    tl.store(Aqk + base_a + rows_a, b_Aqk.to(DT))
    tl.store(Aqb + base_a + rows_a, b_Aqb.to(DT))
    tl.store(Aak + base_a + rows_a, b_Aak)
    tl.store(Ainv + base_a + rows_a, b_T)

    # u = Tinv @ (A_ak @ v)
    b_Aak_d = b_Aak.to(DT)
    b_T_d = b_T.to(DT)
    base_v = (i_b * T + i_t * BC) * (H * V) + i_h * V
    base_uv = (i_bh * T + i_t * BC) * V
    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v < V)[None, :]
        b_v = tl.load(v + base_v + o_c[:, None] * (H * V) + o_v[None, :],
                      mask=m_v, other=0.0).to(tl.float32)
        b_y = tl.dot(b_Aak_d, b_v.to(DT), input_precision=PREC)
        b_u = tl.dot(b_T_d, b_y.to(DT), input_precision=PREC)
        tl.store(U + base_uv + o_c[:, None] * V + o_v[None, :], b_u, mask=m_v)


@triton.jit
def _fwd_scan(
    v, qg, kg, bg, kgl, bgl, W, U, Aqk, Aqb, GL, HS, HSL, VN, O,
    T, NT,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DT: tl.constexpr, PREC: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_s = m_k[:, None] & m_v[None, :]

    b_S = tl.zeros([BK, BV], dtype=tl.float32)
    b_C = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(0, NT):
        m_t = (i_t * BC + o_c) < T
        m_kk = m_t[:, None] & m_k[None, :]
        m_vv = m_t[:, None] & m_v[None, :]

        # the boundary state is checkpointed as the fp32 pair (hi, lo); the lo
        # half is what the backward dq contraction cancels against
        off_h = ((i_bh * NT + i_t) * K + o_k)[:, None] * V + o_v[None, :]
        tl.store(HS + off_h, b_S, mask=m_s)
        tl.store(HSL + off_h, b_C, mask=m_s)

        base_c = (i_bh * T + i_t * BC) * K
        rows_c = o_c[:, None] * K + o_k[None, :]
        base_cv = (i_bh * T + i_t * BC) * V + o_c[:, None] * V + o_v[None, :]
        base_v = (i_b * T + i_t * BC) * (H * V) + i_h * V
        rows_v = o_c[:, None] * (H * V) + o_v[None, :]
        base_a = (i_bh * NT + i_t) * BC * BC
        rows_a = o_c[:, None] * BC + o_c[None, :]

        b_w = tl.load(W + base_c + rows_c, mask=m_kk, other=0.0)
        b_u = tl.load(U + base_cv, mask=m_vv, other=0.0)
        b_Sd = b_S.to(DT)
        b_Cd = b_C.to(DT)
        b_vn = (tl.dot(b_w, b_Sd, input_precision=PREC)
                + tl.dot(b_w, b_Cd, input_precision=PREC) + b_u)
        b_vnd = b_vn.to(DT)

        b_v = tl.load(v + base_v + rows_v, mask=m_vv, other=0.0).to(tl.float32)
        b_Aqk = tl.load(Aqk + base_a + rows_a)
        b_Aqb = tl.load(Aqb + base_a + rows_a)
        b_qg = tl.load(qg + base_c + rows_c, mask=m_kk, other=0.0)

        b_o = tl.dot(b_qg, b_Sd, input_precision=PREC)
        b_o += tl.dot(b_qg, b_Cd, input_precision=PREC)
        b_o += tl.dot(b_Aqk, b_v, input_precision=PREC)
        b_o += tl.dot(b_Aqb, b_vnd, input_precision=PREC)
        tl.store(O + base_v + rows_v, b_o.to(O.dtype.element_ty), mask=m_vv)
        tl.store(VN + base_cv, b_vnd, mask=m_vv)

        b_kg = tl.load(kg + base_c + rows_c, mask=m_kk, other=0.0)
        b_bg = tl.load(bg + base_c + rows_c, mask=m_kk, other=0.0)
        b_gl = tl.load(GL + (i_bh * NT + i_t) * K + o_k, mask=m_k, other=0.0)
        b_dec = tl.exp(tl.minimum(b_gl, EXP_CAP))[:, None]
        # compensated carry: the discarded low half of every chunk update is
        # scaled forward, so the fp32 state does not lose the small components
        # that later cancel against a much larger one in the output
        b_kgl = tl.load(kgl + base_c + rows_c, mask=m_kk, other=0.0)
        b_bgl = tl.load(bgl + base_c + rows_c, mask=m_kk, other=0.0)
        b_i1 = tl.dot(tl.trans(b_kg), b_v, input_precision=PREC)
        b_i2 = tl.dot(tl.trans(b_bg), b_vnd, input_precision=PREC)
        b_lo = (tl.dot(tl.trans(b_kgl), b_v, input_precision=PREC)
                + tl.dot(tl.trans(b_bgl), b_vnd, input_precision=PREC))
        b_inc = b_i1 + b_i2
        b_Cm = b_C * b_dec + ((b_i1 - b_inc) + b_i2) + b_lo
        b_Sm = b_S * b_dec
        b_hi = b_Sm + b_inc
        b_C = ((b_Sm - b_hi) + b_inc) + b_Cm
        b_S = b_hi


@triton.jit
def _bwd_scan(
    do, qg, bg, W, Aqb, GL, DH, DVN,
    T, NT,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DT: tl.constexpr, PREC: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_s = m_k[:, None] & m_v[None, :]

    b_dS = tl.zeros([BK, BV], dtype=tl.float32)
    for i_ in range(0, NT):
        i_t = NT - 1 - i_
        m_t = (i_t * BC + o_c) < T
        m_kk = m_t[:, None] & m_k[None, :]
        m_vv = m_t[:, None] & m_v[None, :]

        base_c = (i_bh * T + i_t * BC) * K
        rows_c = o_c[:, None] * K + o_k[None, :]
        base_v = (i_b * T + i_t * BC) * (H * V) + i_h * V
        rows_v = o_c[:, None] * (H * V) + o_v[None, :]
        base_a = (i_bh * NT + i_t) * BC * BC
        rows_a = o_c[:, None] * BC + o_c[None, :]

        b_do = tl.load(do + base_v + rows_v, mask=m_vv, other=0.0).to(tl.float32)
        b_Aqb = tl.load(Aqb + base_a + rows_a)
        b_bg = tl.load(bg + base_c + rows_c, mask=m_kk, other=0.0)
        b_dSd = b_dS.to(DT)
        b_dvn = tl.dot(tl.trans(b_Aqb), b_do, input_precision=PREC)
        b_dvn += tl.dot(b_bg, b_dSd, input_precision=PREC)
        tl.store(DVN + (i_bh * T + i_t * BC) * V + o_c[:, None] * V + o_v[None, :],
                 b_dvn, mask=m_vv)

        b_qg = tl.load(qg + base_c + rows_c, mask=m_kk, other=0.0)
        b_w = tl.load(W + base_c + rows_c, mask=m_kk, other=0.0)
        b_gl = tl.load(GL + (i_bh * NT + i_t) * K + o_k, mask=m_k, other=0.0)
        b_dS = b_dS * tl.exp(tl.minimum(b_gl, EXP_CAP))[:, None]
        b_dS += tl.dot(tl.trans(b_qg), b_do, input_precision=PREC)
        b_dS += tl.dot(tl.trans(b_w), b_dvn.to(DT), input_precision=PREC)
        tl.store(DH + ((i_bh * (NT + 1) + i_t) * K + o_k)[:, None] * V + o_v[None, :],
                 b_dS, mask=m_s)


@triton.jit
def _bwd_local(
    q, k, a, bt, gk, v, do, VN, DVN, HS, HSL, DH, qg, kg, W, Aqk, Aak, Ainv,
    dq, dk, dv, da, db, dgk,
    T, NT, scale,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DT: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_t = (i_t * BC + o_c) < T
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]

    base_in = (i_b * T + i_t * BC) * (H * K) + i_h * K
    rows = o_c[:, None] * (H * K) + o_k[None, :]
    base_c = (i_bh * T + i_t * BC) * K
    rows_c = o_c[:, None] * K + o_k[None, :]
    base_a = (i_bh * NT + i_t) * BC * BC
    rows_a = o_c[:, None] * BC + o_c[None, :]

    b_Aqk = tl.load(Aqk + base_a + rows_a)
    b_Aak = tl.load(Aak + base_a + rows_a)
    b_T = tl.load(Ainv + base_a + rows_a)
    b_Aak_d = b_Aak.to(DT)
    b_T_d = b_T.to(DT)
    b_kg = tl.load(kg + base_c + rows_c, mask=m, other=0.0)

    b_dAqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_dAqb = tl.zeros([BC, BC], dtype=tl.float32)
    b_dAak = tl.zeros([BC, BC], dtype=tl.float32)
    b_dT = tl.zeros([BC, BC], dtype=tl.float32)
    b_dqg = tl.zeros([BC, BK], dtype=tl.float32)
    b_dw = tl.zeros([BC, BK], dtype=tl.float32)
    b_dkg = tl.zeros([BC, BK], dtype=tl.float32)
    b_dbg = tl.zeros([BC, BK], dtype=tl.float32)
    b_dgl = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_vv = m_t[:, None] & m_v[None, :]
        m_s = m_k[:, None] & m_v[None, :]
        base_v = (i_b * T + i_t * BC) * (H * V) + i_h * V
        rows_v = o_c[:, None] * (H * V) + o_v[None, :]
        base_cv = (i_bh * T + i_t * BC) * V + o_c[:, None] * V + o_v[None, :]

        b_v = tl.load(v + base_v + rows_v, mask=m_vv, other=0.0).to(tl.float32)
        b_do = tl.load(do + base_v + rows_v, mask=m_vv, other=0.0).to(tl.float32)
        b_vn = tl.load(VN + base_cv, mask=m_vv, other=0.0)
        b_dvn = tl.load(DVN + base_cv, mask=m_vv, other=0.0)
        off_h = ((i_bh * NT + i_t) * K + o_k)[:, None] * V + o_v[None, :]
        b_h = tl.load(HS + off_h, mask=m_s, other=0.0)
        b_hl = tl.load(HSL + off_h, mask=m_s, other=0.0)
        b_dSc = tl.load(
            DH + ((i_bh * (NT + 1) + i_t + 1) * K + o_k)[:, None] * V + o_v[None, :],
            mask=m_s, other=0.0)

        b_dvnd = b_dvn.to(DT)
        b_y = tl.dot(b_Aak_d, b_v, input_precision=PREC)
        b_dy = tl.dot(tl.trans(b_T_d), b_dvnd, input_precision=PREC)
        b_dyd = b_dy.to(DT)
        b_dScd = b_dSc.to(DT)
        b_hd = b_h.to(DT)

        b_dv = tl.dot(tl.trans(b_Aqk), b_do, input_precision=PREC)
        b_dv += tl.dot(b_kg, b_dScd, input_precision=PREC)
        b_dv += tl.dot(tl.trans(b_Aak_d), b_dyd, input_precision=PREC)
        tl.store(dv + base_v + rows_v, b_dv.to(dv.dtype.element_ty), mask=m_vv)

        b_dAqk += tl.dot(b_do, tl.trans(b_v), input_precision=PREC)
        b_dAqb += tl.dot(b_do, tl.trans(b_vn), input_precision=PREC)
        b_dAak += tl.dot(b_dyd, tl.trans(b_v), input_precision=PREC)
        b_dT += tl.dot(b_dvnd, tl.trans(b_y.to(DT)), input_precision=PREC)

        b_hld = b_hl.to(DT)
        b_dqg += tl.dot(b_do, tl.trans(b_hd), input_precision=PREC)
        b_dqg += tl.dot(b_do, tl.trans(b_hld), input_precision=PREC)
        b_dw += tl.dot(b_dvnd, tl.trans(b_hd), input_precision=PREC)
        b_dw += tl.dot(b_dvnd, tl.trans(b_hld), input_precision=PREC)
        b_dkg += tl.dot(b_v, tl.trans(b_dScd), input_precision=PREC)
        b_dbg += tl.dot(b_vn, tl.trans(b_dScd), input_precision=PREC)
        b_dgl += tl.sum(b_h * b_dSc, 1)

    m_incl = o_c[:, None] >= o_c[None, :]
    m_strict = o_c[:, None] > o_c[None, :]
    b_dAqk = tl.where(m_incl, b_dAqk, 0.0)
    b_dAqb = tl.where(m_incl, b_dAqb, 0.0)
    b_dAak = tl.where(m_strict, b_dAak, 0.0)

    # rebuild the chunk-local gated operands
    b_q = tl.load(q + base_in + rows, mask=m, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + base_in + rows, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + base_in + rows, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(bt + base_in + rows, mask=m, other=0.0).to(tl.float32)
    b_gk = tl.load(gk + base_in + rows, mask=m, other=0.0).to(tl.float32)
    b_gi = tl.cumsum(b_gk, axis=0)
    b_ge = b_gi - b_gk
    b_gl = tl.sum(tl.where(o_c[:, None] == (BC - 1), b_gi, 0.0), 0)

    e_g = tl.exp(tl.minimum(b_gi, EXP_CAP))
    e_e = tl.exp(tl.minimum(b_ge, EXP_CAP))
    e_end = tl.exp(tl.minimum(b_gl[None, :] - b_gi, EXP_CAP))
    b_ag = b_a * e_e

    b_dT += tl.dot(b_dw.to(DT), tl.trans(b_ag.to(DT)), input_precision=PREC)
    b_dAab = tl.where(
        m_strict,
        tl.dot(tl.trans(b_T), tl.dot(b_dT, tl.trans(b_T), input_precision="ieee"),
               input_precision="ieee"),
        0.0,
    )
    b_dag = tl.dot(tl.trans(b_T), b_dw, input_precision="ieee")

    # Adjoint of the column-by-column coupling-matrix construction.
    # Every gate factor is accumulated directly onto the gk timesteps that
    # actually appear inside its exponent.  Accumulating onto gi / ge first and
    # suffix-summing afterwards would only cancel the shared offset of a chunk
    # to round-off, which the tolerance on a structurally-zero gk cannot absorb.
    d_q = tl.zeros([BC, BK], dtype=tl.float32)
    d_a = tl.zeros([BC, BK], dtype=tl.float32)
    d_k = tl.zeros([BC, BK], dtype=tl.float32)
    d_b = tl.zeros([BC, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BC, BK], dtype=tl.float32)
    acc_p = tl.zeros([BC, BK], dtype=tl.float32)
    acc_q = tl.zeros([BC, BK], dtype=tl.float32)
    # suffix-sum operators; a direct sum avoids the cancellation of total-minus-prefix
    m_suf = (o_c[None, :] >= o_c[:, None]).to(tl.float32)
    m_sufx = (o_c[None, :] > o_c[:, None]).to(tl.float32)
    m_pre = (o_c[None, :] < o_c[:, None]).to(tl.float32)
    for j in range(0, BC):
        m_j = o_c == j
        g_j = tl.sum(tl.where(m_j[:, None], b_gi, 0.0), 0)
        k_j = tl.sum(tl.where(m_j[:, None], b_k, 0.0), 0)
        b_j = tl.sum(tl.where(m_j[:, None], b_b, 0.0), 0)
        e1 = tl.exp(tl.minimum(b_gi - g_j[None, :], EXP_CAP))
        e2 = tl.exp(tl.minimum(b_ge - g_j[None, :], EXP_CAP))
        c_qk = tl.sum(tl.where(m_j[None, :], b_dAqk, 0.0), 1)
        c_qb = tl.sum(tl.where(m_j[None, :], b_dAqb, 0.0), 1)
        c_ak = tl.sum(tl.where(m_j[None, :], b_dAak, 0.0), 1)
        c_ab = tl.sum(tl.where(m_j[None, :], b_dAab, 0.0), 1)

        f_q = (c_qk[:, None] * k_j[None, :] + c_qb[:, None] * b_j[None, :]) * e1
        f_a = (c_ak[:, None] * k_j[None, :] + c_ab[:, None] * b_j[None, :]) * e2
        d_q += f_q
        d_a += f_a
        col_k = tl.sum(c_qk[:, None] * b_q * e1 + c_ak[:, None] * b_a * e2, 0)
        col_b = tl.sum(c_qb[:, None] * b_q * e1 + c_ab[:, None] * b_a * e2, 0)
        d_k += tl.where(m_j[:, None], col_k[None, :], 0.0)
        d_b += tl.where(m_j[:, None], col_b[None, :], 0.0)

        # exponent gi_s - gi_j spans gk[j+1 .. s]; ge_s - gi_j spans gk[j+1 .. s-1].
        # dgk[t] collects sum_{s >= t} of the columns j < t, so a running sum
        # over j lets each iteration emit exactly one finished row (t = j + 1)
        # with a single masked reduction instead of a full suffix matmul.
        acc_p += b_q * f_q
        acc_q += b_a * f_a
        r_p = tl.sum(tl.where((o_c >= j + 1)[:, None], acc_p, 0.0), 0)
        r_q = tl.sum(tl.where((o_c >= j + 2)[:, None], acc_q, 0.0), 0)
        b_dgk += tl.where((o_c == j + 1)[:, None], (r_p + r_q)[None, :], 0.0)

    b_qgf = b_q * e_g
    b_kgf = b_k * e_end
    b_bgf = b_b * e_end

    tl.store(dq + base_in + rows, ((d_q + b_dqg * e_g) * scale).to(dq.dtype.element_ty), mask=m)
    tl.store(dk + base_in + rows, (d_k + b_dkg * e_end).to(dk.dtype.element_ty), mask=m)
    tl.store(da + base_in + rows, (d_a + b_dag * e_e).to(da.dtype.element_ty), mask=m)
    tl.store(db + base_in + rows, (d_b + b_dbg * e_end).to(db.dtype.element_ty), mask=m)

    # exp(gi_t) spans gk[0 .. t]; exp(ge_t) spans gk[0 .. t-1]
    R = b_qgf * b_dqg
    A2 = b_ag * b_dag
    b_dgk += tl.dot(m_suf, R, input_precision="ieee")
    b_dgk += tl.dot(m_sufx, A2, input_precision="ieee")
    # exp(gl - gi_t) spans gk[t+1 .. BC-1]
    E = b_kgf * b_dkg + b_bgf * b_dbg
    b_dgk += tl.dot(m_pre, E, input_precision="ieee")
    # the chunk-boundary decay exp(gl) spans the whole chunk
    b_dgk += (b_dgl * tl.exp(tl.minimum(b_gl, EXP_CAP)))[None, :]
    tl.store(dgk + base_in + rows, b_dgk.to(dgk.dtype.element_ty), mask=m)


def _tl_dtype(t):
    return {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16,
            torch.float32: tl.float32}[t.dtype]


def _fwd(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H
    BC = CHUNK
    LOG_BC = BC.bit_length() - 1
    NT = triton.cdiv(T, BC)
    BK = max(triton.next_power_of_2(K), 16)
    BVs = 16
    BVf = min(max(triton.next_power_of_2(V), 16), 64)
    DT = tl.float32
    PREC = "tf32x3"
    scale = K ** -0.5

    f32 = dict(dtype=torch.float32, device=q.device)
    qg = torch.empty(BH * T * K, **f32)
    kg = torch.empty(BH * T * K, **f32)
    bg = torch.empty(BH * T * K, **f32)
    kgl = torch.empty(BH * T * K, **f32)
    bgl = torch.empty(BH * T * K, **f32)
    W = torch.empty(BH * T * K, **f32)
    U = torch.empty(BH * T * V, **f32)
    Aqk = torch.empty(BH * NT * BC * BC, **f32)
    Aqb = torch.empty(BH * NT * BC * BC, **f32)
    Aak = torch.empty(BH * NT * BC * BC, dtype=torch.float32, device=q.device)
    Ainv = torch.empty(BH * NT * BC * BC, dtype=torch.float32, device=q.device)
    GL = torch.empty(BH * NT * K, dtype=torch.float32, device=q.device)

    _fwd_prep[(NT, BH)](
        q, k, alpha, beta, gk, v,
        qg, kg, bg, kgl, bgl, W, U, Aqk, Aqb, Aak, Ainv, GL,
        T, NT, scale, H=H, K=K, V=V, BC=BC, LOG_BC=LOG_BC, BK=BK, BV=BVf, DT=DT, PREC=PREC,
        num_warps=4, num_stages=2,
    )

    HS = torch.empty(BH * NT * K * V, dtype=torch.float32, device=q.device)
    HSL = torch.empty(BH * NT * K * V, dtype=torch.float32, device=q.device)
    VN = torch.empty(BH * T * V, **f32)
    o = torch.empty_like(v)
    _fwd_scan[(triton.cdiv(V, BVs), BH)](
        v, qg, kg, bg, kgl, bgl, W, U, Aqk, Aqb, GL, HS, HSL, VN, o,
        T, NT, H=H, K=K, V=V, BC=BC, BK=BK, BV=BVs, DT=DT, PREC=PREC,
        num_warps=4, num_stages=2,
    )
    saved = (qg, kg, bg, W, Aqk, Aqb, Aak, Ainv, GL, HS, HSL, VN)
    return o, saved, (BC, NT, BK, BVs, BVf, DT, PREC, scale)


def _bwd(do, q, k, v, alpha, beta, gk, saved, cfg):
    qg, kg, bg, W, Aqk, Aqb, Aak, Ainv, GL, HS, HSL, VN = saved
    BC, NT, BK, BVs, BVf, DT, PREC, scale = cfg
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H

    DH = torch.zeros(BH * (NT + 1) * K * V, dtype=torch.float32, device=q.device)
    DVN = torch.empty(BH * T * V, dtype=torch.float32, device=q.device)
    _bwd_scan[(triton.cdiv(V, BVs), BH)](
        do, qg, bg, W, Aqb, GL, DH, DVN,
        T, NT, H=H, K=K, V=V, BC=BC, BK=BK, BV=BVs, DT=DT, PREC=PREC,
        num_warps=4, num_stages=2,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    da = torch.empty_like(alpha)
    db = torch.empty_like(beta)
    dgkr = torch.empty_like(gk)
    _bwd_local[(NT, BH)](
        q, k, alpha, beta, gk, v, do, VN, DVN, HS, HSL, DH, qg, kg, W, Aqk, Aak, Ainv,
        dq, dk, dv, da, db, dgkr,
        T, NT, scale, H=H, K=K, V=V, BC=BC, BK=BK, BV=BVf, DT=DT, PREC=PREC,
        num_warps=4, num_stages=2,
    )
    return dq, dk, dv, da, db, dgkr


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        o, saved, cfg = _fwd(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, *saved)
        ctx.cfg = cfg
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta, gk = ctx.saved_tensors[:6]
        saved = ctx.saved_tensors[6:]
        return _bwd(do.contiguous(), q, k, v, alpha, beta, gk, saved, ctx.cfg)


def chunked_dplr_delta(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]    DPLR erasure basis
    beta: torch.Tensor,  # [B, T, H, K]    DPLR write basis
    gk: torch.Tensor,  # [B, T, H, K]    per-K-channel log-decay (<= 0)
) -> torch.Tensor:  # [B, T, H, V]
    """Entry point. Signature and semantics are fixed."""
    return _Dplr.apply(q, k, v, alpha, beta, gk)
