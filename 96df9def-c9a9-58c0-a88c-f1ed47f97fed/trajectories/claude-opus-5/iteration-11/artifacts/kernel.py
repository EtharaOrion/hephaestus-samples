"""Chunk-parallel DPLR generalized delta rule (RWKV-7 family), forward + backward.

Per (batch, head), state S in R^{K x V}:

    S_t = diag(exp(g_t)) S_{t-1} + k_t v_t^T + beta_t (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))

Chunkwise factorization (chunk length C = BT; G = inclusive cumsum of g inside
the chunk in float32, gl = G_{C-1}):

    Qg = q*scale*e^{G}   Kg = k*e^{gl-G}   Bg = beta*e^{gl-G}   Ag = alpha*e^{G-g}
    A_qk[i,j] = q_i.k_j e^{G_i-G_j}      A_qb[i,j] = q_i.beta_j e^{G_i-G_j}   (i>=j)
    A_ak[i,j] = a_i.k_j e^{G_i-g_i-G_j}  A_ab[i,j] = a_i.beta_j e^{...}       (i> j)
    M    = (I - A_ab)^{-1}              (exact; A_ab strictly lower => nilpotent)
    U    = M (A_ak V + Ag S_in)         (intra-chunk rank-1 reads)
    O    = Qg S_in + A_qk V + A_qb U
    S_out = e^{gl} S_in + Kg^T V + Bg^T U

The A entries depend only on gate differences G_i - G_j, so each is built from
that difference directly rather than as a product e^{+G_i} . e^{-G_j}: under
heavy decay the split form makes both operands span the chunk's whole dynamic
range and loses the small entries inside the K reduction. The backward pass
forms its intra-chunk operand gradients, and the whole of dgk, the same way.

The boundary state and every scratch buffer that carries it (h, dh, U, dU) stay
in float32 for their whole life, forward and backward.
"""

import torch
import triton
import triton.language as tl

BT = 16
EXP_CAP = tl.constexpr(60.0)


@triton.jit
def _cexp(x):
    return tl.exp(tl.minimum(tl.maximum(x, -EXP_CAP), EXP_CAP))


@triton.jit
def dplr_prep_kernel(
    q, k, a, b, g,
    Qg, Kg, Bg, Ag, Aqk, Aqb, Aak, Minv, Wa, W1,
    scale, T, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.num_programs(0)

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    rows = i_t * BT + o_t
    m_t = rows < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]

    base = (i_b * T * H + i_h).to(tl.int64) * K
    off = base + rows[:, None].to(tl.int64) * (H * K) + o_k[None, :]

    b_q = tl.load(q + off, mask=m_tk, other=0.0).to(tl.float32)
    b_k = tl.load(k + off, mask=m_tk, other=0.0).to(tl.float32)
    b_a = tl.load(a + off, mask=m_tk, other=0.0).to(tl.float32)
    b_b = tl.load(b + off, mask=m_tk, other=0.0).to(tl.float32)
    b_g = tl.load(g + off, mask=m_tk, other=0.0).to(tl.float32)

    # per-K-channel cumulative gate inside the chunk, float32
    b_G = tl.cumsum(b_g, axis=0)
    b_gl = tl.sum(b_g, 0)

    b_qg = tl.where(m_tk, b_q * _cexp(b_G) * scale, 0.0)
    b_kg = tl.where(m_tk, b_k * _cexp(b_gl[None, :] - b_G), 0.0)
    b_bg = tl.where(m_tk, b_b * _cexp(b_gl[None, :] - b_G), 0.0)
    b_ag = tl.where(m_tk, b_a * _cexp(b_G - b_g), 0.0)
    tl.store(Qg + off, b_qg, mask=m_tk)
    tl.store(Kg + off, b_kg, mask=m_tk)
    tl.store(Bg + off, b_bg, mask=m_tk)
    tl.store(Ag + off, b_ag, mask=m_tk)

    # The A entries depend only on gate differences G_i - G_j, so they are built
    # one source column at a time with the difference formed exactly. Splitting
    # them as e^{+G} . e^{-G} instead would make each operand span the chunk's
    # whole decay range and lose the small entries inside the K reduction.
    b_qs = tl.where(m_tk, b_q * scale, 0.0)
    b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_Aqb = tl.zeros([BT, BT], dtype=tl.float32)
    b_Aak = tl.zeros([BT, BT], dtype=tl.float32)
    b_Aab = tl.zeros([BT, BT], dtype=tl.float32)
    for j in tl.static_range(BT):
        m_j = o_t == j
        b_kj = tl.sum(tl.where(m_j[:, None], b_k, 0.0), 0)
        b_bj = tl.sum(tl.where(m_j[:, None], b_b, 0.0), 0)
        b_Gj = tl.sum(tl.where(m_j[:, None], b_G, 0.0), 0)
        b_dq_ = _cexp(b_G - b_Gj[None, :])
        b_da_ = _cexp(b_G - b_g - b_Gj[None, :])
        b_col = (o_t == j)[None, :]
        b_Aqk += tl.where(b_col, tl.sum(b_qs * b_kj[None, :] * b_dq_, 1)[:, None], 0.0)
        b_Aqb += tl.where(b_col, tl.sum(b_qs * b_bj[None, :] * b_dq_, 1)[:, None], 0.0)
        b_Aak += tl.where(b_col, tl.sum(b_a * b_kj[None, :] * b_da_, 1)[:, None], 0.0)
        b_Aab += tl.where(b_col, tl.sum(b_a * b_bj[None, :] * b_da_, 1)[:, None], 0.0)

    m_incl = o_t[:, None] >= o_t[None, :]
    m_strict = o_t[:, None] > o_t[None, :]
    b_Aqk = tl.where(m_incl, b_Aqk, 0.0)
    b_Aqb = tl.where(m_incl, b_Aqb, 0.0)
    b_Aak = tl.where(m_strict, b_Aak, 0.0)
    b_Aab = tl.where(m_strict, b_Aab, 0.0)

    # Exact inverse of the unit lower triangular (I - A_ab). A_ab is strictly
    # lower and hence nilpotent, so the repeated-squaring product terminates:
    #   (I - A)^-1 = (I + A)(I + A^2)(I + A^4)...   for BT = 16, three squarings.
    # This runs wider than float32: the inverse can reach ~1e4 while the entries
    # it must resolve are O(1), and that error would then enter the state carry.
    b_Mw = tl.where(o_t[:, None] == o_t[None, :], 1.0, 0.0).to(tl.float64) + b_Aab.to(tl.float64)
    b_P = b_Aab.to(tl.float64)
    for _ in tl.static_range(3):  # log2(BT) - 1
        b_P = tl.sum(b_P[:, :, None] * b_P[None, :, :], 1)
        b_Mw = b_Mw + tl.sum(b_Mw[:, :, None] * b_P[None, :, :], 1)
    b_M = b_Mw.to(tl.float32)
    b_W1 = tl.dot(b_M, b_Aak, input_precision=PREC)
    b_Wa = tl.dot(b_M, b_ag, input_precision=PREC)

    o_a = (((i_bh * NT + i_t) * BT + o_t[:, None]) * BT + o_t[None, :]).to(tl.int64)
    tl.store(Aqk + o_a, b_Aqk)
    tl.store(Aqb + o_a, b_Aqb)
    tl.store(Aak + o_a, b_Aak)
    tl.store(Minv + o_a, b_M)
    tl.store(W1 + o_a, b_W1)
    o_w = ((i_bh * NT + i_t) * BT + o_t[:, None]).to(tl.int64) * K + o_k[None, :]
    tl.store(Wa + o_w, b_Wa, mask=m_k[None, :])


@triton.jit
def dplr_fwd_scan_kernel(
    v, Kg, Bg, Wa, W1, g, h, U,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    b_S = tl.zeros([BK, BV], dtype=tl.float32)
    kbase = (i_b * T * H + i_h).to(tl.int64) * K
    vbase = (i_b * T * H + i_h).to(tl.int64) * V

    for i_t in range(NT):
        rows = i_t * BT + o_t
        m_t = rows < T
        m_tk = m_t[:, None] & m_k[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        offk = kbase + rows[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        offv = vbase + rows[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        o_a = (((i_bh * NT + i_t) * BT + o_t[:, None]) * BT + o_t[None, :]).to(tl.int64)
        o_w = ((i_bh * NT + i_t) * BT + o_t[:, None]).to(tl.int64) * K + o_k[None, :]
        o_hs = ((i_bh * NT + i_t) * K + o_k[:, None]).to(tl.int64) * V + o_v[None, :]
        o_u = ((i_bh * NT + i_t) * BT + o_t[:, None]).to(tl.int64) * V + o_v[None, :]

        b_v = tl.load(v + offv, mask=m_tv, other=0.0).to(tl.float32)
        b_W1 = tl.load(W1 + o_a)
        b_Wa = tl.load(Wa + o_w, mask=m_k[None, :], other=0.0)

        tl.store(h + o_hs, b_S, mask=m_kv)

        b_U = tl.dot(b_W1, b_v, input_precision=PREC) + tl.dot(b_Wa, b_S, input_precision=PREC)
        b_U = tl.where(m_tv, b_U, 0.0)
        tl.store(U + o_u, b_U, mask=m_tv)

        b_kg = tl.load(Kg + offk, mask=m_tk, other=0.0)
        b_bg = tl.load(Bg + offk, mask=m_tk, other=0.0)
        b_g = tl.load(g + offk, mask=m_tk, other=0.0).to(tl.float32)
        b_gl = tl.sum(b_g, 0)

        b_S = b_S * _cexp(b_gl)[:, None]
        b_S += tl.dot(tl.trans(b_kg), b_v, input_precision=PREC)
        b_S += tl.dot(tl.trans(b_bg), b_U, input_precision=PREC)


@triton.jit
def dplr_fwd_o_kernel(
    Qg, v, U, Aqk, Aqb, h, o,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.num_programs(0)

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    rows = i_t * BT + o_t
    m_t = rows < T
    m_k = o_k < K
    m_v = o_v < V
    m_tk = m_t[:, None] & m_k[None, :]
    m_tv = m_t[:, None] & m_v[None, :]

    offk = (i_b * T * H + i_h).to(tl.int64) * K + rows[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    offv = (i_b * T * H + i_h).to(tl.int64) * V + rows[:, None].to(tl.int64) * (H * V) + o_v[None, :]
    o_a = (((i_bh * NT + i_t) * BT + o_t[:, None]) * BT + o_t[None, :]).to(tl.int64)
    o_hs = ((i_bh * NT + i_t) * K + o_k[:, None]).to(tl.int64) * V + o_v[None, :]
    o_u = ((i_bh * NT + i_t) * BT + o_t[:, None]).to(tl.int64) * V + o_v[None, :]

    b_qg = tl.load(Qg + offk, mask=m_tk, other=0.0)
    b_h = tl.load(h + o_hs, mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_v = tl.load(v + offv, mask=m_tv, other=0.0).to(tl.float32)
    b_U = tl.load(U + o_u, mask=m_tv, other=0.0)
    b_Aqk = tl.load(Aqk + o_a)
    b_Aqb = tl.load(Aqb + o_a)

    b_o = tl.dot(b_qg, b_h, input_precision=PREC)
    b_o += tl.dot(b_Aqk, b_v, input_precision=PREC)
    b_o += tl.dot(b_Aqb, b_U, input_precision=PREC)
    tl.store(o + offv, b_o.to(o.dtype.element_ty), mask=m_tv)


@triton.jit
def dplr_bwd_du_kernel(
    do, Aqb, dU,
    T, H: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.num_programs(0)

    o_t = tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    rows = i_t * BT + o_t
    m_t = rows < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]

    offv = (i_b * T * H + i_h).to(tl.int64) * V + rows[:, None].to(tl.int64) * (H * V) + o_v[None, :]
    o_a = (((i_bh * NT + i_t) * BT + o_t[:, None]) * BT + o_t[None, :]).to(tl.int64)
    o_u = ((i_bh * NT + i_t) * BT + o_t[:, None]).to(tl.int64) * V + o_v[None, :]

    b_do = tl.load(do + offv, mask=m_tv, other=0.0).to(tl.float32)
    b_Aqb = tl.load(Aqb + o_a)
    b_dU = tl.dot(tl.trans(b_Aqb), b_do, input_precision=PREC)
    tl.store(dU + o_u, tl.where(m_tv, b_dU, 0.0), mask=m_tv)


@triton.jit
def dplr_bwd_scan_kernel(
    do, Qg, Bg, Wa, g, dU, dh,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    b_dS = tl.zeros([BK, BV], dtype=tl.float32)
    kbase = (i_b * T * H + i_h).to(tl.int64) * K
    vbase = (i_b * T * H + i_h).to(tl.int64) * V

    for i_r in range(NT):
        i_t = NT - 1 - i_r
        rows = i_t * BT + o_t
        m_t = rows < T
        m_tk = m_t[:, None] & m_k[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        offk = kbase + rows[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        offv = vbase + rows[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        o_w = ((i_bh * NT + i_t) * BT + o_t[:, None]).to(tl.int64) * K + o_k[None, :]
        o_hs = ((i_bh * NT + i_t) * K + o_k[:, None]).to(tl.int64) * V + o_v[None, :]
        o_u = ((i_bh * NT + i_t) * BT + o_t[:, None]).to(tl.int64) * V + o_v[None, :]

        # b_dS entering the iteration is the gradient wrt this chunk's OUT state
        tl.store(dh + o_hs, b_dS, mask=m_kv)

        b_bg = tl.load(Bg + offk, mask=m_tk, other=0.0)
        b_dU = tl.load(dU + o_u, mask=m_tv, other=0.0)
        b_dU += tl.dot(b_bg, b_dS, input_precision=PREC)
        b_dU = tl.where(m_tv, b_dU, 0.0)
        tl.store(dU + o_u, b_dU, mask=m_tv)

        b_qg = tl.load(Qg + offk, mask=m_tk, other=0.0)
        b_do = tl.load(do + offv, mask=m_tv, other=0.0).to(tl.float32)
        b_Wa = tl.load(Wa + o_w, mask=m_k[None, :], other=0.0)
        b_g = tl.load(g + offk, mask=m_tk, other=0.0).to(tl.float32)
        b_gl = tl.sum(b_g, 0)

        b_dS = b_dS * _cexp(b_gl)[:, None]
        b_dS += tl.dot(tl.trans(b_qg), b_do, input_precision=PREC)
        b_dS += tl.dot(tl.trans(b_Wa), b_dU, input_precision=PREC)


@triton.jit
def dplr_bwd_chunk_kernel(
    q, k, a, b, g, v, do, Qg, Kg, Bg, Ag, Aqk, Aak, Minv, h, dh, U, dU,
    dq, dk, dv, da, db, dg,
    scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.num_programs(0)

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    rows = i_t * BT + o_t
    m_t = rows < T
    m_k = o_k < K
    m_v = o_v < V
    m_tk = m_t[:, None] & m_k[None, :]
    m_tv = m_t[:, None] & m_v[None, :]
    m_kv = m_k[:, None] & m_v[None, :]

    offk = (i_b * T * H + i_h).to(tl.int64) * K + rows[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    offv = (i_b * T * H + i_h).to(tl.int64) * V + rows[:, None].to(tl.int64) * (H * V) + o_v[None, :]
    o_a = (((i_bh * NT + i_t) * BT + o_t[:, None]) * BT + o_t[None, :]).to(tl.int64)
    o_hs = ((i_bh * NT + i_t) * K + o_k[:, None]).to(tl.int64) * V + o_v[None, :]
    o_u = ((i_bh * NT + i_t) * BT + o_t[:, None]).to(tl.int64) * V + o_v[None, :]

    b_v = tl.load(v + offv, mask=m_tv, other=0.0).to(tl.float32)
    b_do = tl.load(do + offv, mask=m_tv, other=0.0).to(tl.float32)
    b_U = tl.load(U + o_u, mask=m_tv, other=0.0)
    b_dU = tl.load(dU + o_u, mask=m_tv, other=0.0)
    b_h = tl.load(h + o_hs, mask=m_kv, other=0.0)
    b_dS = tl.load(dh + o_hs, mask=m_kv, other=0.0)
    b_M = tl.load(Minv + o_a)
    b_Aqk = tl.load(Aqk + o_a)
    b_Aak = tl.load(Aak + o_a)

    b_q = tl.load(q + offk, mask=m_tk, other=0.0).to(tl.float32)
    b_k = tl.load(k + offk, mask=m_tk, other=0.0).to(tl.float32)
    b_a = tl.load(a + offk, mask=m_tk, other=0.0).to(tl.float32)
    b_b = tl.load(b + offk, mask=m_tk, other=0.0).to(tl.float32)
    b_g = tl.load(g + offk, mask=m_tk, other=0.0).to(tl.float32)
    b_qg = tl.load(Qg + offk, mask=m_tk, other=0.0)
    b_kg = tl.load(Kg + offk, mask=m_tk, other=0.0)
    b_bg = tl.load(Bg + offk, mask=m_tk, other=0.0)
    b_ag = tl.load(Ag + offk, mask=m_tk, other=0.0)

    b_G = tl.cumsum(b_g, axis=0)
    b_gl = tl.sum(b_g, 0)
    m_incl = o_t[:, None] >= o_t[None, :]
    m_strict = o_t[:, None] > o_t[None, :]

    b_dZ = tl.dot(tl.trans(b_M), b_dU, input_precision=PREC)
    b_dAab = tl.where(m_strict, tl.dot(b_dZ, tl.trans(b_U), input_precision=PREC), 0.0)
    b_dAak = tl.where(m_strict, tl.dot(b_dZ, tl.trans(b_v), input_precision=PREC), 0.0)
    b_dAqk = tl.where(m_incl, tl.dot(b_do, tl.trans(b_v), input_precision=PREC), 0.0)
    b_dAqb = tl.where(m_incl, tl.dot(b_do, tl.trans(b_U), input_precision=PREC), 0.0)

    # Gradients wrt the gated operands. The inter-chunk halves go through the
    # state; the intra-chunk halves are accumulated one source step at a time so
    # each decay factor e^{G_i - G_j} is formed from an exact difference rather
    # than as a product of two chunk-wide exponentials.
    b_dQg = tl.dot(b_do, tl.trans(b_h), input_precision=PREC)
    b_dKg = tl.dot(b_v, tl.trans(b_dS), input_precision=PREC)
    b_dBg = tl.dot(b_U, tl.trans(b_dS), input_precision=PREC)
    b_dAg = tl.dot(b_dZ, tl.trans(b_h), input_precision=PREC)

    b_qs = tl.where(m_tk, b_q * scale, 0.0)
    b_dqi = tl.zeros([BT, BK], dtype=tl.float32)
    b_dai = tl.zeros([BT, BK], dtype=tl.float32)
    b_dki = tl.zeros([BT, BK], dtype=tl.float32)
    b_dbi = tl.zeros([BT, BK], dtype=tl.float32)
    b_Ge = b_G - b_g
    for j in tl.static_range(BT):
        m_j = o_t == j
        b_col = m_j[None, :]
        b_Gj = tl.sum(tl.where(m_j[:, None], b_G, 0.0), 0)
        b_wq = _cexp(b_G - b_Gj[None, :])
        b_wa = _cexp(b_Ge - b_Gj[None, :])
        # accumulate into row j of dk / db: sources are all rows i
        b_ck = tl.sum(tl.where(b_col, b_dAqk, 0.0), 1)[:, None]
        b_cb = tl.sum(tl.where(b_col, b_dAqb, 0.0), 1)[:, None]
        b_ak = tl.sum(tl.where(b_col, b_dAak, 0.0), 1)[:, None]
        b_ab = tl.sum(tl.where(b_col, b_dAab, 0.0), 1)[:, None]
        b_dki += tl.where(m_j[:, None],
                          tl.sum(b_ck * b_qs * b_wq + b_ak * b_a * b_wa, 0)[None, :], 0.0)
        b_dbi += tl.where(m_j[:, None],
                          tl.sum(b_cb * b_qs * b_wq + b_ab * b_a * b_wa, 0)[None, :], 0.0)
        # accumulate row j's k / beta into every dq / dalpha row
        b_kj = tl.sum(tl.where(m_j[:, None], b_k, 0.0), 0)[None, :]
        b_bj = tl.sum(tl.where(m_j[:, None], b_b, 0.0), 0)[None, :]
        b_dqi += (b_ck * b_kj + b_cb * b_bj) * b_wq * scale
        b_dai += (b_ak * b_kj + b_ab * b_bj) * b_wa

    b_dv = tl.dot(b_kg, b_dS, input_precision=PREC)
    b_dv += tl.dot(tl.trans(b_Aqk), b_do, input_precision=PREC)
    b_dv += tl.dot(tl.trans(b_Aak), b_dZ, input_precision=PREC)
    tl.store(dv + offv, b_dv.to(dv.dtype.element_ty), mask=m_tv)

    b_dq = b_dQg * _cexp(b_G) * scale + b_dqi
    b_dk = b_dKg * _cexp(b_gl[None, :] - b_G) + b_dki
    b_db = b_dBg * _cexp(b_gl[None, :] - b_G) + b_dbi
    b_da = b_dAg * _cexp(b_Ge) + b_dai
    tl.store(dq + offk, b_dq.to(dq.dtype.element_ty), mask=m_tk)
    tl.store(dk + offk, b_dk.to(dk.dtype.element_ty), mask=m_tk)
    tl.store(db + offk, b_db.to(db.dtype.element_ty), mask=m_tk)
    tl.store(da + offk, b_da.to(da.dtype.element_ty), mask=m_tk)

    # ---- gate gradient -------------------------------------------------
    # Write S_t = diag(e^{G_t}) P_t. Holding G fixed, dL/dS_t = e^{-G_t} R_t
    # with R_t = dL/dP_t, and the recurrence's own gate term e^{g_t} S_{t-1}
    # equals diag(e^{G_t}) P_{t-1}, so the two exponentials cancel exactly:
    #
    #     dgk_t[k] = sum_v R_t[k, v] * P_{t-1}[k, v]
    #
    # P_{t-1} = P_in + sum_{i<t} (Kl_i v_i^T + Bl_i U_i^T) and
    # R_t = e^{gl} dS_out + sum_{j>=t} Qg_j do_j^T + sum_{j>t} ag_j dU_j^T.
    # Expanding the four sums gives a cross term whose every factor is a decay
    # ratio e^{G_j - G_i} with j > i, hence bounded by one -- nothing here
    # multiplies e^{+G} by e^{-G} across the full span of the chunk.
    # [j, i] pairings of the adjoint rows with the state's writes
    b_Gqk = tl.dot(b_do, tl.trans(b_v), input_precision=PREC)
    b_Gqb = tl.dot(b_do, tl.trans(b_U), input_precision=PREC)
    b_Gak = tl.dot(b_dZ, tl.trans(b_v), input_precision=PREC)
    b_Gab = tl.dot(b_dZ, tl.trans(b_U), input_precision=PREC)

    # b_D[j, k] holds sum_{i < t} (adjoint row j paired with write i), carried in
    # the current step's frame: it is rescaled by e^{g_t} each step, so every
    # stored factor is e^{G_t - G_i} with i < t and every read factor is
    # e^{G_j - G_t} with j >= t. Both are decays, bounded by one.
    b_Dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_Da = tl.zeros([BT, BK], dtype=tl.float32)
    b_cross = tl.zeros([BT, BK], dtype=tl.float32)
    for i_s in tl.static_range(BT):
        m_s = o_t == i_s
        b_Gs = tl.sum(tl.where(m_s[:, None], b_G, 0.0), 0)[None, :]
        # advance the carried frame from step i_s - 1 to i_s (no-op when empty)
        b_gs = tl.sum(tl.where(m_s[:, None], b_g, 0.0), 0)[None, :]
        b_Dq = b_Dq * _cexp(b_gs)
        b_Da = b_Da * _cexp(b_gs)
        b_r = tl.sum(tl.where((o_t >= i_s)[:, None],
                             b_qs * b_Dq * _cexp(b_G - b_Gs), 0.0), 0) \
            + tl.sum(tl.where((o_t > i_s)[:, None],
                              b_a * b_Da * _cexp(b_Ge - b_Gs), 0.0), 0)
        b_cross += tl.where(m_s[:, None], b_r[None, :], 0.0)
        b_ck = tl.sum(tl.where(m_s[None, :], b_Gqk, 0.0), 1)
        b_cb = tl.sum(tl.where(m_s[None, :], b_Gqb, 0.0), 1)
        b_ak = tl.sum(tl.where(m_s[None, :], b_Gak, 0.0), 1)
        b_ab = tl.sum(tl.where(m_s[None, :], b_Gab, 0.0), 1)
        b_ki = tl.sum(tl.where(m_s[:, None], b_k, 0.0), 0)[None, :]
        b_bi = tl.sum(tl.where(m_s[:, None], b_b, 0.0), 0)[None, :]
        b_Dq += b_ck[:, None] * b_ki + b_cb[:, None] * b_bi
        b_Da += b_ak[:, None] * b_ki + b_ab[:, None] * b_bi

    # terms meeting the incoming state P_in and the outgoing adjoint e^{gl} dS
    b_sufq = tl.where(m_tk, b_dQg * b_qg, 0.0)       # rows j >= t
    b_sufa = tl.where(m_tk, b_dAg * b_ag, 0.0)       # rows j >  t
    b_pre = tl.where(m_tk, b_dKg * b_kg + b_dBg * b_bg, 0.0)  # writes i < t
    b_ge = tl.where(o_t[:, None] <= o_t[None, :], 1.0, 0.0)
    b_gt = tl.where(o_t[:, None] < o_t[None, :], 1.0, 0.0)
    b_lt = tl.where(o_t[:, None] > o_t[None, :], 1.0, 0.0)
    b_dgk = b_cross
    b_dgk += tl.dot(b_ge, b_sufq, input_precision=PREC)
    b_dgk += tl.dot(b_gt, b_sufa, input_precision=PREC)
    b_dgk += tl.dot(b_lt, b_pre, input_precision=PREC)
    b_dgk += (tl.sum(b_dS * b_h, 1) * _cexp(b_gl))[None, :]
    tl.store(dg + offk, b_dgk.to(dg.dtype.element_ty), mask=m_tk)


def _nx(x):
    return max(triton.next_power_of_2(x), 16)


def _scan_bv(BV, BH):
    """V tile for the serial chunk scans.

    The scans walk the chunks of one (batch, head) in order, so their only
    parallelism is BH times the number of V tiles. Split V further when BH
    alone leaves the device idle. Shape-only, never value-dependent.
    """
    bvs = min(BV, 64)
    while bvs > 16 and (BV // bvs) * BH < 32:
        bvs //= 2
    return bvs


def _fwd(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = triton.cdiv(T, BT)
    BH = B * H
    scale = K ** -0.5
    prec = "ieee"
    BK = _nx(K)
    BV = _nx(V)
    dev = q.device
    f32 = torch.float32

    Qg = torch.empty(B, T, H, K, dtype=f32, device=dev)
    Kg = torch.empty(B, T, H, K, dtype=f32, device=dev)
    Bg = torch.empty(B, T, H, K, dtype=f32, device=dev)
    Ag = torch.empty(B, T, H, K, dtype=f32, device=dev)
    Aqk = torch.empty(BH, NT, BT, BT, dtype=f32, device=dev)
    Aqb = torch.empty(BH, NT, BT, BT, dtype=f32, device=dev)
    Aak = torch.empty(BH, NT, BT, BT, dtype=f32, device=dev)
    Minv = torch.empty(BH, NT, BT, BT, dtype=f32, device=dev)
    W1 = torch.empty(BH, NT, BT, BT, dtype=f32, device=dev)
    Wa = torch.empty(BH, NT, BT, K, dtype=f32, device=dev)

    dplr_prep_kernel[(NT, BH)](
        q, k, alpha, beta, gk, Qg, Kg, Bg, Ag, Aqk, Aqb, Aak, Minv, Wa, W1,
        scale, T, H=H, K=K, BT=BT, BK=BK, PREC=prec, num_warps=4, num_stages=2,
    )

    BVS = _scan_bv(BV, BH)
    NVS = triton.cdiv(V, BVS)
    h = torch.empty(BH, NT, K, V, dtype=f32, device=dev)
    U = torch.empty(BH, NT, BT, V, dtype=f32, device=dev)
    dplr_fwd_scan_kernel[(NVS, BH)](
        v, Kg, Bg, Wa, W1, gk, h, U,
        T, H=H, K=K, V=V, NT=NT, BT=BT, BK=BK, BV=BVS, PREC=prec,
        num_warps=4, num_stages=2,
    )

    o = torch.empty_like(v)
    dplr_fwd_o_kernel[(NT, BH)](
        Qg, v, U, Aqk, Aqb, h, o,
        T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
        num_warps=8, num_stages=2,
    )
    return o, (Qg, Kg, Bg, Ag, Aqk, Aqb, Aak, Minv, Wa, h, U)


def _bwd(do, q, k, v, alpha, beta, gk, saved):
    Qg, Kg, Bg, Ag, Aqk, Aqb, Aak, Minv, Wa, h, U = saved
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = triton.cdiv(T, BT)
    BH = B * H
    scale = K ** -0.5
    prec = "ieee"
    BK = _nx(K)
    BV = _nx(V)
    dev = q.device
    f32 = torch.float32

    dU = torch.empty(BH, NT, BT, V, dtype=f32, device=dev)
    dplr_bwd_du_kernel[(NT, BH)](
        do, Aqb, dU, T, H=H, V=V, BT=BT, BV=BV, PREC=prec,
        num_warps=4, num_stages=2,
    )

    BVS = _scan_bv(BV, BH)
    NVS = triton.cdiv(V, BVS)
    dh = torch.empty(BH, NT, K, V, dtype=f32, device=dev)
    dplr_bwd_scan_kernel[(NVS, BH)](
        do, Qg, Bg, Wa, gk, dU, dh,
        T, H=H, K=K, V=V, NT=NT, BT=BT, BK=BK, BV=BVS, PREC=prec,
        num_warps=8, num_stages=1,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    da = torch.empty_like(alpha)
    db = torch.empty_like(beta)
    dg = torch.empty_like(gk)
    dplr_bwd_chunk_kernel[(NT, BH)](
        q, k, alpha, beta, gk, v, do, Qg, Kg, Bg, Ag, Aqk, Aak, Minv, h, dh, U, dU,
        dq, dk, dv, da, db, dg,
        scale, T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
        num_warps=8, num_stages=2,
    )
    return dq, dk, dv, da, db, dg


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        o, saved = _fwd(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, *saved)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta, gk = ctx.saved_tensors[:6]
        saved = ctx.saved_tensors[6:]
        return _bwd(do.contiguous(), q, k, v, alpha, beta, gk, saved)


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
