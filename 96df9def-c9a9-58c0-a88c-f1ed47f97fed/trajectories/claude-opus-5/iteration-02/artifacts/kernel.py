"""Chunk-parallel DPLR generalized delta rule -- forward and backward in Triton.

Recurrence (per batch/head, state S in R^{K x V}):

    y_t = S_{t-1}^T alpha_t                          (rank-1 read)
    S_t = diag(exp(gk_t)) S_{t-1} + k_t v_t^T + beta_t y_t^T
    o_t = S_t^T (q_t / sqrt(K))

CHUNKWISE FACTORIZATION.  Time is split into chunks of BT steps.  Inside a chunk
let gi be the inclusive cumulative sum of gk along the chunk, ge = gi - gk the
exclusive one, and glast = gi at the chunk tail; both are genuine per-K-channel
vectors, so every exponential below is a [BT, K] tile, not a scalar.  Unrolling
the recurrence over the chunk with

    q_ops = q*scale*exp(gi)   a_ops = alpha*exp(ge)
    k_ops = k*exp(-gi)        b_ops = beta *exp(-gi)

produces the four intra-chunk matrices of the DPLR cascade

    A_qk = tril (q_ops k_ops^T)    A_qb = tril (q_ops b_ops^T)   (diagonal kept)
    A_ak = tril'(a_ops k_ops^T)    A_e  = -tril'(a_ops b_ops^T)  (diagonal dropped)

and the chunk's rank-1 reads Y (row t is y_t) solve the linear system

    (I + A_e) Y = A_ak V + Ag H,      Ag = rows a_ops,  H = chunk-start state.

Writing M = (I + A_e)^{-1}, W = M Ag and U = M (A_ak V):

    Y  = U + W H
    o  = (Qg + A_qb W) H + A_qk V + A_qb U
    H' = exp(glast) . H + Kg^T V + Bg^T Y,   Kg = k exp(glast-gi), Bg alike.

The output is written with the two state-reading terms combined in K-space,
before the contraction with H, because the fresh rank-1 write largely undoes
what the decayed state contributes and the cancellation is far gentler on a
[BT, K] tile than after both have been amplified by the sum over K.

All four matrices are needed; dropping A_qb/A_e collapses this to plain GLA and
dropping the H terms collapses it to a single chunk.

BACKWARD (hand-derived; no autograd anywhere inside):

    dY    = A_qb^T dO + Bg dH'        dH  = Qg^T dO + W^T dY + exp(glast).dH'
    dR    = M^T dY                    dA_e  = -tril'(dR Y^T)
    dA_ak = tril'(dR V^T)             dAg   = dR H^T
    dA_qk = tril (dO V^T)             dA_qb = tril (dO Y^T)      dQg = dO H^T
    dV    = A_qk^T dO + A_ak^T dR + Kg dH'
    dKg   = V dH'^T                   dBg   = Y dH'^T

(the A_e adjoint follows from (I+A_e)Y = R, so dA_e = -(M^T dY) Y^T).  The
A-matrix adjoints are then pushed back onto the gated operands, and the
per-K-channel gate gradient is assembled from the exponent each operand carries:
a suffix sum for the gi-scaled operands, a shifted suffix sum for the ge-scaled
ones, an exclusive prefix sum for the exp(glast - gi) operands, and the
chunk-tail adjoint of glast returned by the reverse carry.

PRECISION.  The chunk-boundary state, its scratch buffer h across chunks, the
reverse adjoint dh, the transition inverse M, W, U, Y and every dot accumulator
are float32 for their whole life; only the raw inputs and the final output carry
the caller's dtype.  The forward carry additionally keeps a float32 residual
alongside the state, so the bits dropped by each chunk's update are fed back on
the next one rather than compounding over a long sequence's many boundaries.

Numerical policy, identical for every shape and dtype: exponentials are base-2;
the intra-chunk matrices and their adjoints are accumulated one source column at
a time, so every exponential is a causal difference exp2(g_i - g_j) with i >= j
and is bounded by 1 for a decaying gate; every dot uses IEEE float32 inputs with
float32 accumulation.
"""

import torch
import triton
import triton.language as tl

RCP_LN2 = tl.constexpr(1.4426950408889634)
DOT_PREC = tl.constexpr("ieee")
BT = 16
BC = 16
BV_SCAN = 16
BV_LOC = 64


@triton.jit
def dplr_prepare(
    q, k, a, b, gk,
    gi_o, qg, kg, ag, bg, Aqk, Aqb, Aak, Ae,
    scale,
    T, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
):
    """Per-chunk per-channel gate cumsum, gated operands, DPLR cascade matrices."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]
    off = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]

    b_gk = tl.load(gk + off, mask=m, other=0.0).to(tl.float32) * RCP_LN2
    b_gi = tl.cumsum(b_gk, axis=0)
    b_ge = b_gi - b_gk
    b_gl = tl.sum(b_gk, axis=0)
    tl.store(gi_o + off, b_gi, mask=m)

    b_q = tl.load(q + off, mask=m, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + off, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + off, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(b + off, mask=m, other=0.0).to(tl.float32)

    dec = tl.math.exp2(b_gl[None, :] - b_gi)
    tl.store(qg + off, (b_q * tl.math.exp2(b_gi)).to(qg.dtype.element_ty), mask=m)
    tl.store(ag + off, (b_a * tl.math.exp2(b_ge)).to(ag.dtype.element_ty), mask=m)
    tl.store(kg + off, (b_k * dec).to(kg.dtype.element_ty), mask=m)
    tl.store(bg + off, (b_b * dec).to(bg.dtype.element_ty), mask=m)

    # The four cascade matrices are formed one source column j at a time.  Entry
    # (i, j) carries exp2(gi_i - gi_j) per channel, and the causal mask keeps
    # only i >= j, where that exponent is already <= 0 for a decaying gate; the
    # column form therefore never builds a growing exponential and needs no
    # clamp or rescaling.  Contracting instead through operands pre-scaled by a
    # shared per-chunk centre would give two factors of order exp2(+-glast/2),
    # which for a chunk that decays by many orders of magnitude are far apart in
    # scale and cancel only after the fact.
    o_i = tl.arange(0, BT)
    o_col = (i_b * T * H + i_h) * BT + o_t * (H * BT)
    for j in range(BT):
        sel = o_i == j
        g_j = tl.sum(tl.where(sel[:, None], b_gi, 0.0), 0)[None, :]
        k_j = tl.sum(tl.where(sel[:, None], b_k, 0.0), 0)[None, :]
        b_j = tl.sum(tl.where(sel[:, None], b_b, 0.0), 0)[None, :]
        # Masked BEFORE the exponential: above the causal diagonal the exponent
        # is positive and would overflow, and inf * 0 is nan rather than 0.
        e_in = tl.math.exp2(tl.where(o_i[:, None] >= j, b_gi - g_j, -1.0e30))
        e_st = tl.math.exp2(tl.where(o_i[:, None] > j, b_ge - g_j, -1.0e30))
        m_in = (o_i >= j).to(tl.float32)
        m_st = (o_i > j).to(tl.float32)
        tl.store(Aqk + o_col + j, tl.sum(b_q * k_j * e_in, 1) * m_in, mask=m_t)
        tl.store(Aqb + o_col + j, tl.sum(b_q * b_j * e_in, 1) * m_in, mask=m_t)
        tl.store(Aak + o_col + j, tl.sum(b_a * k_j * e_st, 1) * m_st, mask=m_t)
        tl.store(Ae + o_col + j, -tl.sum(b_a * b_j * e_st, 1) * m_st, mask=m_t)


@triton.jit
def dplr_inverse(
    Ae, Minv,
    T, H: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
):
    """M = (I + A_e)^{-1}, exact forward substitution, float32 throughout.

    The strictly lower part L of M satisfies L[i] = -A_e[i] - sum_{j<i} A_e[i,j]
    L[j].  Seeding the working matrix with -A_e and folding each already-solved
    row into the rows below performs exactly that triangular solve; it is a
    finite exact solve, not a truncated series.  For BT = 2*BC the two diagonal
    blocks are solved independently and the off-diagonal block follows from
    [N11 0; N21 N22]^{-1} = [M11 0; -M22 N21 M11, M22].
    """
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_c = tl.arange(0, BC)
    base = (i_b * T * H + i_h) * BT
    o_t1 = i_t * BT + o_c
    o_t2 = i_t * BT + BC + o_c
    m1 = (o_t1 < T)[:, None]
    m2 = (o_t2 < T)[:, None]
    p1 = base + o_t1[:, None] * (H * BT) + o_c[None, :]
    p2 = base + o_t2[:, None] * (H * BT) + (BC + o_c)[None, :]
    p3 = base + o_t2[:, None] * (H * BT) + o_c[None, :]
    p4 = base + o_t1[:, None] * (H * BT) + (BC + o_c)[None, :]
    m_st = o_c[:, None] > o_c[None, :]
    eye = (o_c[:, None] == o_c[None, :]).to(tl.float32)

    b_A1 = tl.where(m_st, -tl.load(Ae + p1, mask=m1, other=0.0), 0.0)
    for i in range(1, BC):
        sel = o_c == i
        r1 = tl.sum(tl.where(sel[:, None], b_A1, 0.0), 0)
        r1 = r1 + tl.sum(r1[:, None] * b_A1, 0) * (o_c < i)
        b_A1 = tl.where(sel[:, None], r1, b_A1)
    b_A1 = b_A1 + eye
    tl.store(Minv + p1, b_A1, mask=m1)

    if BC * 2 == BT:
        b_A2 = tl.where(m_st, -tl.load(Ae + p2, mask=m2, other=0.0), 0.0)
        for i in range(1, BC):
            sel = o_c == i
            r2 = tl.sum(tl.where(sel[:, None], b_A2, 0.0), 0)
            r2 = r2 + tl.sum(r2[:, None] * b_A2, 0) * (o_c < i)
            b_A2 = tl.where(sel[:, None], r2, b_A2)
        b_A2 = b_A2 + eye
        b_A21 = tl.load(Ae + p3, mask=m2, other=0.0)
        b_off = -tl.dot(tl.dot(b_A2, b_A21, input_precision=DOT_PREC), b_A1,
                        input_precision=DOT_PREC)
        tl.store(Minv + p2, b_A2, mask=m2)
        tl.store(Minv + p3, b_off, mask=m2)
        tl.store(Minv + p4, tl.zeros([BC, BC], dtype=tl.float32), mask=m1)


@triton.jit
def dplr_wu(
    Minv, Aak, ag, v, w, u,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    """W = M Ag and U = M (A_ak V) -- the chunk-local half of Y."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_i = tl.arange(0, BT)
    m_t = o_t < T
    base = (i_b * T * H + i_h) * BT
    pA = base + o_t[:, None] * (H * BT) + o_i[None, :]
    b_M = tl.load(Minv + pA, mask=m_t[:, None], other=0.0)
    b_Aak = tl.load(Aak + pA, mask=m_t[:, None], other=0.0)

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    pk = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    mk = m_t[:, None] & m_k[None, :]
    b_ag = tl.load(ag + pk, mask=mk, other=0.0).to(tl.float32)
    tl.store(w + pk, tl.dot(b_M, b_ag, input_precision=DOT_PREC), mask=mk)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        mv = m_t[:, None] & (o_v < V)[None, :]
        pv = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_v = tl.load(v + pv, mask=mv, other=0.0).to(tl.float32)
        b_p = tl.dot(b_Aak, b_v, input_precision=DOT_PREC)
        tl.store(u + pv, tl.dot(b_M, b_p, input_precision=DOT_PREC), mask=mv)


@triton.jit
def dplr_scan(
    kg, bg, ag, Aak, Minv, v, gi, h, y,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    """Forward carry across chunk boundaries; the state stays in float32."""
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    hbase = (i_b * (NT + 1) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    c_h = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        # b_h + c_h is the accurate carry; c_h holds the bits that fell off the
        # bottom of the float32 accumulator on the previous chunk's update.  The
        # corrected value is what both this chunk's read and the stored
        # checkpoint use, so the forward output and the backward pass that
        # reloads h see exactly the same state.
        s_h0 = b_h + c_h
        c_h = c_h - (s_h0 - b_h)
        b_h = s_h0
        tl.store(h + hbase + i_t * H * K * V, b_h, mask=m_h)

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        pk = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        pv = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        pAA = (i_b * T * H + i_h) * BT + o_t[:, None] * (H * BT) + tl.arange(0, BT)[None, :]
        mk = m_t[:, None] & m_k[None, :]
        mv = m_t[:, None] & m_v[None, :]
        # Y = M (A_ak V + Ag H), with M applied AFTER the two sources are summed.
        # Pre-multiplying each source separately (as M Ag and M (A_ak V)) is
        # algebraically identical but numerically far worse here: the erasure
        # inverse can be large, so it amplifies both parts well above the size
        # of their sum and the cancellation then happens in the wrong order.
        b_ag = tl.load(ag + pk, mask=mk, other=0.0)
        b_Aak = tl.load(Aak + pAA, mask=m_t[:, None], other=0.0)
        b_v = tl.load(v + pv, mask=mv, other=0.0).to(tl.float32)
        b_M = tl.load(Minv + pAA, mask=m_t[:, None], other=0.0)
        b_r = tl.dot(b_ag, b_h, input_precision=DOT_PREC)
        b_r += tl.dot(b_Aak, b_v, input_precision=DOT_PREC)
        b_y = tl.dot(b_M, b_r, input_precision=DOT_PREC)
        b_y = tl.where(mv, b_y, 0.0)
        tl.store(y + pv, b_y, mask=mv)

        b_kg = tl.load(kg + pk, mask=mk, other=0.0).to(tl.float32)
        b_bg = tl.load(bg + pk, mask=mk, other=0.0).to(tl.float32)
        b_v = tl.load(v + pv, mask=mv, other=0.0).to(tl.float32)
        last = tl.minimum((i_t + 1) * BT, T) - 1
        b_gl = tl.load(gi + (i_b * T * H + i_h) * K + last * (H * K) + o_k,
                       mask=m_k, other=0.0)
        # Compensated carry.  The state is float32, and c_h holds the residual
        # that each chunk's update drops -- the bits below the low end of b_h.
        # Feeding it back on the next chunk keeps the carry accurate across the
        # many boundaries of a long sequence, where a plain float32 accumulation
        # would let one rounding per chunk compound, and also matters within a
        # single chunk whenever the decayed old state and the fresh k v^T + beta
        # y^T contributions are far apart in magnitude and largely cancel.
        e = tl.math.exp2(b_gl[:, None])
        upd = tl.dot(tl.trans(b_kg), b_v, input_precision=DOT_PREC)
        upd += tl.dot(tl.trans(b_bg), b_y, input_precision=DOT_PREC)
        upd += c_h * e
        s_h = b_h * e
        n_h = s_h + upd
        c_h = upd - (n_h - s_h)
        b_h = n_h


@triton.jit
def dplr_output(
    qg, Aqk, Aqb, v, u, w, h, o,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    """o = Qg H + A_qk V + A_qb Y."""
    i_t = tl.program_id(0).to(tl.int64)
    i_v = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    m_k = o_k < K
    m_v = o_v < V
    mv = m_t[:, None] & m_v[None, :]

    ph = (i_b * (NT + 1) * H + i_h) * K * V + i_t * H * K * V \
        + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(h + ph, mask=m_k[:, None] & m_v[None, :], other=0.0)
    pq = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    mkq = m_t[:, None] & m_k[None, :]
    b_qg = tl.load(qg + pq, mask=mkq, other=0.0)

    pA = (i_b * T * H + i_h) * BT + o_t[:, None] * (H * BT) + o_i[None, :]
    pv = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
    b_Aqk = tl.load(Aqk + pA, mask=m_t[:, None], other=0.0)
    b_Aqb = tl.load(Aqb + pA, mask=m_t[:, None], other=0.0)
    b_v = tl.load(v + pv, mask=mv, other=0.0).to(tl.float32)
    b_u = tl.load(u + pv, mask=mv, other=0.0)
    b_w = tl.load(w + pq, mask=mkq, other=0.0)

    # o = (Qg + A_qb W) H + A_qk V + A_qb U, i.e. the two state-reading terms
    # are combined in K-space BEFORE the contraction with H.  Substituting
    # Y = U + W H and contracting Qg H and A_qb (W H) separately is algebraically
    # the same, but those two terms routinely cancel by several digits -- the
    # fresh rank-1 write undoes most of what the decayed state contributes --
    # and doing that cancellation on a [BT, K] tile, at the magnitude of the
    # operands, loses far less than doing it after both have been amplified by
    # the full contraction over K.
    b_qeff = b_qg + tl.dot(b_Aqb, b_w, input_precision=DOT_PREC)
    b_o = tl.dot(b_qeff, b_h, input_precision=DOT_PREC)
    b_o += tl.dot(b_Aqk, b_v, input_precision=DOT_PREC)
    b_o += tl.dot(b_Aqb, b_u, input_precision=DOT_PREC)
    tl.store(o + pv, b_o.to(o.dtype.element_ty), mask=mv)


@triton.jit
def dplr_bwd_scan(
    qg, bg, w, Aqb, do, gi, dh, dy,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    """Reverse carry in float32: dY = A_qb^T dO + Bg dH', dH from dO, dY, decay."""
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_i = tl.arange(0, BT)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    hbase = (i_b * (NT + 1) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for s in range(NT):
        i_t = NT - 1 - s
        tl.store(dh + hbase + (i_t + 1) * H * K * V, b_dh, mask=m_h)

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        mk = m_t[:, None] & m_k[None, :]
        mv = m_t[:, None] & m_v[None, :]
        pk = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        pv = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        pA = (i_b * T * H + i_h) * BT + o_t[:, None] * (H * BT) + o_i[None, :]

        b_do = tl.load(do + pv, mask=mv, other=0.0).to(tl.float32)
        b_Aqb = tl.load(Aqb + pA, mask=m_t[:, None], other=0.0)
        b_bg = tl.load(bg + pk, mask=mk, other=0.0).to(tl.float32)
        b_dyo = tl.dot(tl.trans(b_Aqb), b_do, input_precision=DOT_PREC)
        b_dys = tl.dot(b_bg, b_dh, input_precision=DOT_PREC)
        b_dy = tl.where(mv, b_dyo + b_dys, 0.0)
        tl.store(dy + pv, b_dy, mask=mv)

        b_qg = tl.load(qg + pk, mask=mk, other=0.0)
        b_w = tl.load(w + pk, mask=mk, other=0.0)
        last = tl.minimum((i_t + 1) * BT, T) - 1
        b_gl = tl.load(gi + (i_b * T * H + i_h) * K + last * (H * K) + o_k,
                       mask=m_k, other=0.0)
        # Transpose of the forward output fusion.  dH picks up Qg^T dO from the
        # direct read and W^T (A_qb^T dO) from the same dO routed through the
        # rank-1 path; those two largely cancel, so the effective read operator
        # (Qg + A_qb W) is assembled on the [BT, K] tile first and contracted
        # against dO once, instead of cancelling after the contraction.
        b_qeff = b_qg + tl.dot(b_Aqb, b_w, input_precision=DOT_PREC)
        b_dh = b_dh * tl.math.exp2(b_gl[:, None])
        b_dh += tl.dot(tl.trans(b_qeff), b_do, input_precision=DOT_PREC)
        b_dh += tl.dot(tl.trans(b_w), tl.where(mv, b_dys, 0.0),
                       input_precision=DOT_PREC)
    tl.store(dh + hbase, b_dh, mask=m_h)


@triton.jit
def dplr_bwd_local(
    kg, bg, Minv, Aak, Aqk, v, y, u, w, h, dh, do, dy, gi,
    dv, dqg, dkg, dbg, dag, dAqk, dAqb, dAak, dAe, dgl,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    """Chunk-local adjoints of the four A matrices, the gated operands and dv."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    mk = m_t[:, None] & m_k[None, :]
    pk = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    pA = (i_b * T * H + i_h) * BT + o_t[:, None] * (H * BT) + o_i[None, :]

    b_M = tl.load(Minv + pA, mask=m_t[:, None], other=0.0)
    b_Aak = tl.load(Aak + pA, mask=m_t[:, None], other=0.0)
    b_Aqk = tl.load(Aqk + pA, mask=m_t[:, None], other=0.0)
    b_kg = tl.load(kg + pk, mask=mk, other=0.0).to(tl.float32)
    last = tl.minimum((i_t + 1) * BT, T) - 1
    e_gl = tl.math.exp2(tl.load(gi + (i_b * T * H + i_h) * K + last * (H * K) + o_k,
                                mask=m_k, other=0.0))

    acc_dAqk = tl.zeros([BT, BT], dtype=tl.float32)
    acc_dAqb = tl.zeros([BT, BT], dtype=tl.float32)
    acc_dAak = tl.zeros([BT, BT], dtype=tl.float32)
    acc_dAe = tl.zeros([BT, BT], dtype=tl.float32)
    acc_dqg = tl.zeros([BT, BK], dtype=tl.float32)
    acc_dag = tl.zeros([BT, BK], dtype=tl.float32)
    acc_dkg = tl.zeros([BT, BK], dtype=tl.float32)
    acc_dbg = tl.zeros([BT, BK], dtype=tl.float32)
    acc_gl = tl.zeros([BK], dtype=tl.float32)
    hbase = (i_b * (NT + 1) * H + i_h) * K * V + i_t * H * K * V

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        mv = m_t[:, None] & m_v[None, :]
        m_h = m_k[:, None] & m_v[None, :]
        pv = (i_b * T * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        ph = hbase + o_k[:, None] * V + o_v[None, :]

        b_v = tl.load(v + pv, mask=mv, other=0.0).to(tl.float32)
        b_y = tl.load(y + pv, mask=mv, other=0.0)
        b_do = tl.load(do + pv, mask=mv, other=0.0).to(tl.float32)
        b_dy = tl.load(dy + pv, mask=mv, other=0.0)
        b_u = tl.load(u + pv, mask=mv, other=0.0)
        b_h = tl.load(h + ph, mask=m_h, other=0.0)
        b_dh = tl.load(dh + ph + H * K * V, mask=m_h, other=0.0)

        b_dr = tl.dot(tl.trans(b_M), b_dy, input_precision=DOT_PREC)
        acc_dAqk += tl.dot(b_do, tl.trans(b_v), input_precision=DOT_PREC)
        # dA_qb via U and the W-routed state term rather than via Y.  With
        # o = (Qg + A_qb W) H + A_qk V + A_qb U the same adjoint reads
        # dO U^T + (dO H^T) W^T, which equals dO Y^T exactly but never forms the
        # amplified Y: the two pieces stay at the size of U and of the state
        # read, so the cancellation between them happens at operand magnitude.
        acc_dAqb += tl.dot(b_do, tl.trans(b_u), input_precision=DOT_PREC)
        acc_dAak += tl.dot(b_dr, tl.trans(b_v), input_precision=DOT_PREC)
        acc_dAe -= tl.dot(b_dr, tl.trans(b_y), input_precision=DOT_PREC)
        acc_dqg += tl.dot(b_do, tl.trans(b_h), input_precision=DOT_PREC)
        acc_dag += tl.dot(b_dr, tl.trans(b_h), input_precision=DOT_PREC)
        acc_dkg += tl.dot(b_v, tl.trans(b_dh), input_precision=DOT_PREC)
        acc_dbg += tl.dot(b_y, tl.trans(b_dh), input_precision=DOT_PREC)
        # Adjoint of glast, the per-channel decay carrying H_start to H_end.
        acc_gl += tl.sum(b_dh * b_h, 1) * e_gl

        b_dv = tl.dot(tl.trans(b_Aqk), b_do, input_precision=DOT_PREC)
        b_dv += tl.dot(tl.trans(b_Aak), b_dr, input_precision=DOT_PREC)
        b_dv += tl.dot(b_kg, b_dh, input_precision=DOT_PREC)
        tl.store(dv + pv, b_dv.to(dv.dtype.element_ty), mask=mv)

    m_in = o_i[:, None] >= o_i[None, :]
    m_st = o_i[:, None] > o_i[None, :]
    tl.store(dAqk + pA, tl.where(m_in, acc_dAqk, 0.0), mask=m_t[:, None])
    # acc_dqg is exactly dO H^T, so the W-routed half of dA_qb is a [BT, K] x
    # [K, BT] product of quantities already in registers.
    b_w = tl.load(w + pk, mask=mk, other=0.0)
    acc_dAqb += tl.dot(acc_dqg, tl.trans(b_w), input_precision=DOT_PREC)
    tl.store(dAqb + pA, tl.where(m_in, acc_dAqb, 0.0), mask=m_t[:, None])
    tl.store(dAak + pA, tl.where(m_st, acc_dAak, 0.0), mask=m_t[:, None])
    tl.store(dAe + pA, tl.where(m_st, acc_dAe, 0.0), mask=m_t[:, None])
    tl.store(dqg + pk, acc_dqg, mask=mk)
    tl.store(dag + pk, acc_dag, mask=mk)
    tl.store(dkg + pk, acc_dkg, mask=mk)
    tl.store(dbg + pk, acc_dbg, mask=mk)
    pgl = (i_b * NT + i_t) * H * K + i_h * K + o_k
    tl.store(dgl + pgl, acc_gl, mask=m_k)


@triton.jit
def dplr_bwd_gates(
    q, k, a, b, gk, dqg, dkg, dbg, dag, dAqk, dAqb, dAak, dAe, dgl,
    dq, dk, da, db, dgk,
    scale,
    T, H: tl.constexpr, K: tl.constexpr, NT: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
):
    """A-matrix adjoints -> operand adjoints -> dq, dk, dalpha, dbeta, dgk."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]
    off = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    pA = (i_b * T * H + i_h) * BT + o_t[:, None] * (H * BT) + o_i[None, :]

    b_gk = tl.load(gk + off, mask=m, other=0.0).to(tl.float32) * RCP_LN2
    b_gi = tl.cumsum(b_gk, axis=0)
    b_ge = b_gi - b_gk
    b_gl = tl.sum(b_gk, axis=0)

    b_q = tl.load(q + off, mask=m, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + off, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + off, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(b + off, mask=m, other=0.0).to(tl.float32)

    dec = tl.math.exp2(b_gl[None, :] - b_gi)
    v_qg = b_q * tl.math.exp2(b_gi)
    v_ag = b_a * tl.math.exp2(b_ge)
    v_kg = b_k * dec
    v_bg = b_b * dec

    b_dAqk = tl.load(dAqk + pA, mask=m_t[:, None], other=0.0)
    b_dAqb = tl.load(dAqb + pA, mask=m_t[:, None], other=0.0)
    b_dAak = tl.load(dAak + pA, mask=m_t[:, None], other=0.0)
    b_dAe = tl.load(dAe + pA, mask=m_t[:, None], other=0.0)

    # Intra-chunk adjoints, accumulated one source column j at a time so that
    # every exponential is the causal difference exp2(g_i - g_j) with i >= j and
    # never a separately-scaled pair of large factors.  Alongside the operand
    # adjoints this also accumulates the gate adjoint of the cascade: entry
    # (i, j) carries exp2(gi_i - gi_j), so a unit shift of gk_t moves it by
    # (i >= t) - (j >= t), i.e. it contributes to rows j+1..i only.  That
    # contribution is added as a prefix/suffix pair per column, which keeps
    # every term at the magnitude of the entry it came from; at t = 0, where the
    # true gradient vanishes because the state is still zero, the pair cancels
    # exactly instead of leaving a rounding residue behind.
    d_qc = tl.zeros([BT, BK], dtype=tl.float32)
    d_ac = tl.zeros([BT, BK], dtype=tl.float32)
    d_kc = tl.zeros([BT, BK], dtype=tl.float32)
    d_bc = tl.zeros([BT, BK], dtype=tl.float32)
    A_mat = tl.zeros([BT, BK], dtype=tl.float32)
    for j in range(BT):
        sel = o_i == j
        g_j = tl.sum(tl.where(sel[:, None], b_gi, 0.0), 0)[None, :]
        k_j = tl.sum(tl.where(sel[:, None], b_k, 0.0), 0)[None, :]
        b_j = tl.sum(tl.where(sel[:, None], b_b, 0.0), 0)[None, :]
        # Masked BEFORE the exponential: above the causal diagonal the exponent
        # is positive and would overflow, and inf * 0 is nan rather than 0.
        e_in = tl.math.exp2(tl.where(o_i[:, None] >= j, b_gi - g_j, -1.0e30))
        e_st = tl.math.exp2(tl.where(o_i[:, None] > j, b_ge - g_j, -1.0e30))
        m_in = (o_i >= j).to(tl.float32)[:, None]
        m_st = (o_i > j).to(tl.float32)[:, None]

        c_qk = tl.sum(tl.where(sel[None, :], b_dAqk, 0.0), 1)[:, None] * m_in
        c_qb = tl.sum(tl.where(sel[None, :], b_dAqb, 0.0), 1)[:, None] * m_in
        c_ak = tl.sum(tl.where(sel[None, :], b_dAak, 0.0), 1)[:, None] * m_st
        c_e = tl.sum(tl.where(sel[None, :], b_dAe, 0.0), 1)[:, None] * m_st

        d_qc += (c_qk * k_j + c_qb * b_j) * e_in
        d_ac += (c_ak * k_j - c_e * b_j) * e_st
        col_k = c_qk * b_q * e_in + c_ak * b_a * e_st
        col_b = c_qb * b_q * e_in - c_e * b_a * e_st
        d_kc += tl.sum(col_k, 0)[None, :] * sel[:, None]
        d_bc += tl.sum(col_b, 0)[None, :] * sel[:, None]

        # Gate adjoint of this column.  The query-side entries carry
        # exp2(gi_i - gi_j), so gk_t moves them for j < t <= i; the alpha-side
        # entries carry exp2(ge_i - gi_j) and so move for j < t < i.  Each is
        # therefore the suffix sum of its own contribution, masked to t > j --
        # inclusive for the first, exclusive for the second.
        t_in = (c_qk * b_q * k_j + c_qb * b_q * b_j) * e_in
        t_st = (c_ak * b_a * k_j - c_e * b_a * b_j) * e_st
        s_in = tl.flip(tl.cumsum(tl.flip(t_in, 0), axis=0), 0)
        s_st = tl.flip(tl.cumsum(tl.flip(t_st, 0), axis=0), 0) - t_st
        A_mat += (s_in + s_st) * m_st

    d_qg = tl.load(dqg + off, mask=m, other=0.0)
    d_ag = tl.load(dag + off, mask=m, other=0.0)
    d_kg = tl.load(dkg + off, mask=m, other=0.0)
    d_bg = tl.load(dbg + off, mask=m, other=0.0)

    o_qg = d_qg * v_qg
    o_ag = d_ag * v_ag
    o_kg = d_kg * v_kg
    o_bg = d_bg * v_bg

    # Chunk-wide gate quantities coming back from the reverse carry.
    pgl = (i_b * NT + i_t) * H * K + i_h * K + o_k
    A_const = tl.load(dgl + pgl, mask=m_k, other=0.0)

    # State-coupled part, each term following the exponent its operand carries.
    #   qg ~ exp2(gi)           -> inclusive suffix sum along the chunk
    #   ag ~ exp2(ge)           -> exclusive suffix sum
    #   kg, bg ~ exp2(glast-gi) -> exclusive PREFIX sum, since d(glast - gi_s)
    #                              w.r.t. gk_t is 1 for s < t and 0 otherwise.
    #                              Written directly as a prefix sum; expanding it
    #                              as a chunk total minus a suffix sum would
    #                              difference two quantities the size of the
    #                              carried state that cancel to nothing.
    #   A_const                 -> adjoint of glast from the carry, uniform in t.
    r_qg = tl.flip(tl.cumsum(tl.flip(o_qg, 0), axis=0), 0)
    r_ag = tl.flip(tl.cumsum(tl.flip(o_ag, 0), axis=0), 0) - o_ag
    A_pre = o_kg + o_bg
    p_pre = tl.cumsum(A_pre, axis=0) - A_pre
    b_dgk = A_mat + r_qg + r_ag + p_pre + A_const[None, :]

    tl.store(dq + off, ((d_qg * tl.math.exp2(b_gi) + d_qc) * scale
                        ).to(dq.dtype.element_ty), mask=m)
    tl.store(dk + off, (d_kg * dec + d_kc).to(dk.dtype.element_ty), mask=m)
    tl.store(da + off, (d_ag * tl.math.exp2(b_ge) + d_ac).to(da.dtype.element_ty),
             mask=m)
    tl.store(db + off, (d_bg * dec + d_bc).to(db.dtype.element_ty), mask=m)
    tl.store(dgk + off, b_dgk.to(dgk.dtype.element_ty), mask=m)


def _fwd(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    dev = q.device
    NT = triton.cdiv(T, BT)
    BK = max(triton.next_power_of_2(K), 16)
    scale = K ** -0.5

    gi = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    qg = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    kg = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    ag = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    bg = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    Aqk = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    Aqb = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    Aak = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    Ae = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    dplr_prepare[(NT, B * H)](
        q, k, alpha, beta, gk, gi, qg, kg, ag, bg, Aqk, Aqb, Aak, Ae,
        scale, T, H, K, BT=BT, BK=BK, num_warps=4, num_stages=2)

    Minv = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    dplr_inverse[(NT, B * H)](Ae, Minv, T, H, BT=BT, BC=BC,
                              num_warps=4, num_stages=2)

    w = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    u = torch.empty(B, T, H, V, dtype=torch.float32, device=dev)
    dplr_wu[(NT, B * H)](Minv, Aak, ag, v, w, u, T, H, K, V,
                         BT=BT, BK=BK, BV=64, num_warps=4, num_stages=2)

    h = torch.empty(B, NT + 1, H, K, V, dtype=torch.float32, device=dev)
    y = torch.empty(B, T, H, V, dtype=torch.float32, device=dev)
    dplr_scan[(triton.cdiv(V, BV_SCAN), B * H)](
        kg, bg, ag, Aak, Minv, v, gi, h, y, T, H, K, V, NT,
        BT=BT, BK=BK, BV=BV_SCAN, num_warps=8, num_stages=4)

    o = torch.empty_like(v)
    dplr_output[(NT, triton.cdiv(V, 64), B * H)](
        qg, Aqk, Aqb, v, u, w, h, o, T, H, K, V, NT,
        BT=BT, BK=BK, BV=64, num_warps=4, num_stages=2)
    return o, gi, qg, kg, ag, bg, Aqk, Aqb, Aak, Minv, w, u, h, y


def _bwd(do, q, k, v, alpha, beta, gk,
         gi, qg, kg, ag, bg, Aqk, Aqb, Aak, Minv, w, u, h, y):
    B, T, H, K = q.shape
    V = v.shape[-1]
    dev = q.device
    NT = triton.cdiv(T, BT)
    BK = max(triton.next_power_of_2(K), 16)
    scale = K ** -0.5

    dh = torch.empty(B, NT + 1, H, K, V, dtype=torch.float32, device=dev)
    dy = torch.empty(B, T, H, V, dtype=torch.float32, device=dev)
    dplr_bwd_scan[(triton.cdiv(V, BV_SCAN), B * H)](
        qg, bg, w, Aqb, do, gi, dh, dy, T, H, K, V, NT,
        BT=BT, BK=BK, BV=BV_SCAN, num_warps=8, num_stages=4)

    dv = torch.empty_like(v)
    dqg = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dkg = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dbg = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dag = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dAqk = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    dAqb = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    dAak = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    dAe = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    dgl = torch.empty(B, NT, H, K, dtype=torch.float32, device=dev)
    dplr_bwd_local[(NT, B * H)](
        kg, bg, Minv, Aak, Aqk, v, y, u, w, h, dh, do, dy, gi,
        dv, dqg, dkg, dbg, dag, dAqk, dAqb, dAak, dAe, dgl,
        T, H, K, V, NT, BT=BT, BK=BK, BV=BV_LOC, num_warps=4, num_stages=2)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    da = torch.empty_like(alpha)
    db = torch.empty_like(beta)
    dgk = torch.empty_like(gk)
    dplr_bwd_gates[(NT, B * H)](
        q, k, alpha, beta, gk, dqg, dkg, dbg, dag, dAqk, dAqb, dAak, dAe, dgl,
        dq, dk, da, db, dgk, scale, T, H, K, NT,
        BT=BT, BK=BK, num_warps=4, num_stages=2)
    return dq, dk, dv, da, db, dgk


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta = alpha.contiguous(), beta.contiguous()
        gk = gk.contiguous().float()
        out, gi, qg, kg, ag, bg, Aqk, Aqb, Aak, Minv, w, u, h, y = _fwd(
            q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, gi, qg, kg, ag, bg,
                              Aqk, Aqb, Aak, Minv, w, u, h, y)
        return out

    @staticmethod
    def backward(ctx, do):
        (q, k, v, alpha, beta, gk, gi, qg, kg, ag, bg,
         Aqk, Aqb, Aak, Minv, w, u, h, y) = ctx.saved_tensors
        return _bwd(do.contiguous(), q, k, v, alpha, beta, gk,
                    gi, qg, kg, ag, bg, Aqk, Aqb, Aak, Minv, w, u, h, y)


def chunked_dplr_delta(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]    DPLR erasure basis
    beta: torch.Tensor,  # [B, T, H, K]    DPLR write basis
    gk: torch.Tensor,  # [B, T, H, K]    per-K-channel log-decay (<= 0)
) -> torch.Tensor:  # [B, T, H, V]
    """Entry point: chunk-parallel DPLR generalized delta rule."""
    return _Dplr.apply(q, k, v, alpha, beta, gk)
