"""Chunk-parallel DPLR (Diagonal-Plus-Low-Rank) generalized delta rule.

Forward and backward, implemented entirely with Triton kernels declared in
this module.  Per (batch, head), with state ``S`` of shape [K, V]:

    S_t = diag(exp(gk_t)) . S_{t-1} + k_t v_t^T + beta_t . (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))

Time is cut into chunks of ``BT`` steps.  Within a chunk the per-K-channel
log-decay is accumulated into an inclusive cumsum ``Gi`` and an exclusive
cumsum ``Ge``, and the four intra-chunk interaction matrices of the DPLR
chunkwise algorithm are formed with tensor-core matmuls over the tile:

    A_qk[i,j] = <q_i, k_j> exp(Gi_i - Gi_j)   for i >= j
    A_qb[i,j] = <q_i, b_j> exp(Gi_i - Gi_j)   for i >= j
    A_ak[i,j] = <a_i, k_j> exp(Ge_i - Gi_j)   for i >  j
    A_ab[i,j] = <a_i, b_j> exp(Ge_i - Gi_j)   for i >  j

``A_ab`` is strictly lower triangular, hence nilpotent, and the alpha/beta
erasure cascade inside the chunk is resolved *exactly* by

    M = (I - A_ab)^{-1} = I + A_ab + A_ab^2 + ... + A_ab^(BT-1)

built by repeated squaring of the finite Neumann series -- no truncation and
no approximation.  With ``u = M (A_ak v)``, ``w = M ag`` and
``v2 = u + w S`` a whole chunk collapses to

    o  = A_qk v + A_qb v2 + qg S
    S' = diag(exp(Gi_last)) S + kg^T v + bg^T v2

The chunk-boundary state ``S``, its backward counterpart ``dS``, and every
scratch buffer that carries either across a chunk boundary or between the
forward and the backward pass are float32 for the whole of their life.
"""

import torch
import triton
import triton.language as tl

BT = 16  # time chunk length
LOG_BT = 4  # log2(BT): repeated-squaring steps for the exact WY inverse
EXP_CAP = tl.constexpr(60.0)   # log2-domain exponent clamp; keeps every operand finite
LOG2E = tl.constexpr(1.4426950408889634)


def _nx(n, lo=16, hi=None):
    v = max(triton.next_power_of_2(n), lo)
    return min(v, hi) if hi is not None else v


@triton.jit
def _exp2c(x):
    return tl.exp2(tl.minimum(tl.maximum(x, -EXP_CAP), EXP_CAP))


@triton.jit
def _mid_row(x, BT: tl.constexpr, n):
    """Row n // 2 of x. n is the chunk's VALID length, so a partial trailing
    chunk anchors inside its own data rather than on a padded row."""
    return tl.sum(tl.where((tl.arange(0, BT) == n // 2)[:, None], x, 0.0), 0)


@triton.jit
def _tri_inv(N, BT: tl.constexpr, LOG_BT: tl.constexpr):
    """(I - N)^{-1} for strictly lower triangular (nilpotent) N, exactly."""
    o = tl.arange(0, BT)
    eye = (o[:, None] == o[None, :]).to(tl.float32)
    R = eye + N
    P = N
    for _ in tl.static_range(LOG_BT - 1):
        P = tl.dot(P, P, input_precision="ieee")
        R = tl.dot(R, eye + P, input_precision="ieee")
    return R


@triton.jit
def diag_dplr_gate_wy(
    q, k, a, b, gk, v,
    qg, kg, bg, wq, uq, Aqk, Aqb, Minv, GL,
    T, scale,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    LOG_BT: tl.constexpr, PREC: tl.constexpr,
):
    """Per-chunk gate cumsum, the (A_qk, A_qb, A_ab, A_ak) cascade, and WY."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]
    p = (i_b * T * H + i_h).to(tl.int64) * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]

    b_q = tl.load(q + p, mask=m, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + p, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + p, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(b + p, mask=m, other=0.0).to(tl.float32)
    b_g = tl.load(gk + p, mask=m, other=0.0).to(tl.float32) * LOG2E

    Gi = tl.cumsum(b_g, 0)
    Ge = Gi - b_g
    gl = tl.sum(tl.where((tl.arange(0, BT) == BT - 1)[:, None], Gi, 0.0), 0)
    # Anchor the intra-chunk gate at the MIDDLE step rather than at the chunk
    # boundary.  Every A entry weighs exp(G_i - Gi_j), so the anchor cancels
    # exactly and the result is unchanged; but it halves the largest exponent
    # either operand has to carry, which keeps the products that build the
    # gate adjoint from spanning more decades than float32 can resolve.
    off = _mid_row(Gi, BT, min(T - i_t * BT, BT))

    eq = _exp2c(Gi - off[None, :])
    ek = _exp2c(off[None, :] - Gi)
    ea = _exp2c(Ge - off[None, :])
    Q1 = b_q * eq
    K1 = b_k * ek
    B1 = b_b * ek
    A1 = b_a * ea

    oi = tl.arange(0, BT)
    incl = oi[:, None] >= oi[None, :]
    strict = oi[:, None] > oi[None, :]
    # Build the four intra-chunk matrices one source step j at a time, against
    # the EXACT pairwise gate exp(G_i - Gi_j).  Contracting the commonly
    # anchored operands instead (Q1 @ K1^T) is algebraically the same but lets
    # each operand carry the gate's full dynamic range even where their product
    # is a plain decay, which costs accuracy in the small entries and, through
    # them, in the gate gradient.
    A_qk = tl.zeros([BT, BT], dtype=tl.float32)
    A_qb = tl.zeros([BT, BT], dtype=tl.float32)
    A_ak = tl.zeros([BT, BT], dtype=tl.float32)
    A_ab = tl.zeros([BT, BT], dtype=tl.float32)
    for j in tl.static_range(BT):
        rowj = (oi == j)[:, None]
        b_kj = tl.sum(tl.where(rowj, b_k, 0.0), 0)[None, :]
        b_bj = tl.sum(tl.where(rowj, b_b, 0.0), 0)[None, :]
        g_ij = tl.sum(tl.where(rowj, Gi, 0.0), 0)[None, :]
        w_i = _exp2c(Gi - g_ij)
        w_e = _exp2c(Ge - g_ij)
        colj = (oi == j)[None, :]
        A_qk += colj * tl.sum(b_q * b_kj * w_i, 1)[:, None]
        A_qb += colj * tl.sum(b_q * b_bj * w_i, 1)[:, None]
        A_ak += colj * tl.sum(b_a * b_kj * w_e, 1)[:, None]
        A_ab += colj * tl.sum(b_a * b_bj * w_e, 1)[:, None]
    A_qk = tl.where(incl, A_qk, 0.0)
    A_qb = tl.where(incl, A_qb, 0.0)
    A_ak = tl.where(strict, A_ak, 0.0)
    A_ab = tl.where(strict, A_ab, 0.0)

    M = _tri_inv(A_ab, BT, LOG_BT)

    E_q = _exp2c(Gi)
    E_g = _exp2c(gl[None, :] - Gi)
    E_a = _exp2c(Ge)
    b_ag = b_a * E_a
    b_w = tl.dot(M, b_ag, input_precision=PREC)
    tl.store(qg + p, b_q * E_q, mask=m)
    tl.store(kg + p, b_k * E_g, mask=m)
    tl.store(bg + p, b_b * E_g, mask=m)
    tl.store(wq + p, b_w, mask=m)

    o_a = tl.arange(0, BT)
    pa = (i_b * T * H + i_h).to(tl.int64) * BT + o_t[:, None].to(tl.int64) * (H * BT) + o_a[None, :]
    tl.store(Aqk + pa, A_qk, mask=m_t[:, None])
    tl.store(Aqb + pa, A_qb, mask=m_t[:, None])
    tl.store(Minv + pa, M, mask=m_t[:, None])
    tl.store(GL + (i_bh * NT + i_t) * K + o_k, gl, mask=m_k)

    # u = M (A_ak v), the intra-chunk part of the rank-1 corrected values.
    base_v = (i_b * T * H + i_h).to(tl.int64) * V
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        mv = m_t[:, None] & m_v[None, :]
        pv = base_v + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        b_v = tl.load(v + pv, mask=mv, other=0.0).to(tl.float32)
        b_z = tl.dot(A_ak, b_v, input_precision=PREC)
        b_u = tl.dot(M, b_z, input_precision=PREC)
        tl.store(uq + pv, b_u, mask=mv)


@triton.jit
def diag_dplr_state_scan(
    qg, kg, bg, wq, uq, v, Aqk, Aqb, GL, o, v2, Sbuf,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    """Forward chunk scan: carries the [K, V] state in float32 registers."""
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_s = m_k[:, None] & m_v[None, :]
    o_a = tl.arange(0, BT)

    b_S = tl.zeros([BK, BV], dtype=tl.float32)
    base_k = (i_b * T * H + i_h).to(tl.int64) * K
    base_v = (i_b * T * H + i_h).to(tl.int64) * V
    base_a = (i_b * T * H + i_h).to(tl.int64) * BT

    for i_t in range(NT):
        p_S = Sbuf + (i_bh * NT + i_t).to(tl.int64) * (K * V) + o_k[:, None].to(tl.int64) * V + o_v[None, :]
        tl.store(p_S, b_S, mask=m_s)

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        mk = m_t[:, None] & m_k[None, :]
        mv = m_t[:, None] & m_v[None, :]
        pk = base_k + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        pv = base_v + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        pa = base_a + o_t[:, None].to(tl.int64) * (H * BT) + o_a[None, :]

        b_w = tl.load(wq + pk, mask=mk, other=0.0)
        b_u = tl.load(uq + pv, mask=mv, other=0.0)
        b_v = tl.load(v + pv, mask=mv, other=0.0).to(tl.float32)
        b_qg = tl.load(qg + pk, mask=mk, other=0.0)
        b_kg = tl.load(kg + pk, mask=mk, other=0.0)
        b_bg = tl.load(bg + pk, mask=mk, other=0.0)
        b_Aqk = tl.load(Aqk + pa, mask=m_t[:, None], other=0.0)
        b_Aqb = tl.load(Aqb + pa, mask=m_t[:, None], other=0.0)

        b_v2 = b_u + tl.dot(b_w, b_S, input_precision=PREC)
        b_o = tl.dot(b_Aqk, b_v, input_precision=PREC)
        b_o += tl.dot(b_Aqb, b_v2, input_precision=PREC)
        b_o += tl.dot(b_qg, b_S, input_precision=PREC)
        tl.store(v2 + pv, b_v2, mask=mv)
        tl.store(o + pv, b_o.to(o.dtype.element_ty), mask=mv)

        b_gl = tl.load(GL + (i_bh * NT + i_t).to(tl.int64) * K + o_k, mask=m_k, other=0.0)
        b_S = b_S * _exp2c(b_gl)[:, None]
        b_S += tl.dot(tl.trans(b_kg), b_v, input_precision=PREC)
        b_S += tl.dot(tl.trans(b_bg), b_v2, input_precision=PREC)


@triton.jit
def diag_dplr_dstate_scan(
    qg, kg, bg, wq, do, Aqk, Aqb, GL, dv2, dvp, DH,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    """Reverse chunk scan: carries the adjoint [K, V] state in float32."""
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_s = m_k[:, None] & m_v[None, :]
    o_a = tl.arange(0, BT)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    base_k = (i_b * T * H + i_h).to(tl.int64) * K
    base_v = (i_b * T * H + i_h).to(tl.int64) * V
    base_a = (i_b * T * H + i_h).to(tl.int64) * BT

    for j in range(NT):
        i_t = NT - 1 - j
        p_dh = DH + (i_bh * NT + i_t).to(tl.int64) * (K * V) + o_k[:, None].to(tl.int64) * V + o_v[None, :]
        tl.store(p_dh, b_dh, mask=m_s)

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        mk = m_t[:, None] & m_k[None, :]
        mv = m_t[:, None] & m_v[None, :]
        pk = base_k + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        pv = base_v + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        pa = base_a + o_t[:, None].to(tl.int64) * (H * BT) + o_a[None, :]

        b_do = tl.load(do + pv, mask=mv, other=0.0).to(tl.float32)
        b_Aqk = tl.load(Aqk + pa, mask=m_t[:, None], other=0.0)
        b_Aqb = tl.load(Aqb + pa, mask=m_t[:, None], other=0.0)
        b_kg = tl.load(kg + pk, mask=mk, other=0.0)
        b_bg = tl.load(bg + pk, mask=mk, other=0.0)
        b_qg = tl.load(qg + pk, mask=mk, other=0.0)
        b_w = tl.load(wq + pk, mask=mk, other=0.0)

        b_dv2 = tl.dot(tl.trans(b_Aqb), b_do, input_precision=PREC)
        b_dv2 += tl.dot(b_bg, b_dh, input_precision=PREC)
        b_dvp = tl.dot(tl.trans(b_Aqk), b_do, input_precision=PREC)
        b_dvp += tl.dot(b_kg, b_dh, input_precision=PREC)
        tl.store(dv2 + pv, b_dv2, mask=mv)
        tl.store(dvp + pv, b_dvp, mask=mv)

        b_gl = tl.load(GL + (i_bh * NT + i_t).to(tl.int64) * K + o_k, mask=m_k, other=0.0)
        b_dh = b_dh * _exp2c(b_gl)[:, None]
        b_dh += tl.dot(tl.trans(b_qg), b_do, input_precision=PREC)
        b_dh += tl.dot(tl.trans(b_w), b_dv2, input_precision=PREC)


@triton.jit
def diag_dplr_bwd_chunk(
    q, k, a, b, gk, v, v2, do, dv2, dvp, uq, wq, Minv, Sbuf, DH,
    dq, dk, dv, da, db, dgk,
    T, scale,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    """Per-chunk backward: pulls the adjoints through the DPLR cascade."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]
    o_a = tl.arange(0, BT)
    base_k = (i_b * T * H + i_h).to(tl.int64) * K
    base_v = (i_b * T * H + i_h).to(tl.int64) * V
    p = base_k + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    pa = (i_b * T * H + i_h).to(tl.int64) * BT + o_t[:, None].to(tl.int64) * (H * BT) + o_a[None, :]

    b_q = tl.load(q + p, mask=m, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + p, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + p, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(b + p, mask=m, other=0.0).to(tl.float32)
    b_g = tl.load(gk + p, mask=m, other=0.0).to(tl.float32) * LOG2E

    Gi = tl.cumsum(b_g, 0)
    Ge = Gi - b_g
    gl = tl.sum(tl.where((tl.arange(0, BT) == BT - 1)[:, None], Gi, 0.0), 0)
    # Anchor the intra-chunk gate at the MIDDLE step rather than at the chunk
    # boundary.  Every A entry weighs exp(G_i - Gi_j), so the anchor cancels
    # exactly and the result is unchanged; but it halves the largest exponent
    # either operand has to carry, which keeps the products that build the
    # gate adjoint from spanning more decades than float32 can resolve.
    off = _mid_row(Gi, BT, min(T - i_t * BT, BT))
    eq = _exp2c(Gi - off[None, :])
    ek = _exp2c(off[None, :] - Gi)
    ea = _exp2c(Ge - off[None, :])
    Q1 = b_q * eq
    K1 = b_k * ek
    B1 = b_b * ek
    A1 = b_a * ea

    strict = o_a[:, None] > o_a[None, :]
    incl = o_a[:, None] >= o_a[None, :]
    A_ak = tl.zeros([BT, BT], dtype=tl.float32)
    for j in tl.static_range(BT):
        rowj = (o_a == j)[:, None]
        b_kj = tl.sum(tl.where(rowj, b_k, 0.0), 0)[None, :]
        g_ij = tl.sum(tl.where(rowj, Gi, 0.0), 0)[None, :]
        A_ak += (o_a == j)[None, :] * tl.sum(b_a * b_kj * _exp2c(Ge - g_ij), 1)[:, None]
    A_ak = tl.where(strict, A_ak, 0.0)
    A_akt = tl.trans(A_ak)

    b_M = tl.load(Minv + pa, mask=m_t[:, None], other=0.0)
    b_Mt = tl.trans(b_M)
    b_w = tl.load(wq + p, mask=m, other=0.0).to(tl.float32)

    dqg = tl.zeros([BT, BK], dtype=tl.float64)
    dw = tl.zeros([BT, BK], dtype=tl.float64)
    dkg = tl.zeros([BT, BK], dtype=tl.float64)
    dbg = tl.zeros([BT, BK], dtype=tl.float64)
    dgl_s = tl.zeros([BK], dtype=tl.float64)
    # These four are contracted against v2 and u, which grow with the recurrent
    # state and can span far more decades than the bounded gate adjoint later
    # built from them, so the reductions cancel away most of their own leading
    # digits.  The ACCUMULATORS are widened; every value entering and leaving
    # them is float32, so nothing is ever narrowed.
    dA_qk = tl.zeros([BT, BT], dtype=tl.float64)
    dA_qb = tl.zeros([BT, BT], dtype=tl.float64)
    dA_ak = tl.zeros([BT, BT], dtype=tl.float64)
    dA_ab = tl.zeros([BT, BT], dtype=tl.float64)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        mv = m_t[:, None] & m_v[None, :]
        m_s = m_k[:, None] & m_v[None, :]
        pv = base_v + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        ps = (i_bh * NT + i_t).to(tl.int64) * (K * V) + o_k[:, None].to(tl.int64) * V + o_v[None, :]

        b_S = tl.load(Sbuf + ps, mask=m_s, other=0.0)
        b_dhn = tl.load(DH + ps, mask=m_s, other=0.0)
        b_v = tl.load(v + pv, mask=mv, other=0.0).to(tl.float32)
        b_v2 = tl.load(v2 + pv, mask=mv, other=0.0).to(tl.float32)
        b_do = tl.load(do + pv, mask=mv, other=0.0).to(tl.float32)
        b_dv2 = tl.load(dv2 + pv, mask=mv, other=0.0).to(tl.float32)
        b_u = tl.load(uq + pv, mask=mv, other=0.0).to(tl.float32)
        b_dvp = tl.load(dvp + pv, mask=mv, other=0.0).to(tl.float32)

        b_St = tl.trans(b_S)
        dqg += tl.dot(b_do, b_St, input_precision=PREC).to(tl.float64)
        dw += tl.dot(b_dv2, b_St, input_precision=PREC).to(tl.float64)
        b_dhnt = tl.trans(b_dhn)
        dkg += tl.dot(b_v, b_dhnt, input_precision=PREC).to(tl.float64)
        dbg += tl.dot(b_v2, b_dhnt, input_precision=PREC).to(tl.float64)
        dgl_s += tl.sum(b_S.to(tl.float64) * b_dhn.to(tl.float64), 1)

        dA_qk += tl.dot(b_do, tl.trans(b_v), input_precision=PREC).to(tl.float64)
        dA_qb += tl.dot(b_do, tl.trans(b_v2), input_precision=PREC).to(tl.float64)

        b_dy = tl.dot(b_Mt, b_dv2, input_precision=PREC)
        dA_ab += tl.dot(b_dy, tl.trans(b_u), input_precision=PREC).to(tl.float64)
        dA_ak += tl.dot(b_dy, tl.trans(b_v), input_precision=PREC).to(tl.float64)
        b_dv = b_dvp + tl.dot(A_akt, b_dy, input_precision=PREC)
        tl.store(dv + pv, b_dv.to(dv.dtype.element_ty), mask=mv)

    dA_qk = tl.where(incl, dA_qk, 0.0)
    dA_qb = tl.where(incl, dA_qb, 0.0)
    dA_ak = tl.where(strict, dA_ak, 0.0)
    b_dag = tl.dot(b_Mt, dw.to(tl.float32), input_precision=PREC)
    dA_ab = tl.where(strict, dA_ab + tl.dot(b_dag, tl.trans(b_w), input_precision=PREC).to(tl.float64), 0.0)

    # Pull the A-matrix adjoints back onto q, k, a, b and onto the gate, one
    # source step j at a time, against the EXACT pairwise gate exp(G_i - Gi_j).
    # That factor is a decay, so every summand stays at the scale of its own
    # contribution.
    #
    # The tempting shortcut -- contracting with the already-gated operands, as
    # (dA @ K1) * Q1 -- is algebraically identical but numerically not: with a
    # common anchor, Q1 and K1 each span the gate's full dynamic range even
    # though their product does not, so the sum over j becomes a sum of terms
    # spanning many decades whose total is small, and float32 has no digits
    # left for it.  That is exactly the error that surfaces in dgk.
    #
    # Each pair contributes to the gate with opposite sign at its two
    # endpoints, so this group sums to exactly zero over the chunk -- the
    # property the reverse cumsum below relies on.
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_da = tl.zeros([BT, BK], dtype=tl.float32)
    b_db = tl.zeros([BT, BK], dtype=tl.float32)
    dGi_rel = tl.zeros([BT, BK], dtype=tl.float64)
    dGe_rel = tl.zeros([BT, BK], dtype=tl.float64)
    dGj_rel = tl.zeros([BT, BK], dtype=tl.float64)
    # The dA matrices are contracted against v2, which grows with the recurrent
    # state and can span far more decades than the bounded gate adjoint built
    # from it, so these reductions cancel away most of their leading digits.
    # Only the accumulation widens; every value in and out stays float32.
    for j in tl.static_range(BT):
        rowj = (o_a == j)[:, None]
        colj = (o_a == j)[None, :]
        b_qj = tl.sum(tl.where(rowj, b_q, 0.0), 0)[None, :]
        b_kj = tl.sum(tl.where(rowj, b_k, 0.0), 0)[None, :]
        b_aj = tl.sum(tl.where(rowj, b_a, 0.0), 0)[None, :]
        b_bj = tl.sum(tl.where(rowj, b_b, 0.0), 0)[None, :]
        g_ij = tl.sum(tl.where(rowj, Gi, 0.0), 0)[None, :]
        g_ej = tl.sum(tl.where(rowj, Ge, 0.0), 0)[None, :]

        # column j of each dA: step i reads source step j
        c_qk = tl.sum(tl.where(colj, dA_qk, 0.0), 1)[:, None]
        c_qb = tl.sum(tl.where(colj, dA_qb, 0.0), 1)[:, None]
        c_ak = tl.sum(tl.where(colj, dA_ak, 0.0), 1)[:, None]
        c_ab = tl.sum(tl.where(colj, dA_ab, 0.0), 1)[:, None]
        w_i = tl.where(o_a[:, None] >= j, _exp2c(Gi - g_ij), 0.0).to(tl.float64)
        w_e = tl.where(o_a[:, None] > j, _exp2c(Ge - g_ij), 0.0).to(tl.float64)
        t_i = (c_qk * b_kj + c_qb * b_bj) * b_q * w_i
        t_e = (c_ak * b_kj + c_ab * b_bj) * b_a * w_e
        b_dq += ((c_qk * b_kj + c_qb * b_bj) * w_i).to(tl.float32)
        b_da += ((c_ak * b_kj + c_ab * b_bj) * w_e).to(tl.float32)
        dGi_rel += t_i
        dGe_rel += t_e
        dGj_rel += rowj * tl.sum(t_i + t_e, 0)[None, :]

        # row j of each dA: step j reads source step i, so the roles swap
        r_qk = tl.sum(tl.where(rowj, dA_qk, 0.0), 0)[:, None]
        r_qb = tl.sum(tl.where(rowj, dA_qb, 0.0), 0)[:, None]
        r_ak = tl.sum(tl.where(rowj, dA_ak, 0.0), 0)[:, None]
        r_ab = tl.sum(tl.where(rowj, dA_ab, 0.0), 0)[:, None]
        u_i = tl.where(o_a[:, None] <= j, _exp2c(g_ij - Gi), 0.0).to(tl.float64)
        u_e = tl.where(o_a[:, None] < j, _exp2c(g_ej - Gi), 0.0).to(tl.float64)
        b_dk += (r_qk * b_qj * u_i + r_ak * b_aj * u_e).to(tl.float32)
        b_db += (r_qb * b_qj * u_i + r_ab * b_aj * u_e).to(tl.float32)

    E_q = _exp2c(Gi)
    E_g = _exp2c(gl[None, :] - Gi)
    E_a = _exp2c(Ge)
    b_kg = b_k * E_g
    b_bg = b_b * E_g

    dqg = dqg.to(tl.float32)
    dkg = dkg.to(tl.float32)
    dbg = dbg.to(tl.float32)
    b_dq += dqg * E_q
    b_dk += dkg * E_g
    b_db += dbg * E_g
    b_da += b_dag * E_a
    # "abs" comes from qg = q exp(Gi) and ag = a exp(Ge), which carry an
    # absolute gate reaching every earlier step of the chunk; it does not
    # telescope.
    dGi_abs = dqg * (b_q * E_q)
    dGe_abs = b_dag * (b_a * E_a)

    # kg = k exp(Gi_last - Gi) and bg = b exp(Gi_last - Gi) depend on gk through
    # BOTH the chunk-boundary gate Gi_last and their own -Gi, with opposite
    # sign.  As reverse-cumsum(-dKB) + sum(dKB) the two cancel to a forward
    # EXCLUSIVE cumsum; taking that form directly keeps the cancellation exact,
    # and in particular gives step 0 of the chunk its structural zero.
    dKB = dkg * b_kg + dbg * b_bg
    dgk_kb = tl.cumsum(dKB, 0) - dKB

    # gk[l] reaches step i whenever l <= i, so the adjoint is a reverse cumsum.
    # For the telescoping "rel" group that reverse cumsum equals MINUS the
    # forward exclusive cumsum, and the forward form is the one to use: it makes
    # the zero at step 0 structural instead of a floating-point cancellation of
    # chunk-scale terms.
    D = dGi_rel + dGe_rel - dGj_rel
    b_dgk = ((D - tl.cumsum(D, 0)) - dGe_rel).to(tl.float32) + dgk_kb
    # The chunk-boundary decay exp(gl) reaches every step of the chunk, so it is
    # injected at the LAST row of the reverse cumsum rather than added flat
    # afterwards: algebraically identical, but it then combines with the
    # absolute-gate terms inside one summation tree instead of as a final
    # difference of two chunk-scale numbers.
    Da = dGi_abs + dGe_abs
    Da += (o_a == BT - 1)[:, None] * (dgl_s * _exp2c(gl).to(tl.float64)).to(tl.float32)[None, :]
    b_dgk += tl.cumsum(Da, 0, reverse=True) - dGe_abs

    tl.store(dq + p, (b_dq * scale).to(dq.dtype.element_ty), mask=m)
    tl.store(dk + p, b_dk.to(dk.dtype.element_ty), mask=m)
    tl.store(da + p, b_da.to(da.dtype.element_ty), mask=m)
    tl.store(db + p, b_db.to(db.dtype.element_ty), mask=m)
    tl.store(dgk + p, b_dgk.to(dgk.dtype.element_ty), mask=m)


def _plan(q, v):
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = triton.cdiv(T, BT)
    BK = _nx(K)
    BV = _nx(V, 16, 64)
    prec = "ieee"
    return B, T, H, K, V, NT, BK, BV, triton.cdiv(V, BV), prec


def _fwd(q, k, v, alpha, beta, gk):
    B, T, H, K, V, NT, BK, BV, NV, prec = _plan(q, v)
    dev = q.device
    f32 = torch.float32
    qg = torch.empty(B, T, H, K, dtype=f32, device=dev)
    kg = torch.empty(B, T, H, K, dtype=f32, device=dev)
    bg = torch.empty(B, T, H, K, dtype=f32, device=dev)
    wq = torch.empty(B, T, H, K, dtype=f32, device=dev)
    uq = torch.empty(B, T, H, V, dtype=f32, device=dev)
    Aqk = torch.empty(B, T, H, BT, dtype=f32, device=dev)
    Aqb = torch.empty(B, T, H, BT, dtype=f32, device=dev)
    Minv = torch.empty(B, T, H, BT, dtype=f32, device=dev)
    GL = torch.empty(B * H, NT, K, dtype=f32, device=dev)

    diag_dplr_gate_wy[(NT, B * H)](
        q, k, alpha, beta, gk, v,
        qg, kg, bg, wq, uq, Aqk, Aqb, Minv, GL,
        T, K ** -0.5, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
        LOG_BT=LOG_BT, PREC=prec, num_warps=4, num_stages=2,
    )

    o = torch.empty_like(v)
    v2 = torch.empty(B, T, H, V, dtype=f32, device=dev)
    Sbuf = torch.empty(B * H, NT, K, V, dtype=f32, device=dev)
    diag_dplr_state_scan[(NV, B * H)](
        qg, kg, bg, wq, uq, v, Aqk, Aqb, GL, o, v2, Sbuf,
        T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
        num_warps=4, num_stages=2,
    )
    return o, (qg, kg, bg, wq, uq, Aqk, Aqb, Minv, GL, v2, Sbuf)


def _bwd(do, q, k, v, alpha, beta, gk, cache):
    qg, kg, bg, wq, uq, Aqk, Aqb, Minv, GL, v2, Sbuf = cache
    B, T, H, K, V, NT, BK, BV, NV, prec = _plan(q, v)
    dev = q.device
    f32 = torch.float32

    dv2 = torch.empty(B, T, H, V, dtype=f32, device=dev)
    dvp = torch.empty(B, T, H, V, dtype=f32, device=dev)
    DH = torch.empty(B * H, NT, K, V, dtype=f32, device=dev)
    diag_dplr_dstate_scan[(NV, B * H)](
        qg, kg, bg, wq, do, Aqk, Aqb, GL, dv2, dvp, DH,
        T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
        num_warps=4, num_stages=2,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    da = torch.empty_like(alpha)
    db = torch.empty_like(beta)
    dgk = torch.empty_like(gk)
    diag_dplr_bwd_chunk[(NT, B * H)](
        q, k, alpha, beta, gk, v, v2, do, dv2, dvp, uq, wq, Minv, Sbuf, DH,
        dq, dk, dv, da, db, dgk,
        T, K ** -0.5, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
        num_warps=4, num_stages=2,
    )
    return dq, dk, dv, da, db, dgk


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v, alpha, beta, gk = (
            x.contiguous() for x in (q, k, v, alpha, beta, gk)
        )
        o, cache = _fwd(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, *cache)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta, gk = ctx.saved_tensors[:6]
        cache = ctx.saved_tensors[6:]
        return _bwd(do.contiguous(), q, k, v, alpha, beta, gk, cache)


def chunked_dplr_delta(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]    DPLR erasure basis
    beta: torch.Tensor,  # [B, T, H, K]    DPLR write basis
    gk: torch.Tensor,  # [B, T, H, K]    per-K-channel log-decay (<= 0)
) -> torch.Tensor:  # [B, T, H, V]
    """Entry point. Signature and semantics are fixed; the body is ours."""
    return _Dplr.apply(q, k, v, alpha, beta, gk)
