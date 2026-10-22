"""Chunk-parallel DPLR (diagonal-plus-low-rank) generalized delta rule.

Forward and backward are implemented entirely with Triton kernels declared in
this module.  Per (batch, head) with state S in R^{K x V}:

    S_t = diag(exp(gk_t)) S_{t-1} + k_t v_t^T + beta_t (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))

Chunked formulation used here (chunk length BT, gi = inclusive intra-chunk
cumulative sum of gk, ge = gi - gk, glast = gi at the last row of the chunk):

    A_qk[t,j] = <q_t, k_j> exp(gi_t - gi_j) / sqrt(K)      (t >= j)
    A_qb[t,j] = <q_t, b_j> exp(gi_t - gi_j) / sqrt(K)      (t >= j)
    A_ak[t,j] = <a_t, k_j> exp(ge_t - gi_j)                (t >  j)
    A_ab[t,j] = <a_t, b_j> exp(ge_t - gi_j)                (t >  j)

    U = A_ab U + A_ak V + AG S   ->   U = P (A_ak V) + P AG S,  P = (I-A_ab)^-1
    O = A_qk V + A_qb U + QG S
    S' = diag(exp(glast)) S + KG^T V + BG^T U

with QG = q exp(gi)/sqrt(K), KG = k exp(glast-gi), BG = b exp(glast-gi),
AG = a exp(ge).  P is formed exactly: A_ab is strictly lower triangular, hence
nilpotent, and (I-A_ab)^-1 = prod_i (I + A_ab^(2^i)) terminates exactly.

The recurrent state (h, dh and every accumulator that carries it) is float32
from end to end.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def dot(x, y):
    """float32 matmul at IEEE precision.

    Every matmul in this module runs on float32 operands.  tf32 (and even
    tf32x3) mantissas are not enough: the DPLR recurrence makes the state and
    the (I - A_ab)^-1 cascade ill-conditioned, and the gate gradient is a small
    residue of much larger terms.
    """
    return tl.dot(x, y, input_precision="ieee")


BT = 16  # chunk length
LOG_BT = 4  # log2(BT); number of exact doubling factors for (I - A_ab)^-1
EXP_CAP = tl.constexpr(40.0)  # ceiling on the inverse-gate exponent, keeps exp() finite


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
@triton.jit
def dplr_cumsum_gate_kernel(
    gk, gi, T, H: tl.constexpr, K: tl.constexpr,
    BTc: tl.constexpr, BK: tl.constexpr,
):
    """Per-K-channel inclusive cumulative sum of the log-decay inside a chunk."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    o_t = i_t * BTc + tl.arange(0, BTc)
    o_k = tl.arange(0, BK)
    m = (o_t < T)[:, None] & (o_k < K)[None, :]
    p = (i_b.to(tl.int64) * T * H + i_h) * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    g = tl.load(gk + p, mask=m, other=0.0).to(tl.float32)
    tl.store(gi + p, tl.cumsum(g, axis=0), mask=m)


@triton.jit
def dplr_intra_kernel(
    q, k, a, b, gi, gk, qg, kg, ag, bg, Aqk, Aqb, Aak, Pinv,
    scale, T, H: tl.constexpr, K: tl.constexpr,
    BTc: tl.constexpr, BK: tl.constexpr, NLOG: tl.constexpr,
):
    """Intra-chunk (A_qk, A_qb, A_ab, A_ak) cascade plus the exact (I-A_ab)^-1."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BTc)
    o_t = i_t * BTc + o_i
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]
    base = (i_b.to(tl.int64) * T * H + i_h) * K
    off_r = base + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]

    b_q = tl.load(q + off_r, mask=m, other=0.0).to(tl.float32)
    b_k = tl.load(k + off_r, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + off_r, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(b + off_r, mask=m, other=0.0).to(tl.float32)
    b_gi = tl.load(gi + off_r, mask=m, other=0.0).to(tl.float32)
    b_gk = tl.load(gk + off_r, mask=m, other=0.0).to(tl.float32)
    b_ge = b_gi - b_gk

    # Re-reference the per-channel cumulative gate to the chunk midpoint.  The
    # factored decay exp(gi_t - gi_j) is split into exp(gi_t - c) * exp(c - gi_j)
    # and centering at c halves the exponent range each factor has to span,
    # which is what keeps the two halves from cancelling against each other.
    nval = min(BTc, T - i_t * BTc)
    last = nval - 1
    b_gfirst = tl.sum(tl.where((o_i == 0)[:, None], b_gi, 0.0), 0)
    b_glast = tl.sum(tl.where((o_i == last)[:, None], b_gi, 0.0), 0)
    # gi decreases along the chunk, so its span is [glast, gfirst]; centring on
    # the midpoint of that span minimises the largest exponent either half of
    # the factored decay exp(gi_t - c) * exp(c - gi_i) has to carry
    b_c = 0.5 * (b_gfirst + b_glast)
    b_gic = b_gi - b_c[None, :]

    e_q = tl.exp(tl.minimum(b_gic, EXP_CAP))
    e_k = tl.exp(tl.minimum(-b_gic, EXP_CAP))
    e_a = tl.exp(tl.minimum(b_ge - b_c[None, :], EXP_CAP))

    b_qs = b_q * scale * e_q
    b_ks = b_k * e_k
    b_bs = b_b * e_k
    b_as = b_a * e_a

    b_Aqk = dot(b_qs, tl.trans(b_ks))
    b_Aqb = dot(b_qs, tl.trans(b_bs))
    b_Aak = dot(b_as, tl.trans(b_ks))
    b_Aab = dot(b_as, tl.trans(b_bs))
    m_incl = o_i[:, None] >= o_i[None, :]
    m_strict = o_i[:, None] > o_i[None, :]
    b_Aqk = tl.where(m_incl, b_Aqk, 0.0)
    b_Aqb = tl.where(m_incl, b_Aqb, 0.0)
    b_Aak = tl.where(m_strict, b_Aak, 0.0)
    b_Aab = tl.where(m_strict, b_Aab, 0.0)

    # exact inverse of the unit lower triangular (I - A_ab) by nilpotent doubling:
    # (I-N)^-1 = (I+N)(I+N^2)(I+N^4)...  terminates since N^BT = 0.
    b_eye = (o_i[:, None] == o_i[None, :]).to(tl.float32)
    b_M = b_Aab
    b_P = b_eye + b_Aab
    for _ in tl.static_range(NLOG - 1):
        b_M = dot(b_M, b_M)
        b_P = b_P + dot(b_P, b_M)
    # Newton refinement P <- P (2I - (I - A_ab) P).  The doubling product above
    # is exact in exact arithmetic but accumulates rounding across its factors;
    # this step drives the residual back to the float32 floor and is cheap next
    # to the rest of the chunk.
    b_P = dot(b_P, 2.0 * b_eye - dot(b_eye - b_Aab, b_P))

    # qg/ag multiply the incoming state, so they carry the full exp(gi) / exp(ge)
    e_c = tl.exp(b_c)
    e_gl = tl.exp(tl.minimum(b_glast[None, :] - b_gi, 0.0))
    tl.store(qg + off_r, (b_qs * e_c[None, :]).to(qg.dtype.element_ty), mask=m)
    tl.store(ag + off_r, (b_as * e_c[None, :]).to(ag.dtype.element_ty), mask=m)
    tl.store(kg + off_r, (b_k * e_gl).to(kg.dtype.element_ty), mask=m)
    tl.store(bg + off_r, (b_b * e_gl).to(bg.dtype.element_ty), mask=m)

    o_A = tl.arange(0, BTc)
    base_A = (i_b.to(tl.int64) * T * H + i_h) * BTc
    off_A = base_A + o_t[:, None].to(tl.int64) * (H * BTc) + o_A[None, :]
    m_A = m_t[:, None]
    tl.store(Aqk + off_A, b_Aqk.to(Aqk.dtype.element_ty), mask=m_A)
    tl.store(Aqb + off_A, b_Aqb.to(Aqb.dtype.element_ty), mask=m_A)
    tl.store(Aak + off_A, b_Aak, mask=m_A)
    tl.store(Pinv + off_A, b_P, mask=m_A)


@triton.jit
def dplr_wu_kernel(
    Pinv, ag, Aak, v, w, u, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BTc: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    """W = P AG and Upre = P (A_ak V) -- the WY-style intra-chunk representation."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BTc)
    o_t = i_t * BTc + o_i
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_t = o_t < T
    off_A = (i_b.to(tl.int64) * T * H + i_h) * BTc + o_t[:, None].to(tl.int64) * (H * BTc) + o_i[None, :]
    b_P = tl.load(Pinv + off_A, mask=m_t[:, None], other=0.0)
    b_Aak = tl.load(Aak + off_A, mask=m_t[:, None], other=0.0)

    off_k = (i_b.to(tl.int64) * T * H + i_h) * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    m_kk = m_t[:, None] & (o_k < K)[None, :]
    b_ag = tl.load(ag + off_k, mask=m_kk, other=0.0).to(tl.float32)
    b_w = dot(b_P, b_ag)
    tl.store(w + off_k, b_w.to(w.dtype.element_ty), mask=m_kk)

    off_v = (i_b.to(tl.int64) * T * H + i_h) * V + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
    m_vv = m_t[:, None] & (o_v < V)[None, :]
    b_v = tl.load(v + off_v, mask=m_vv, other=0.0).to(tl.float32)
    b_u = dot(b_P, dot(b_Aak, b_v))
    tl.store(u + off_v, b_u.to(u.dtype.element_ty), mask=m_vv)


@triton.jit
def dplr_fwd_h_kernel(
    w, u, kg, bg, v, gi, h, v_new, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BTc: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr,
):
    """Chunk-boundary state carry.  b_h is float32 for its whole life."""
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BTc)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    bK = (i_b.to(tl.int64) * T * H + i_h) * K
    bV = (i_b.to(tl.int64) * T * H + i_h) * V
    for i_t in range(NT):
        p_h = h + ((i_b.to(tl.int64) * NT + i_t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h, b_h, mask=m_h)
        o_t = i_t * BTc + o_i
        m_t = o_t < T
        off_k = bK + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        off_v = bV + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        off_kT = bK + o_k[:, None] + o_t[None, :].to(tl.int64) * (H * K)
        m_kk = m_t[:, None] & m_k[None, :]
        m_vv = m_t[:, None] & m_v[None, :]
        b_w = tl.load(w + off_k, mask=m_kk, other=0.0).to(tl.float32)
        b_u = tl.load(u + off_v, mask=m_vv, other=0.0).to(tl.float32)
        b_vn = dot(b_w, b_h) + b_u
        tl.store(v_new + off_v, b_vn.to(v_new.dtype.element_ty), mask=m_vv)

        b_kg = tl.load(kg + off_kT, mask=m_k[:, None] & m_t[None, :], other=0.0).to(tl.float32)
        b_bg = tl.load(bg + off_kT, mask=m_k[:, None] & m_t[None, :], other=0.0).to(tl.float32)
        b_v = tl.load(v + off_v, mask=m_vv, other=0.0).to(tl.float32)
        last = min(BTc, T - i_t * BTc) - 1
        b_gl = tl.load(gi + bK + (i_t * BTc + last).to(tl.int64) * (H * K) + o_k, mask=m_k, other=0.0)
        b_h = b_h * tl.exp(b_gl)[:, None]
        b_h += dot(b_kg, b_v)
        b_h += dot(b_bg, b_vn)


@triton.jit
def dplr_fwd_o_kernel(
    qg, Aqk, Aqb, v, v_new, h, o, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BTc: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BTc)
    o_t = i_t * BTc + o_i
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_t = o_t < T
    m_kk = m_t[:, None] & (o_k < K)[None, :]
    m_vv = m_t[:, None] & (o_v < V)[None, :]
    off_A = (i_b.to(tl.int64) * T * H + i_h) * BTc + o_t[:, None].to(tl.int64) * (H * BTc) + o_i[None, :]
    off_k = (i_b.to(tl.int64) * T * H + i_h) * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    off_v = (i_b.to(tl.int64) * T * H + i_h) * V + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]

    b_Aqk = tl.load(Aqk + off_A, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_Aqb = tl.load(Aqb + off_A, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_v = tl.load(v + off_v, mask=m_vv, other=0.0).to(tl.float32)
    b_vn = tl.load(v_new + off_v, mask=m_vv, other=0.0).to(tl.float32)
    b_o = dot(b_Aqk, b_v) + dot(b_Aqb, b_vn)

    p_h = h + ((i_b.to(tl.int64) * NT + i_t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
    b_hs = tl.load(p_h, mask=(o_k < K)[:, None] & (o_v < V)[None, :], other=0.0)
    b_qg = tl.load(qg + off_k, mask=m_kk, other=0.0).to(tl.float32)
    b_o += dot(b_qg, b_hs)
    tl.store(o + off_v, b_o.to(o.dtype.element_ty), mask=m_vv)


# --------------------------------------------------------------------------- #
# backward
# --------------------------------------------------------------------------- #
@triton.jit
def dplr_bwd_dh_kernel(
    qg, bg, w, Aqb, do, gi, h, dht, dglast,
    T, B_: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BTc: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr,
):
    """Reverse chunk-boundary carry of the state gradient, float32 throughout."""
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BTc)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    bK = (i_b.to(tl.int64) * T * H + i_h) * K
    bV = (i_b.to(tl.int64) * T * H + i_h) * V
    bA = (i_b.to(tl.int64) * T * H + i_h) * BTc
    for i_r in range(NT):
        i_t = NT - 1 - i_r
        p_dh = dht + ((i_b.to(tl.int64) * NT + i_t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh, b_dh, mask=m_k[:, None] & m_v[None, :])
        o_t = i_t * BTc + o_i
        m_t = o_t < T
        m_kk = m_t[:, None] & m_k[None, :]
        m_vv = m_t[:, None] & m_v[None, :]
        off_k = bK + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        off_v = bV + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        off_A = bA + o_t[:, None].to(tl.int64) * (H * BTc) + o_i[None, :]

        b_do = tl.load(do + off_v, mask=m_vv, other=0.0).to(tl.float32)
        b_Aqb = tl.load(Aqb + off_A, mask=m_t[:, None], other=0.0).to(tl.float32)
        b_bg = tl.load(bg + off_k, mask=m_kk, other=0.0).to(tl.float32)
        b_w = tl.load(w + off_k, mask=m_kk, other=0.0).to(tl.float32)
        b_qg = tl.load(qg + off_k, mask=m_kk, other=0.0).to(tl.float32)
        # gradient flowing into u (v_new) of this chunk
        b_du = dot(tl.trans(b_Aqb), b_do) + dot(b_bg, b_dh)
        last = min(BTc, T - i_t * BTc) - 1
        b_gl = tl.load(gi + bK + (i_t * BTc + last).to(tl.int64) * (H * K) + o_k, mask=m_k, other=0.0)
        b_dh = b_dh * tl.exp(b_gl)[:, None]
        # d/d(glast) of S_next = diag(exp(glast)) S_t + ...: pair the decayed
        # state gradient with the incoming state h[i_t]
        p_h = h + ((i_b.to(tl.int64) * NT + i_t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        b_hs = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
        p_gl = dglast + (((i_v * B_ + i_b).to(tl.int64) * NT + i_t) * H + i_h) * K + o_k
        tl.store(p_gl, tl.sum(b_hs * b_dh, 1), mask=m_k)
        b_dh += dot(tl.trans(b_qg), b_do)
        b_dh += dot(tl.trans(b_w), b_du)


@triton.jit
def dplr_bwd_du_kernel(
    do, v, v_new, kg, bg, qg, Aqk, Aqb, h, dht,
    dAqk, dAqb, du, dv0, dqg, dkg, dbg, dw,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BTc: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr, NT: tl.constexpr,
):
    """Per-chunk gradients that need the boundary states h and dh."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BTc)
    o_t = i_t * BTc + o_i
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m_kk = m_t[:, None] & m_k[None, :]
    bK = (i_b.to(tl.int64) * T * H + i_h) * K
    bV = (i_b.to(tl.int64) * T * H + i_h) * V
    bA = (i_b.to(tl.int64) * T * H + i_h) * BTc
    off_k = bK + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    off_A = bA + o_t[:, None].to(tl.int64) * (H * BTc) + o_i[None, :]

    b_Aqk = tl.load(Aqk + off_A, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_Aqb = tl.load(Aqb + off_A, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_kg = tl.load(kg + off_k, mask=m_kk, other=0.0).to(tl.float32)
    b_bg = tl.load(bg + off_k, mask=m_kk, other=0.0).to(tl.float32)
    b_qg = tl.load(qg + off_k, mask=m_kk, other=0.0).to(tl.float32)

    b_dAqk = tl.zeros([BTc, BTc], dtype=tl.float32)
    b_dAqb = tl.zeros([BTc, BTc], dtype=tl.float32)
    b_dw = tl.zeros([BTc, BK], dtype=tl.float32)
    b_dqg = tl.zeros([BTc, BK], dtype=tl.float32)
    b_dkg = tl.zeros([BTc, BK], dtype=tl.float32)
    b_dbg = tl.zeros([BTc, BK], dtype=tl.float32)

    for i_v in range(NV):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_vv = m_t[:, None] & m_v[None, :]
        off_v = bV + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        p_h = h + ((i_b.to(tl.int64) * NT + i_t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        p_dh = dht + ((i_b.to(tl.int64) * NT + i_t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        m_hh = m_k[:, None] & m_v[None, :]
        b_hs = tl.load(p_h, mask=m_hh, other=0.0)
        b_dhs = tl.load(p_dh, mask=m_hh, other=0.0)
        b_do = tl.load(do + off_v, mask=m_vv, other=0.0).to(tl.float32)
        b_v = tl.load(v + off_v, mask=m_vv, other=0.0).to(tl.float32)
        b_vn = tl.load(v_new + off_v, mask=m_vv, other=0.0).to(tl.float32)

        b_dAqk += dot(b_do, tl.trans(b_v))
        b_dAqb += dot(b_do, tl.trans(b_vn))
        b_du_t = dot(tl.trans(b_Aqb), b_do) + dot(b_bg, b_dhs)
        b_dv0 = dot(tl.trans(b_Aqk), b_do) + dot(b_kg, b_dhs)
        tl.store(du + off_v, b_du_t, mask=m_vv)
        tl.store(dv0 + off_v, b_dv0, mask=m_vv)

        b_dw += dot(b_du_t, tl.trans(b_hs))
        b_dqg += dot(b_do, tl.trans(b_hs))
        b_dkg += dot(b_v, tl.trans(b_dhs))
        b_dbg += dot(b_vn, tl.trans(b_dhs))

    m_incl = o_i[:, None] >= o_i[None, :]
    tl.store(dAqk + off_A, tl.where(m_incl, b_dAqk, 0.0), mask=m_t[:, None])
    tl.store(dAqb + off_A, tl.where(m_incl, b_dAqb, 0.0), mask=m_t[:, None])
    tl.store(dw + off_k, b_dw, mask=m_kk)
    tl.store(dqg + off_k, b_dqg, mask=m_kk)
    tl.store(dkg + off_k, b_dkg, mask=m_kk)
    tl.store(dbg + off_k, b_dbg, mask=m_kk)


@triton.jit
def dplr_bwd_dA_kernel(
    Pinv, w, u, v, Aak, du, dv0, dw, dAak, dAab, dag, dv,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BTc: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr,
):
    """Backward through the exact (I - A_ab)^-1 solve and the A_ak V product."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BTc)
    o_t = i_t * BTc + o_i
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m_kk = m_t[:, None] & m_k[None, :]
    bK = (i_b.to(tl.int64) * T * H + i_h) * K
    bV = (i_b.to(tl.int64) * T * H + i_h) * V
    bA = (i_b.to(tl.int64) * T * H + i_h) * BTc
    off_k = bK + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    off_A = bA + o_t[:, None].to(tl.int64) * (H * BTc) + o_i[None, :]

    b_P = tl.load(Pinv + off_A, mask=m_t[:, None], other=0.0)
    b_PT = tl.trans(b_P)
    b_Aak = tl.load(Aak + off_A, mask=m_t[:, None], other=0.0)
    b_w = tl.load(w + off_k, mask=m_kk, other=0.0).to(tl.float32)
    b_dw = tl.load(dw + off_k, mask=m_kk, other=0.0)
    b_dag = dot(b_PT, b_dw)
    tl.store(dag + off_k, b_dag, mask=m_kk)

    b_dAab = dot(b_dag, tl.trans(b_w))
    b_dAak = tl.zeros([BTc, BTc], dtype=tl.float32)
    for i_v in range(NV):
        o_v = i_v * BV + tl.arange(0, BV)
        m_vv = m_t[:, None] & (o_v < V)[None, :]
        off_v = bV + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        b_du = tl.load(du + off_v, mask=m_vv, other=0.0)
        b_u = tl.load(u + off_v, mask=m_vv, other=0.0).to(tl.float32)
        b_v = tl.load(v + off_v, mask=m_vv, other=0.0).to(tl.float32)
        b_dv0 = tl.load(dv0 + off_v, mask=m_vv, other=0.0)
        b_dx = dot(b_PT, b_du)
        b_dAab += dot(b_dx, tl.trans(b_u))
        b_dAak += dot(b_dx, tl.trans(b_v))
        b_dvv = b_dv0 + dot(tl.trans(b_Aak), b_dx)
        tl.store(dv + off_v, b_dvv.to(dv.dtype.element_ty), mask=m_vv)

    m_strict = o_i[:, None] > o_i[None, :]
    tl.store(dAab + off_A, tl.where(m_strict, b_dAab, 0.0), mask=m_t[:, None])
    tl.store(dAak + off_A, tl.where(m_strict, b_dAak, 0.0), mask=m_t[:, None])


@triton.jit
def dplr_bwd_gate_kernel(
    q, k, a, b, gi, gk, dAqk, dAqb, dAak, dAab, dqg, dkg, dbg, dag, dglast,
    dq, dk, da, db, dgk,
    scale, T, H: tl.constexpr, K: tl.constexpr,
    BTc: tl.constexpr, BK: tl.constexpr, NT: tl.constexpr,
):
    """Fold the A-cascade and boundary gradients back onto q, k, alpha, beta, gk."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BTc)
    o_t = i_t * BTc + o_i
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]
    bK = (i_b.to(tl.int64) * T * H + i_h) * K
    bA = (i_b.to(tl.int64) * T * H + i_h) * BTc
    off_k = bK + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    off_A = bA + o_t[:, None].to(tl.int64) * (H * BTc) + o_i[None, :]

    b_q = tl.load(q + off_k, mask=m, other=0.0).to(tl.float32)
    b_k = tl.load(k + off_k, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + off_k, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(b + off_k, mask=m, other=0.0).to(tl.float32)
    b_gi = tl.load(gi + off_k, mask=m, other=0.0)
    b_gkv = tl.load(gk + off_k, mask=m, other=0.0).to(tl.float32)
    b_ge = b_gi - b_gkv

    nval = min(BTc, T - i_t * BTc)
    last = nval - 1
    b_gfirst = tl.sum(tl.where((o_i == 0)[:, None], b_gi, 0.0), 0)
    b_glast = tl.sum(tl.where((o_i == last)[:, None], b_gi, 0.0), 0)
    # gi decreases along the chunk, so its span is [glast, gfirst]; centring on
    # the midpoint of that span minimises the largest exponent either half of
    # the factored decay exp(gi_t - c) * exp(c - gi_i) has to carry
    b_c = 0.5 * (b_gfirst + b_glast)
    b_gic = b_gi - b_c[None, :]
    e_q = tl.exp(tl.minimum(b_gic, EXP_CAP))
    e_k = tl.exp(tl.minimum(-b_gic, EXP_CAP))
    e_a = tl.exp(tl.minimum(b_ge - b_c[None, :], EXP_CAP))
    e_gi = tl.exp(b_gi)
    e_ge = tl.exp(b_ge)
    e_gl = tl.exp(tl.minimum(b_glast[None, :] - b_gi, 0.0))

    b_qs = b_q * scale * e_q
    b_ks = b_k * e_k
    b_bs = b_b * e_k
    b_as = b_a * e_a

    b_dAqk = tl.load(dAqk + off_A, mask=m_t[:, None], other=0.0)
    b_dAqb = tl.load(dAqb + off_A, mask=m_t[:, None], other=0.0)
    b_dAak = tl.load(dAak + off_A, mask=m_t[:, None], other=0.0)
    b_dAab = tl.load(dAab + off_A, mask=m_t[:, None], other=0.0)

    b_dQGc = dot(b_dAqk, b_ks) + dot(b_dAqb, b_bs)
    b_dKGc = dot(tl.trans(b_dAqk), b_qs) + dot(tl.trans(b_dAak), b_as)
    b_dBGc = dot(tl.trans(b_dAqb), b_qs) + dot(tl.trans(b_dAab), b_as)
    b_dAGc = dot(b_dAak, b_ks) + dot(b_dAab, b_bs)

    b_dqg = tl.load(dqg + off_k, mask=m, other=0.0)
    b_dkg = tl.load(dkg + off_k, mask=m, other=0.0)
    b_dbg = tl.load(dbg + off_k, mask=m, other=0.0)
    b_dag = tl.load(dag + off_k, mask=m, other=0.0)

    # the state-facing terms use the exact exp(gi) / exp(ge) rather than the
    # re-centred product e_q * exp(c), which would round the two factors apart
    b_qgp = b_dqg * e_gi
    b_agp = b_dag * e_ge
    b_kgp = b_dkg * e_gl
    b_bgp = b_dbg * e_gl
    b_dq = scale * (b_dQGc * e_q + b_qgp)
    b_dk = b_dKGc * e_k + b_kgp
    b_db = b_dBGc * e_k + b_bgp
    b_da = b_dAGc * e_a + b_agp

    # ------------------------------------------------------------------ #
    # gate gradient
    #
    # A[t,i] carries the factor exp(gi_t - gi_i), so gk_j enters A[t,i] with
    # weight ([j <= t] - [j <= i]), which is 1 exactly on the band i < j <= t.
    # Summing the row and the column contributions separately and differencing
    # them -- the reverse-cumsum-of-dgi form -- makes those two indicator halves
    # cancel: at j = 0 they cancel identically, and the DPLR recurrence makes
    # the individual halves many orders of magnitude larger than the surviving
    # difference, so fp32 keeps none of it.  Accumulating the band directly
    # never forms that difference.
    #
    # For split j the band contributes
    #     sum_{t >= j} sum_{i < j} dA[t,i] * A[t,i],
    # and A[t,i] factors as (row_t exp(gi_t - gi_j)) . (col_i exp(gi_j - gi_i)).
    # Carrying the i < j prefix RELATIVE TO gi_j turns its extension into a
    # multiply by exp(gk_j) <= 1, and the row factor exp(gi_t - gi_j) <= 1 for
    # t >= j, so every quantity in this loop is bounded by the raw operands.
    # ------------------------------------------------------------------ #
    # The k-column and b-column prefixes are kept apart: the two carry different
    # magnitudes and summing them before the row contraction loses the smaller.
    b_acc_qk = tl.zeros([BTc, BK], dtype=tl.float32)
    b_acc_qb = tl.zeros([BTc, BK], dtype=tl.float32)
    b_acc_ak = tl.zeros([BTc, BK], dtype=tl.float32)
    b_acc_ab = tl.zeros([BTc, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BTc, BK], dtype=tl.float32)
    b_qn = b_q * scale

    for j in tl.static_range(BTc):
        m_row = o_i == j
        g_j = tl.sum(tl.where(m_row[:, None], b_gi, 0.0), 0)[None, :]
        gk_j = tl.sum(tl.where(m_row[:, None], b_gkv, 0.0), 0)[None, :]
        # rescale the running prefix from reference gi_{j-1} to gi_j
        d_j = tl.exp(gk_j)
        b_acc_qk = b_acc_qk * d_j
        b_acc_qb = b_acc_qb * d_j
        b_acc_ak = b_acc_ak * d_j
        b_acc_ab = b_acc_ab * d_j

        # row factors, referenced to gi_j: <= 1 on the rows that contribute
        r_q = b_qn * tl.exp(tl.minimum(b_gi - g_j, 0.0)) * (o_i >= j).to(tl.float32)[:, None]
        r_a = b_a * tl.exp(tl.minimum(b_ge - g_j, 0.0)) * (o_i > j).to(tl.float32)[:, None]
        band = tl.sum(r_q * b_acc_qk, 0) + tl.sum(r_q * b_acc_qb, 0) \
            + tl.sum(r_a * b_acc_ak, 0) + tl.sum(r_a * b_acc_ab, 0)
        b_dgk += tl.where(m_row[:, None], band[None, :], 0.0)

        # extend the i < j prefix by column j (its own reference is gi_j, so the
        # new column enters with weight one)
        c_qk = tl.sum(tl.where(m_row[None, :], b_dAqk, 0.0), 1)[:, None]
        c_qb = tl.sum(tl.where(m_row[None, :], b_dAqb, 0.0), 1)[:, None]
        c_ak = tl.sum(tl.where(m_row[None, :], b_dAak, 0.0), 1)[:, None]
        c_ab = tl.sum(tl.where(m_row[None, :], b_dAab, 0.0), 1)[:, None]
        r_k = tl.sum(tl.where(m_row[:, None], b_k, 0.0), 0)[None, :]
        r_b = tl.sum(tl.where(m_row[:, None], b_b, 0.0), 0)[None, :]
        b_acc_qk += c_qk * r_k
        b_acc_qb += c_qb * r_b
        b_acc_ak += c_ak * r_k
        b_acc_ab += c_ab * r_b

    # the state-facing gated operands depend on gk through a single exponent
    # each, so these fold in as plain prefix / suffix sums with no cancellation
    m_ge = (o_i[:, None] <= o_i[None, :]).to(tl.float32)  # [j, t] = (t >= j)
    m_gt = (o_i[:, None] < o_i[None, :]).to(tl.float32)   # [j, t] = (t >  j)
    m_lt = (o_i[:, None] > o_i[None, :]).to(tl.float32)   # [j, t] = (t <  j)
    b_dgk += dot(m_ge, tl.where(m, b_q * scale * b_qgp, 0.0))
    b_dgk += dot(m_gt, tl.where(m, b_a * b_agp, 0.0))
    b_dgk += dot(m_lt, tl.where(m, b_k * b_kgp + b_b * b_bgp, 0.0))

    # exp(glast) decays the outgoing state and contains every gk_j in the chunk.
    # KG/BG carry exp(glast - gi_i), whose dependence on gk_j is the band (i < j)
    # already accumulated above -- the glast half cancels there and is not
    # re-added here.
    b_dgl = tl.load(dglast + ((i_b.to(tl.int64) * NT + i_t) * H + i_h) * K + o_k, mask=m_k, other=0.0)
    b_dgk += b_dgl[None, :]
    b_dgk = tl.where(m, b_dgk, 0.0)

    tl.store(dq + off_k, b_dq.to(dq.dtype.element_ty), mask=m)
    tl.store(dk + off_k, b_dk.to(dk.dtype.element_ty), mask=m)
    tl.store(da + off_k, b_da.to(da.dtype.element_ty), mask=m)
    tl.store(db + off_k, b_db.to(db.dtype.element_ty), mask=m)
    tl.store(dgk + off_k, b_dgk.to(dgk.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------- #
# python drivers
# --------------------------------------------------------------------------- #
def _cfg(K, V):
    BK = max(triton.next_power_of_2(K), 16)
    BV = max(triton.next_power_of_2(V), 16)
    return BK, BV


def _fwd(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    # intra-chunk operands feed the state carry; the recurrence amplifies them,
    # so they are kept at float32 rather than the input dtype
    dt = torch.float32
    BK, BV = _cfg(K, V)
    NT = triton.cdiv(T, BT)
    scale = K ** -0.5
    dev = q.device

    gi = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dplr_cumsum_gate_kernel[(NT, B * H)](gk, gi, T, H, K, BTc=BT, BK=BK, num_warps=4)

    qg = torch.empty(B, T, H, K, dtype=dt, device=dev)
    kg = torch.empty(B, T, H, K, dtype=dt, device=dev)
    ag = torch.empty(B, T, H, K, dtype=dt, device=dev)
    bg = torch.empty(B, T, H, K, dtype=dt, device=dev)
    Aqk = torch.empty(B, T, H, BT, dtype=dt, device=dev)
    Aqb = torch.empty(B, T, H, BT, dtype=dt, device=dev)
    Aak = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    Pinv = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    dplr_intra_kernel[(NT, B * H)](
        q, k, alpha, beta, gi, gk, qg, kg, ag, bg, Aqk, Aqb, Aak, Pinv,
        scale, T, H, K, BTc=BT, BK=BK, NLOG=LOG_BT, num_warps=4, num_stages=2,
    )

    w = torch.empty(B, T, H, K, dtype=dt, device=dev)
    u = torch.empty(B, T, H, V, dtype=dt, device=dev)
    dplr_wu_kernel[(NT, B * H)](
        Pinv, ag, Aak, v, w, u, T, H, K, V, BTc=BT, BK=BK, BV=BV, num_warps=4, num_stages=2,
    )

    h = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=dev)
    v_new = torch.empty(B, T, H, V, dtype=dt, device=dev)
    BVs = min(BV, 16)
    NVs = triton.cdiv(V, BVs)
    dplr_fwd_h_kernel[(NVs, B * H)](
        w, u, kg, bg, v, gi, h, v_new, T, H, K, V,
        BTc=BT, BK=BK, BV=BVs, NT=NT, num_warps=8, num_stages=3,
    )

    out_dt = q.dtype  # the entry point returns the input dtype
    o = torch.empty(B, T, H, V, dtype=out_dt, device=dev)
    dplr_fwd_o_kernel[(NT, B * H)](
        qg, Aqk, Aqb, v, v_new, h, o, T, H, K, V,
        BTc=BT, BK=BK, BV=BV, NT=NT, num_warps=8, num_stages=2,
    )
    return o, gi, qg, kg, ag, bg, Aqk, Aqb, Aak, Pinv, w, u, h, v_new


def _bwd(do, q, k, v, alpha, beta, gkt, gi, qg, kg, ag, bg, Aqk, Aqb, Aak, Pinv, w, u, h, v_new):
    B, T, H, K = q.shape
    V = v.shape[-1]
    dt = q.dtype
    gdt = gkt.dtype
    BK, BV = _cfg(K, V)
    NT = triton.cdiv(T, BT)
    scale = K ** -0.5
    dev = q.device
    BVb = min(BV, 16)
    NV = triton.cdiv(V, BVb)

    dht = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=dev)
    BVs = min(BV, 16)
    NVs = triton.cdiv(V, BVs)
    dglast_p = torch.empty(NVs * B, NT, H, K, dtype=torch.float32, device=dev)
    dplr_bwd_dh_kernel[(NVs, B * H)](
        qg, bg, w, Aqb, do, gi, h, dht, dglast_p, T, B, H, K, V,
        BTc=BT, BK=BK, BV=BVs, NT=NT, num_warps=8, num_stages=3,
    )
    dglast = dglast_p.view(NVs, B, NT, H, K).sum(0)

    dAqk = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    dAqb = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    dAak = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    dAab = torch.empty(B, T, H, BT, dtype=torch.float32, device=dev)
    du = torch.empty(B, T, H, V, dtype=torch.float32, device=dev)
    dv0 = torch.empty(B, T, H, V, dtype=torch.float32, device=dev)
    dqg = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dkg = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dbg = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dag = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dw = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dplr_bwd_du_kernel[(NT, B * H)](
        do, v, v_new, kg, bg, qg, Aqk, Aqb, h, dht,
        dAqk, dAqb, du, dv0, dqg, dkg, dbg, dw,
        T, H, K, V, BTc=BT, BK=BK, BV=BVb, NV=NV, NT=NT, num_warps=4, num_stages=2,
    )

    dv = torch.empty(B, T, H, V, dtype=dt, device=dev)
    dplr_bwd_dA_kernel[(NT, B * H)](
        Pinv, w, u, v, Aak, du, dv0, dw, dAak, dAab, dag, dv,
        T, H, K, V, BTc=BT, BK=BK, BV=BVb, NV=NV, num_warps=4, num_stages=2,
    )

    dq = torch.empty(B, T, H, K, dtype=dt, device=dev)
    dk = torch.empty(B, T, H, K, dtype=dt, device=dev)
    da = torch.empty(B, T, H, K, dtype=dt, device=dev)
    dbeta = torch.empty(B, T, H, K, dtype=dt, device=dev)
    dgk = torch.empty(B, T, H, K, dtype=gdt, device=dev)
    dplr_bwd_gate_kernel[(NT, B * H)](
        q, k, alpha, beta, gi, gkt, dAqk, dAqb, dAak, dAab, dqg, dkg, dbg, dag, dglast,
        dq, dk, da, dbeta, dgk, scale, T, H, K, BTc=BT, BK=BK, NT=NT, num_warps=4, num_stages=2,
    )
    return dq, dk, dv, da, dbeta, dgk


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v, alpha, beta, gk = (x.contiguous() for x in (q, k, v, alpha, beta, gk))
        out, gi, qg, kg, ag, bg, Aqk, Aqb, Aak, Pinv, w, u, h, v_new = _fwd(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, gi, qg, kg, ag, bg,
                              Aqk, Aqb, Aak, Pinv, w, u, h, v_new)
        return out

    @staticmethod
    def backward(ctx, do):
        (q, k, v, alpha, beta, gk, gi, qg, kg, ag, bg,
         Aqk, Aqb, Aak, Pinv, w, u, h, v_new) = ctx.saved_tensors
        return _bwd(do.contiguous(), q, k, v, alpha, beta, gk, gi, qg, kg, ag, bg,
                    Aqk, Aqb, Aak, Pinv, w, u, h, v_new)


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
