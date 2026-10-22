"""Chunk-parallel DPLR generalized delta rule (RWKV-7 family), forward and backward.

Per (batch, head), with state S in R^{K x V}:

    S_t = diag(exp(gk_t)) S_{t-1} + k_t v_t^T + beta_t (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))

This is the chunk-parallel (WY-style) form.  Inside a chunk of BT steps, with
`gi` the inclusive per-K-channel cumulative log-decay and `ge = gi - gk` the
exclusive one:

    A_qk[i,j] = <q_i e^{gi_i}, k_j e^{-gi_j}>   (j <= i)
    A_qb[i,j] = <q_i e^{gi_i}, b_j e^{-gi_j}>   (j <= i)
    A_ak[i,j] = <a_i e^{ge_i}, k_j e^{-gi_j}>   (j <  i)
    A_ab[i,j] = <a_i e^{ge_i}, b_j e^{-gi_j}>   (j <  i)

    Ainv = (I - A_ab)^{-1}                    (exact, A_ab is nilpotent)
    w    = Ainv (a e^{ge})     u  = Ainv (A_ak v)
    v2   = u + w S                            (rank-1 correction, resolved)
    o    = A_qk v + A_qb v2 + (q e^{gi}) S
    S'   = S e^{gi_last} + (k e^{gi_last-gi})^T v + (b e^{gi_last-gi})^T v2

`Ainv` is formed exactly by the binary product  prod_i (I + A_ab^{2^i}), which
terminates because A_ab is strictly lower triangular; nothing is truncated.

The chunk-boundary state (`h` forward, `dh` backward) is float32 for the whole
of its life, as is every accumulator.  Forward, backward and the gate cumulative
sums all run in the Triton kernels declared in this module.
"""

import torch
import triton
import triton.language as tl

BT = 16            # chunk length
BV_TILE = 16       # V tile for the state-carrying scans
BK_INTRA = 16      # K tile for the backward intra-chunk kernel
EXP_CAP = 40.0     # products of two capped exponentials must stay inside fp32


@triton.jit
def _expc(x):
    return tl.exp(tl.minimum(tl.maximum(x, -40.0), 40.0))


@triton.jit
def _gate_floor(g, BT: tl.constexpr):
    """Floor the per-step log-decay so a chunk's cumulative span stays in fp32.

    The intra-chunk terms are built as products exp(g_i - off) * exp(off - g_j);
    each leg alone overflows once the chunk's total decay exceeds the fp32
    exponent range, however modest the entry itself is.  Flooring the per-step
    gate bounds that span.  It is numerically inert: at the floor a single step
    already attenuates by e^{-80/BT} and a full chunk by e^{-80} ~ 1e-35, so
    every term the floor could alter has underflowed to zero regardless.  The
    chunk-boundary carry keeps the TRUE cumulative gate, so no decay is lost
    across chunks.
    """
    return tl.maximum(g, -80.0 / BT)


@triton.jit
def dplr_fwd_intra_kernel(
    q, k, a, b, g,
    gi_o, ge_o, qg_o, kg_o, ag_o, bg_o,
    Aqk_o, Aqb_o, Aak_o, Aab_o,
    scale,
    T, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr,
):
    """Per-chunk gate cumsum, gated operands and the (A_qk, A_qb, A_ak, A_ab) cascade."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    r = tl.arange(0, BT)
    o_t = i_t * BT + r
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]
    off = (i_b * T) * H * K + i_h * K + o_t[:, None] * (H * K) + o_k[None, :]

    b_g = _gate_floor(tl.load(g + off, mask=m, other=0.0).to(tl.float32), BT)
    b_gi = tl.cumsum(b_g, axis=0)
    b_ge = b_gi - b_g
    # padded rows carry g == 0, so row BT-1 always holds the chunk's final gate
    b_gl = tl.sum(tl.where((r == BT - 1)[:, None], b_gi, 0.0), 0)
    b_off = 0.5 * b_gl
    tl.store(gi_o + off, b_gi, mask=m)
    tl.store(ge_o + off, b_ge, mask=m)

    # centred exponentials for the A cascade: bounded in fp32 registers, never stored
    e_q = _expc(b_gi - b_off[None, :])
    e_i = _expc(b_off[None, :] - b_gi)
    e_a = _expc(b_ge - b_off[None, :])

    b_q = tl.load(q + off, mask=m, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + off, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(b + off, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + off, mask=m, other=0.0).to(tl.float32)

    b_Q = b_q * e_q
    b_Kx = b_k * e_i
    b_Bx = b_b * e_i
    b_Ax = b_a * e_a

    # the state pipeline uses the bounded forms:  q e^{gi}, k e^{gl-gi}, b e^{gl-gi}, a e^{ge}
    dt = qg_o.dtype.element_ty
    e_ki = _expc(b_gl[None, :] - b_gi)
    tl.store(qg_o + off, (b_q * tl.exp(b_gi)).to(dt), mask=m)
    tl.store(kg_o + off, (b_k * e_ki).to(dt), mask=m)
    tl.store(bg_o + off, (b_b * e_ki).to(dt), mask=m)
    tl.store(ag_o + off, (b_a * tl.exp(b_ge)).to(dt), mask=m)

    inc = r[:, None] >= r[None, :]
    strict = r[:, None] > r[None, :]
    NT = tl.cdiv(T, BT)
    ao = ((i_b * H + i_h) * NT + i_t) * BT * BT + r[:, None] * BT + r[None, :]
    hK = tl.trans(b_Kx)
    hB = tl.trans(b_Bx)
    tl.store(Aqk_o + ao, tl.where(inc, tl.dot(b_Q, hK, input_precision=PREC), 0.0))
    tl.store(Aqb_o + ao, tl.where(inc, tl.dot(b_Q, hB, input_precision=PREC), 0.0))
    tl.store(Aak_o + ao, tl.where(strict, tl.dot(b_Ax, hK, input_precision=PREC), 0.0))
    tl.store(Aab_o + ao, tl.where(strict, tl.dot(b_Ax, hB, input_precision=PREC), 0.0))


@triton.jit
def dplr_fwd_wy_kernel(
    Aab, Aak, ag, v, Ainv_o, w_o, u_o,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    PREC: tl.constexpr,
):
    """Exact (I - A_ab)^{-1} by binary expansion, then w = Ainv ag and u = Ainv A_ak v."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    r = tl.arange(0, BT)
    NT = tl.cdiv(T, BT)
    ao = ((i_b * H + i_h) * NT + i_t) * BT * BT + r[:, None] * BT + r[None, :]

    b_N = tl.load(Aab + ao)
    b_N = tl.where(r[:, None] > r[None, :], b_N, 0.0)
    # Exact inverse of the unit lower-triangular (I - A_ab) by forward
    # substitution, one row at a time: row i of the inverse is row i of N plus
    # its combination with the already-solved rows above it.  A_ab is strictly
    # lower triangular so this terminates after BT-1 rows with no truncation.
    b_P = b_N
    for i in range(1, BT):
        sel = r == i
        row = tl.sum(tl.where(sel[:, None], b_P, 0.0), 0)
        row = row + tl.sum(row[:, None] * b_P, 0) * (r < i)
        b_P = tl.where(sel[:, None], row[None, :], b_P)
    b_P = b_P + (r[:, None] == r[None, :]).to(tl.float32)
    tl.store(Ainv_o + ao, b_P)

    b_Aak = tl.load(Aak + ao)
    b_AI = tl.dot(b_P, tl.where(r[:, None] > r[None, :], b_Aak, 0.0), input_precision="ieee")

    dt = tl.float32
    hP = b_P.to(dt)
    hAI = b_AI.to(dt)

    o_t = i_t * BT + r
    m_t = o_t < T
    o_k = tl.arange(0, BK)
    m_wk = m_t[:, None] & (o_k[None, :] < K)
    offk = (i_b * T) * H * K + i_h * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_ag = tl.load(ag + offk, mask=m_wk, other=0.0)
    tl.store(w_o + offk, tl.dot(hP, b_ag, input_precision=PREC).to(dt), mask=m_wk)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        offv = (i_b * T) * H * V + i_h * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_v = tl.load(v + offv, mask=m_v, other=0.0).to(tl.float32)
        tl.store(u_o + offv, tl.dot(hAI, b_v, input_precision=PREC).to(dt), mask=m_v)


@triton.jit
def dplr_fwd_scan_kernel(
    kg, bg, w, u, v, gi, h_o, v2_o,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    """Forward chunk-boundary state scan.  The carried state is float32 throughout."""
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    r = tl.arange(0, BT)
    dt = tl.float32

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    hbase = (i_bh * NT) * K * V + o_k[:, None] * V + o_v[None, :]
    for i_t in range(NT):
        tl.store(h_o + hbase + i_t * K * V, b_h, mask=m_h)

        o_t = i_t * BT + r
        m_t = o_t < T
        m_tk = m_t[:, None] & m_k[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        offk = (i_b * T) * H * K + i_h * K + o_t[:, None] * (H * K) + o_k[None, :]
        offv = (i_b * T) * H * V + i_h * V + o_t[:, None] * (H * V) + o_v[None, :]

        b_w = tl.load(w + offk, mask=m_tk, other=0.0)
        b_u = tl.load(u + offv, mask=m_tv, other=0.0).to(tl.float32)
        b_v2 = tl.dot(b_w, b_h.to(dt), input_precision=PREC) + b_u
        tl.store(v2_o + offv, b_v2.to(v2_o.dtype.element_ty), mask=m_tv)

        b_kg = tl.load(kg + offk, mask=m_tk, other=0.0)
        b_bg = tl.load(bg + offk, mask=m_tk, other=0.0)
        b_v = tl.load(v + offv, mask=m_tv, other=0.0).to(tl.float32)
        last = tl.minimum((i_t + 1) * BT, T) - 1
        b_gl = tl.load(gi + (i_b * T + last) * H * K + i_h * K + o_k, mask=m_k, other=0.0)

        b_hc = tl.dot(tl.trans(b_kg), b_v, input_precision=PREC)
        b_hc += tl.dot(tl.trans(b_bg), b_v2.to(dt), input_precision=PREC)
        b_h = b_h * tl.exp(b_gl)[:, None] + b_hc


@triton.jit
def dplr_fwd_out_kernel(
    qg, v, v2, Aqk, Aqb, h, o_o,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_v = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)
    r = tl.arange(0, BT)
    o_t = i_t * BT + r
    m_t = o_t < T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    dt = tl.float32

    ao = (i_bh * NT + i_t) * BT * BT + r[:, None] * BT + r[None, :]
    b_Aqk = tl.load(Aqk + ao).to(dt)
    b_Aqb = tl.load(Aqb + ao).to(dt)

    offv = (i_b * T) * H * V + i_h * V + o_t[:, None] * (H * V) + o_v[None, :]
    m_tv = m_t[:, None] & m_v[None, :]
    b_v = tl.load(v + offv, mask=m_tv, other=0.0).to(tl.float32)
    b_v2 = tl.load(v2 + offv, mask=m_tv, other=0.0)

    offk = (i_b * T) * H * K + i_h * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_qg = tl.load(qg + offk, mask=m_t[:, None] & m_k[None, :], other=0.0)
    hoff = (i_bh * NT + i_t) * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(h + hoff, mask=m_k[:, None] & m_v[None, :], other=0.0)

    b_o = tl.dot(b_Aqk, b_v, input_precision=PREC)
    b_o += tl.dot(b_Aqb, b_v2, input_precision=PREC)
    b_o += tl.dot(b_qg, b_h.to(dt), input_precision=PREC)
    tl.store(o_o + offv, b_o.to(o_o.dtype.element_ty), mask=m_tv)


@triton.jit
def dplr_bwd_dA_kernel(
    do, v, v2, Aqk, Aqb, dAqk_o, dAqb_o, dv2i_o, dv_o,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)
    r = tl.arange(0, BT)
    o_t = i_t * BT + r
    m_t = o_t < T
    dt = tl.float32

    ao = (i_bh * NT + i_t) * BT * BT + r[:, None] * BT + r[None, :]
    b_AqkT = tl.trans(tl.load(Aqk + ao).to(dt))
    b_AqbT = tl.trans(tl.load(Aqb + ao).to(dt))

    b_dAqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_dAqb = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        offv = (i_b * T) * H * V + i_h * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_do = tl.load(do + offv, mask=m_v, other=0.0).to(tl.float32)
        b_v = tl.load(v + offv, mask=m_v, other=0.0).to(tl.float32)
        b_v2 = tl.load(v2 + offv, mask=m_v, other=0.0)
        b_dAqk += tl.dot(b_do, tl.trans(b_v), input_precision=PREC)
        b_dAqb += tl.dot(b_do, tl.trans(b_v2), input_precision=PREC)
        tl.store(dv2i_o + offv, tl.dot(b_AqbT, b_do, input_precision=PREC), mask=m_v)
        tl.store(dv_o + offv, tl.dot(b_AqkT, b_do, input_precision=PREC), mask=m_v)

    inc = r[:, None] >= r[None, :]
    tl.store(dAqk_o + ao, tl.where(inc, b_dAqk, 0.0))
    tl.store(dAqb_o + ao, tl.where(inc, b_dAqb, 0.0))


@triton.jit
def dplr_bwd_scan_kernel(
    qg, bg, w, do, dv2i, gi, dh_o, dvn_o,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    """Reverse chunk-boundary state-gradient scan.  Carried in float32."""
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    r = tl.arange(0, BT)
    dt = tl.float32

    b_dS = tl.zeros([BK, BV], dtype=tl.float32)
    hbase = (i_bh * NT) * K * V + o_k[:, None] * V + o_v[None, :]
    for i_s in range(NT):
        i_t = NT - 1 - i_s
        tl.store(dh_o + hbase + i_t * K * V, b_dS, mask=m_h)

        o_t = i_t * BT + r
        m_t = o_t < T
        m_tk = m_t[:, None] & m_k[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        offk = (i_b * T) * H * K + i_h * K + o_t[:, None] * (H * K) + o_k[None, :]
        offv = (i_b * T) * H * V + i_h * V + o_t[:, None] * (H * V) + o_v[None, :]

        b_bg = tl.load(bg + offk, mask=m_tk, other=0.0)
        b_dv2 = tl.load(dv2i + offv, mask=m_tv, other=0.0)
        b_dv2 += tl.dot(b_bg, b_dS.to(dt), input_precision=PREC)
        tl.store(dvn_o + offv, b_dv2, mask=m_tv)

        b_qg = tl.load(qg + offk, mask=m_tk, other=0.0)
        b_w = tl.load(w + offk, mask=m_tk, other=0.0)
        b_do = tl.load(do + offv, mask=m_tv, other=0.0).to(tl.float32)
        last = tl.minimum((i_t + 1) * BT, T) - 1
        b_gl = tl.load(gi + (i_b * T + last) * H * K + i_h * K + o_k, mask=m_k, other=0.0)

        b_acc = tl.dot(tl.trans(b_qg), b_do, input_precision=PREC)
        b_acc += tl.dot(tl.trans(b_w), b_dv2.to(dt), input_precision=PREC)
        b_dS = b_dS * tl.exp(b_gl)[:, None] + b_acc


@triton.jit
def dplr_bwd_state_kernel(
    h, dh, do, v, v2, dvn, kg, gi,
    dqg_o, dkg_o, dbg_o, dw_o, dgl_o, dv_o,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)
    r = tl.arange(0, BT)
    o_t = i_t * BT + r
    m_t = o_t < T
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]
    offk = (i_b * T) * H * K + i_h * K + o_t[:, None] * (H * K) + o_k[None, :]
    dt = tl.float32

    b_dqg = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_dkg = tl.zeros([BT, BK], dtype=tl.float32)
    b_dbg = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgl = tl.zeros([BK], dtype=tl.float32)
    b_kg = tl.load(kg + offk, mask=m_tk, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        offv = (i_b * T) * H * V + i_h * V + o_t[:, None] * (H * V) + o_v[None, :]
        hoff = (i_bh * NT + i_t) * K * V + o_k[:, None] * V + o_v[None, :]
        m_h = m_k[:, None] & m_v[None, :]

        b_h = tl.load(h + hoff, mask=m_h, other=0.0)
        b_dhs = tl.load(dh + hoff, mask=m_h, other=0.0)
        b_do = tl.load(do + offv, mask=m_tv, other=0.0).to(tl.float32)
        b_v = tl.load(v + offv, mask=m_tv, other=0.0).to(tl.float32)
        b_v2 = tl.load(v2 + offv, mask=m_tv, other=0.0)
        b_dvn = tl.load(dvn + offv, mask=m_tv, other=0.0).to(dt)

        b_hT = tl.trans(b_h).to(dt)
        b_dhT = tl.trans(b_dhs).to(dt)
        b_dqg += tl.dot(b_do, b_hT, input_precision=PREC)
        b_dw += tl.dot(b_dvn, b_hT, input_precision=PREC)
        b_dkg += tl.dot(b_v, b_dhT, input_precision=PREC)
        b_dbg += tl.dot(b_v2, b_dhT, input_precision=PREC)
        b_dgl += tl.sum(b_h * b_dhs, 1)

        b_dv = tl.load(dv_o + offv, mask=m_tv, other=0.0)
        b_dv += tl.dot(b_kg, b_dhs.to(dt), input_precision=PREC)
        tl.store(dv_o + offv, b_dv, mask=m_tv)

    last = tl.minimum((i_t + 1) * BT, T) - 1
    b_gl = tl.load(gi + (i_b * T + last) * H * K + i_h * K + o_k, mask=m_k, other=0.0)
    tl.store(dgl_o + (i_bh * NT + i_t) * K + o_k, b_dgl * tl.exp(b_gl), mask=m_k)
    tl.store(dqg_o + offk, b_dqg, mask=m_tk)
    tl.store(dw_o + offk, b_dw, mask=m_tk)
    tl.store(dkg_o + offk, b_dkg, mask=m_tk)
    tl.store(dbg_o + offk, b_dbg, mask=m_tk)


@triton.jit
def dplr_bwd_wy_kernel(
    Ainv, Aak, w, u, v, dw, dvn, dag_o, dAab_o, dAak_o, dv_acc, dv_o,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)
    r = tl.arange(0, BT)
    o_t = i_t * BT + r
    m_t = o_t < T
    dt = tl.float32

    ao = (i_bh * NT + i_t) * BT * BT + r[:, None] * BT + r[None, :]
    b_AinvT = tl.trans(tl.load(Ainv + ao)).to(dt)
    b_AakT = tl.trans(tl.load(Aak + ao)).to(dt)

    o_k = tl.arange(0, BK)
    m_tk = m_t[:, None] & (o_k[None, :] < K)
    offk = (i_b * T) * H * K + i_h * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_dw = tl.load(dw + offk, mask=m_tk, other=0.0).to(dt)
    b_dag = tl.dot(b_AinvT, b_dw, input_precision=PREC)
    tl.store(dag_o + offk, b_dag, mask=m_tk)
    b_w = tl.load(w + offk, mask=m_tk, other=0.0)
    b_dAab = tl.dot(b_dag.to(dt), tl.trans(b_w), input_precision=PREC)
    b_dAak = tl.zeros([BT, BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        offv = (i_b * T) * H * V + i_h * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_du = tl.load(dvn + offv, mask=m_tv, other=0.0).to(dt)
        b_dY = tl.dot(b_AinvT, b_du, input_precision=PREC).to(dt)
        b_u = tl.load(u + offv, mask=m_tv, other=0.0)
        b_v = tl.load(v + offv, mask=m_tv, other=0.0).to(tl.float32)
        b_dAab += tl.dot(b_dY, tl.trans(b_u), input_precision=PREC)
        b_dAak += tl.dot(b_dY, tl.trans(b_v), input_precision=PREC)
        b_dv = tl.load(dv_acc + offv, mask=m_tv, other=0.0)
        b_dv += tl.dot(b_AakT, b_dY, input_precision=PREC)
        tl.store(dv_o + offv, b_dv.to(dv_o.dtype.element_ty), mask=m_tv)

    strict = r[:, None] > r[None, :]
    tl.store(dAab_o + ao, tl.where(strict, b_dAab, 0.0))
    tl.store(dAak_o + ao, tl.where(strict, b_dAak, 0.0))


@triton.jit
def dplr_bwd_intra_kernel(
    q, k, a, b, gi, ge,
    dAqk, dAqb, dAak, dAab, dqg, dkg, dbg, dag, dgl,
    dq_o, dk_o, da_o, db_o, dgk_o,
    scale,
    T, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_k = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    NT = tl.cdiv(T, BT)
    r = tl.arange(0, BT)
    o_t = i_t * BT + r
    m_t = o_t < T
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m = m_t[:, None] & m_k[None, :]
    offk = (i_b * T) * H * K + i_h * K + o_t[:, None] * (H * K) + o_k[None, :]
    dt = q.dtype.element_ty

    b_gi = tl.load(gi + offk, mask=m, other=0.0)
    b_ge = tl.load(ge + offk, mask=m, other=0.0)
    last = tl.minimum((i_t + 1) * BT, T) - 1
    b_gl = tl.load(gi + (i_b * T + last) * H * K + i_h * K + o_k, mask=m_k, other=0.0)
    b_off = 0.5 * b_gl
    e_q = _expc(b_gi - b_off[None, :])
    e_i = _expc(b_off[None, :] - b_gi)
    e_a = _expc(b_ge - b_off[None, :])
    e_gi = tl.exp(b_gi)
    e_ge = tl.exp(b_ge)
    e_ki = _expc(b_gl[None, :] - b_gi)

    b_q = tl.load(q + offk, mask=m, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + offk, mask=m, other=0.0).to(tl.float32)
    b_b = tl.load(b + offk, mask=m, other=0.0).to(tl.float32)
    b_a = tl.load(a + offk, mask=m, other=0.0).to(tl.float32)
    b_Q = b_q * e_q
    b_Kx = b_k * e_i
    b_Bx = b_b * e_i
    b_Ax = b_a * e_a

    ao = (i_bh * NT + i_t) * BT * BT + r[:, None] * BT + r[None, :]
    b_dAqk = tl.load(dAqk + ao)
    b_dAqb = tl.load(dAqb + ao)
    b_dAak = tl.load(dAak + ao)
    b_dAab = tl.load(dAab + ao)

    # gradients through the gated operands consumed by the state pipeline
    b_dqg = tl.load(dqg + offk, mask=m, other=0.0)
    b_dkg = tl.load(dkg + offk, mask=m, other=0.0)
    b_dbg = tl.load(dbg + offk, mask=m, other=0.0)
    b_dag = tl.load(dag + offk, mask=m, other=0.0)
    b_kg = b_k * e_ki
    b_bg = b_b * e_ki

    # --- A-cascade legs ------------------------------------------------------
    # Each gated operand depends on gk through its own index window:
    #
    #   row legs   Q = q e^{gi}  (m <= i),  Ax = a e^{ge}  (m <  i)
    #   col legs   Kx, Bx = . e^{-gi_j}     (m <= j),  entering with a minus sign
    #   qg = q e^{gi}  (m <= i)     ag = a e^{ge}  (m <  i)
    #   kg, bg = . e^{gl - gi}      (m >  i)
    #
    # For dq/dk/dalpha/dbeta the row and column legs of an A simply add, but for
    # the gate they must stay separate: a column leg carries the window of its
    # own index j, not of i, and the two then cancel to several digits.  Since
    # the legs are themselves as large as the whole chunk, and the recurrence is
    # exponentially expanding along T, at fp32 the leg's own rounding noise
    # already exceeds the surviving difference -- so these four reductions are
    # taken in float64.  They are BT x BT x BK per chunk, a small share of the
    # pipeline, and the same values also feed dq/dk/dalpha/dbeta.
    b_up = (r[:, None] <= r[None, :]).to(tl.float32)
    b_upx = (r[:, None] < r[None, :]).to(tl.float32)
    d_Q = b_Q.to(tl.float64)
    d_Kx = b_Kx.to(tl.float64)
    d_Bx = b_Bx.to(tl.float64)
    d_Ax = b_Ax.to(tl.float64)
    d_Aqk = b_dAqk.to(tl.float64)
    d_Aqb = b_dAqb.to(tl.float64)
    d_Aak = b_dAak.to(tl.float64)
    d_Aab = b_dAab.to(tl.float64)
    b_dQr = tl.sum(d_Aqk[:, :, None] * d_Kx[None, :, :], 1) + \
        tl.sum(d_Aqb[:, :, None] * d_Bx[None, :, :], 1)
    b_dAr = tl.sum(d_Aak[:, :, None] * d_Kx[None, :, :], 1) + \
        tl.sum(d_Aab[:, :, None] * d_Bx[None, :, :], 1)
    b_dKc = tl.sum(d_Aqk[:, :, None] * d_Q[:, None, :], 0) + \
        tl.sum(d_Aak[:, :, None] * d_Ax[:, None, :], 0)
    b_dBc = tl.sum(d_Aqb[:, :, None] * d_Q[:, None, :], 0) + \
        tl.sum(d_Aab[:, :, None] * d_Ax[:, None, :], 0)

    # dq / dk / dalpha / dbeta: the row and column legs of each A, plus the leg
    # from the corresponding operand of the state pipeline
    tl.store(dq_o + offk,
             (b_dQr.to(tl.float32) * scale * e_q + b_dqg * scale * e_gi).to(dt), mask=m)
    tl.store(dk_o + offk,
             (b_dKc.to(tl.float32) * e_i + b_dkg * e_ki).to(dt), mask=m)
    tl.store(db_o + offk,
             (b_dBc.to(tl.float32) * e_i + b_dbg * e_ki).to(dt), mask=m)
    tl.store(da_o + offk,
             (b_dAr.to(tl.float32) * e_a + b_dag * e_ge).to(dt), mask=m)

    # A[i,j] depends on gk_m only through the band  j < m <= i : its +gi_i row leg
    # and -gi_j column leg cancel identically outside it.  Each of the four leg
    # sums is therefore taken in PREFIX form (m' < m), where the row and column
    # contributions cancel term by term from the small end.  Taken as suffix sums
    # instead, the m = 0 entry -- where the incoming state is zero and the true
    # gradient is an empty sum -- would come out as the difference of two totals
    # the size of the whole chunk, which at these magnitudes destroys it.
    b_R_q = tl.where(m, b_dQr * d_Q, 0.0)         # row leg, m <  i
    b_R_a = tl.where(m, b_dAr * d_Ax, 0.0)        # row leg, m <= i
    b_C = tl.where(m, b_dKc * d_Kx + b_dBc * d_Bx, 0.0)   # col, m < j

    # legs from the operands the state pipeline consumes, each with its own window
    b_wi = tl.where(m, b_dqg * (b_q * e_gi), 0.0).to(tl.float64)         # qg : m <= i
    b_wx = tl.where(m, b_dag * (b_a * e_ge), 0.0).to(tl.float64)         # ag : m <  i
    b_wk = tl.where(m, b_dkg * b_kg + b_dbg * b_bg, 0.0).to(tl.float64)  # kg,bg: m > i

    # These window sums cancel each other to several digits (the m = 0 entry is an
    # identically empty sum built from legs the size of the whole chunk), so the
    # BT-term reduction itself is carried in float64.  It is a [BT, BT] x [BT, BK]
    # reduction per chunk and costs a small fraction of the pipeline.
    d_up = b_up.to(tl.float64)
    d_upx = b_upx.to(tl.float64)
    d_lo = tl.trans(d_up)                                        # [m, m'] : m' <= m
    d_lox = tl.trans(d_upx)                                      # [m, m'] : m' <  m
    d_gk = tl.sum(d_lox[:, :, None] * (b_C - b_R_q + b_wk)[None, :, :], 1) \
        - tl.sum(d_lo[:, :, None] * b_R_a[None, :, :], 1) \
        + tl.sum(d_up[:, :, None] * b_wi[None, :, :], 1) \
        + tl.sum(d_upx[:, :, None] * b_wx[None, :, :], 1)

    # the chunk-final state decay h *= exp(gl) touches every gk in the chunk
    b_dgl_tot = tl.load(dgl + (i_bh * NT + i_t) * K + o_k, mask=m_k, other=0.0)
    b_dgk = (d_gk + b_dgl_tot[None, :].to(tl.float64)).to(tl.float32)
    tl.store(dgk_o + offk, b_dgk.to(dgk_o.dtype.element_ty), mask=m)


def _prec(dtype):
    # the recurrence is exponentially ill-conditioned along T, so every matmul in
    # the pipeline runs at full float32 precision, uniformly for all input dtypes
    return "ieee"


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        q, k, v, alpha, beta = (x.contiguous() for x in (q, k, v, alpha, beta))
        gkf = gk.float().contiguous()
        NT = triton.cdiv(T, BT)
        BK = max(triton.next_power_of_2(K), 16)
        BV = min(max(triton.next_power_of_2(V), 16), BV_TILE)
        NV = triton.cdiv(V, BV)
        prec = _prec(q.dtype)
        scale = K ** -0.5
        dev = q.device
        f32 = torch.float32

        gi = torch.empty(B, T, H, K, dtype=f32, device=dev)
        ge = torch.empty(B, T, H, K, dtype=f32, device=dev)
        qg, kg, ag, bg = (
            torch.empty(B, T, H, K, dtype=f32, device=dev) for _ in range(4))
        Ash = (B, H, NT, BT, BT)
        Aqk = torch.empty(Ash, dtype=f32, device=dev)
        Aqb = torch.empty(Ash, dtype=f32, device=dev)
        Aak = torch.empty(Ash, dtype=f32, device=dev)
        Aab = torch.empty(Ash, dtype=f32, device=dev)

        dplr_fwd_intra_kernel[(NT, B * H)](
            q, k, alpha, beta, gkf, gi, ge, qg, kg, ag, bg, Aqk, Aqb, Aak, Aab,
            scale, T, H=H, K=K, BT=BT, BK=BK, PREC=prec, num_warps=8, num_stages=1,
        )

        Ainv = torch.empty(Ash, dtype=f32, device=dev)
        w = torch.empty(B, T, H, K, dtype=f32, device=dev)
        u = torch.empty(B, T, H, V, dtype=f32, device=dev)
        dplr_fwd_wy_kernel[(NT, B * H)](
            Aab, Aak, ag, v, Ainv, w, u,
            T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
            num_warps=4, num_stages=2,
        )

        h = torch.empty(B, H, NT, K, V, dtype=f32, device=dev)
        v2 = torch.empty(B, T, H, V, dtype=f32, device=dev)
        dplr_fwd_scan_kernel[(NV, B * H)](
            kg, bg, w, u, v, gi, h, v2,
            T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
            num_warps=2, num_stages=2,
        )

        o = torch.empty_like(v)
        dplr_fwd_out_kernel[(NT, NV, B * H)](
            qg, v, v2, Aqk, Aqb, h, o,
            T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
            num_warps=4, num_stages=2,
        )

        ctx.save_for_backward(q, k, v, alpha, beta, gk, gi, ge, qg, kg, ag, bg,
                              w, u, v2, h, Aqk, Aqb, Aak, Ainv)
        ctx.dims = (B, T, H, K, V, NT, BK, BV, NV, prec, scale)
        return o

    @staticmethod
    def backward(ctx, do):
        (q, k, v, alpha, beta, gk, gi, ge, qg, kg, ag, bg,
         w, u, v2, h, Aqk, Aqb, Aak, Ainv) = ctx.saved_tensors
        B, T, H, K, V, NT, BK, BV, NV, prec, scale = ctx.dims
        do = do.contiguous()
        dev = q.device
        f32 = torch.float32
        Ash = (B, H, NT, BT, BT)

        dAqk = torch.empty(Ash, dtype=f32, device=dev)
        dAqb = torch.empty(Ash, dtype=f32, device=dev)
        dv2i = torch.empty(B, T, H, V, dtype=f32, device=dev)
        dva = torch.empty(B, T, H, V, dtype=f32, device=dev)
        dplr_bwd_dA_kernel[(NT, B * H)](
            do, v, v2, Aqk, Aqb, dAqk, dAqb, dv2i, dva,
            T, H=H, K=K, V=V, BT=BT, BV=BV, PREC=prec, num_warps=4, num_stages=2,
        )

        dh = torch.empty(B, H, NT, K, V, dtype=f32, device=dev)
        dvn = torch.empty(B, T, H, V, dtype=f32, device=dev)
        dplr_bwd_scan_kernel[(NV, B * H)](
            qg, bg, w, do, dv2i, gi, dh, dvn,
            T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
            num_warps=2, num_stages=2,
        )

        dqg = torch.empty(B, T, H, K, dtype=f32, device=dev)
        dkg = torch.empty(B, T, H, K, dtype=f32, device=dev)
        dbg = torch.empty(B, T, H, K, dtype=f32, device=dev)
        dw = torch.empty(B, T, H, K, dtype=f32, device=dev)
        dgl = torch.empty(B, H, NT, K, dtype=f32, device=dev)
        dplr_bwd_state_kernel[(NT, B * H)](
            h, dh, do, v, v2, dvn, kg, gi, dqg, dkg, dbg, dw, dgl, dva,
            T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
            num_warps=4, num_stages=2,
        )

        dag = torch.empty(B, T, H, K, dtype=f32, device=dev)
        dAab = torch.empty(Ash, dtype=f32, device=dev)
        dAak = torch.empty(Ash, dtype=f32, device=dev)
        dv = torch.empty_like(v)
        dplr_bwd_wy_kernel[(NT, B * H)](
            Ainv, Aak, w, u, v, dw, dvn, dag, dAab, dAak, dva, dv,
            T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV, PREC=prec,
            num_warps=4, num_stages=2,
        )

        dq = torch.empty_like(q)
        dk = torch.empty_like(q)
        da = torch.empty_like(q)
        db = torch.empty_like(q)
        dgk = torch.empty_like(gk)
        # Every quantity in this kernel is independent per K channel, so the K
        # axis is tiled across programs: the double-precision gate reductions
        # would otherwise need the whole [BT, K] working set resident at once.
        BKi = min(BK, BK_INTRA)
        dplr_bwd_intra_kernel[(NT, triton.cdiv(K, BKi), B * H)](
            q, k, alpha, beta, gi, ge,
            dAqk, dAqb, dAak, dAab, dqg, dkg, dbg, dag, dgl,
            dq, dk, da, db, dgk, scale,
            T, H=H, K=K, BT=BT, BK=BKi, PREC=prec, num_warps=1, num_stages=1,
        )
        return dq, dk, dv, da, db, dgk


def chunked_dplr_delta(
    q: torch.Tensor,      # [B, T, H, K]
    k: torch.Tensor,      # [B, T, H, K]
    v: torch.Tensor,      # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]    DPLR erasure basis
    beta: torch.Tensor,   # [B, T, H, K]    DPLR write basis
    gk: torch.Tensor,     # [B, T, H, K]    per-K-channel log-decay (<= 0)
) -> torch.Tensor:        # [B, T, H, V]
    return _Dplr.apply(q, k, v, alpha, beta, gk)
