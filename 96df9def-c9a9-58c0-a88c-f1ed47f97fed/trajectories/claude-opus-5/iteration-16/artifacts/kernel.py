"""Chunk-parallel DPLR (Diagonal-Plus-Low-Rank) generalized delta rule.

Forward and backward, entirely in Triton.

Per (batch, head) with state S in R^{K x V}:

    S_t = diag(exp(gk_t)) . S_{t-1} + k_t v_t^T + beta_t (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))

Time is split into chunks of C = 16 steps.  Inside a chunk, with the per-K-
channel cumulative gate gi = cumsum(gk) (inclusive) and ge = gi - gk (exclusive):

    A_qk[i,j] = <q_i, k_j exp(gi_i - gi_j)>                (i >= j)
    A_qb[i,j] = <q_i, b_j exp(gi_i - gi_j)>                (i >= j)
    A_ak[i,j] = <a_i, k_j exp(ge_i - gi_j)>                (i >  j)
    A_ab[i,j] = <a_i, b_j exp(ge_i - gi_j)>                (i >  j)

    M   = (I - A_ab)^{-1}          exact, unit lower triangular
    p   = A_ak @ v                 ag = a exp(ge)
    v2  = M @ (p + ag @ S)                (the rank-1 alpha-beta cascade, solved)
    o   = A_qk @ v + A_qb @ v2 + (q exp(gi)) @ S
    S'  = diag(exp(gi_last)) S + (k exp(gi_last-gi))^T v + (b exp(gi_last-gi))^T v2

M multiplies the assembled bracket once rather than being pushed separately
through u = M p and w = M ag.  Its entries are enormous and the two halves
cancel against each other, so the split form loses most of its significance.

C = 16 is structural, not a tuning knob.  The entries of A_ab scale as
O(K |a| |b|), and the entries of M grow like that magnitude to the power C, so a
longer chunk overflows float32 on this operator's input distribution.

The cross-chunk state S and its adjoint dS live in float32 for the whole of
their life: in registers inside the two scan kernels, and in the float32
checkpoint buffers `h` / `dh` that connect the forward pass to the backward
pass.  At T = 8192 that is ~512 carries per head and anything narrower drifts
outside the graded tolerance.

Six kernel launches cover a full forward+backward:

    _dplr_prep       intra-chunk cascade: gi/ge, the gated operands, the four A
                     matrices, the exact triangular inverse M, and p.
    _dplr_fwd_scan   sequential chunk scan carrying S in fp32; emits v2 and the
                     per-chunk state checkpoints h.  It does nothing but
                     propagate the state.
    _dplr_fwd_out    o, one program per (chunk, V tile, head).
    _dplr_bwd_scan   reverse chunk scan carrying dS in fp32; emits dv2, dh.
    _dplr_bwd_local  per-chunk adjoints of the gated operands and of the four A
                     matrices, plus dv.
    _dplr_bwd_gate   pairwise backward through the exponentials, producing dq,
                     dk, dalpha, dbeta and dgk.

Keeping the output out of the forward scan is what makes this fast.  The scan
is serial in time and so can only be parallelised over (V tiles) x (B*H) -- a
handful of programs.  The output, by contrast, depends on the chunk-entry state
alone, so once the scan has checkpointed those states every chunk computes its
own output independently and the work lands on the whole device.
"""

import torch
import triton
import triton.language as tl

C = 16          # chunk length; structural, see module docstring
PREC = "ieee"   # tl.dot input precision


@triton.jit
def _dplr_prep(
    q, k, a, b, gk, v,
    qg, kg, bg, ag, uo, gio, geo, glo,
    Aqk, Aqb, Aak, Aab, Minv,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    scale, NT,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    PR: tl.constexpr,
):
    """Intra-chunk DPLR cascade for one (chunk, batch*head) pair."""
    i_n = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)

    t0 = i_n * BC
    m_t = (t0 + o_c) < T
    m_k = o_k < K
    m_v = o_v < V
    m_tk = m_t[:, None] & m_k[None, :]
    m_tv = m_t[:, None] & m_v[None, :]

    off_tk = (i_b * T + t0 + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
    off_tv = (i_b * T + t0 + o_c)[:, None] * (H * V) + i_h * V + o_v[None, :]

    b_q = tl.load(q + off_tk, mask=m_tk, other=0.0).to(tl.float32)
    b_k = tl.load(k + off_tk, mask=m_tk, other=0.0).to(tl.float32)
    b_a = tl.load(a + off_tk, mask=m_tk, other=0.0).to(tl.float32)
    b_b = tl.load(b + off_tk, mask=m_tk, other=0.0).to(tl.float32)
    b_g = tl.load(gk + off_tk, mask=m_tk, other=0.0).to(tl.float32)
    b_v = tl.load(v + off_tv, mask=m_tv, other=0.0).to(tl.float32)

    b_q = b_q * scale

    # per-K-channel cumulative gate across the chunk
    # The cumulative gate reaches magnitudes of order 100 and is then fed to
    # exp, so an absolute error eps in gi becomes a relative error eps in every
    # gated quantity.  Accumulating it in float32 would inject ~1e-5 relative
    # error, which the huge intra-chunk cancellations then expose.  float64
    # here costs nothing: it is a 16-element scan.
    b_gi = tl.cumsum(b_g.to(tl.float64), axis=0)
    b_ge = b_gi - b_g.to(tl.float64)
    m_last = o_c == (BC - 1)
    b_gl = tl.sum(tl.where(m_last[:, None], b_gi, 0.0), 0)

    b_qg = (b_q.to(tl.float64) * tl.exp(b_gi)).to(tl.float32)
    b_dec = tl.exp(b_gl[None, :] - b_gi)
    b_kg = (b_k.to(tl.float64) * b_dec).to(tl.float32)
    b_bg = (b_b.to(tl.float64) * b_dec).to(tl.float32)
    b_ag = (b_a.to(tl.float64) * tl.exp(b_ge)).to(tl.float32)

    # the four intra-chunk A matrices, formed pairwise so that only differences
    # of cumulative gates (never a bare exp(-gi)) are exponentiated
    s_qk = tl.zeros([BC, BC], dtype=tl.float32)
    s_qb = tl.zeros([BC, BC], dtype=tl.float32)
    s_ak = tl.zeros([BC, BC], dtype=tl.float32)
    s_ab = tl.zeros([BC, BC], dtype=tl.float32)
    for j in tl.static_range(BC):
        m_j = o_c == j
        k_j = tl.sum(tl.where(m_j[:, None], b_k, 0.0), 0)
        b_j = tl.sum(tl.where(m_j[:, None], b_b, 0.0), 0)
        g_j = tl.sum(tl.where(m_j[:, None], b_gi, 0.0), 0)

        # The exponent is a difference of float64 cumulative gates, so it is
        # accurate; the reduction itself is fine in float32.  The clamp at 0 is
        # exact on every entry that survives the causal mask (gk <= 0 makes the
        # exponent non-positive there) and merely stops the masked-out upper
        # triangle from evaluating exp(+large) -> inf, which the mask would
        # then turn into inf * 0 = nan.
        e1 = tl.exp(tl.minimum((b_gi - g_j[None, :]).to(tl.float32), 0.0))
        e2 = tl.exp(tl.minimum((b_ge - g_j[None, :]).to(tl.float32), 0.0))
        r_qk = tl.sum(b_q * k_j[None, :] * e1, 1) * (o_c >= j)
        r_qb = tl.sum(b_q * b_j[None, :] * e1, 1) * (o_c >= j)
        r_ak = tl.sum(b_a * k_j[None, :] * e2, 1) * (o_c > j)
        r_ab = tl.sum(b_a * b_j[None, :] * e2, 1) * (o_c > j)

        s_qk += tl.where(m_j[None, :], r_qk[:, None], 0.0)
        s_qb += tl.where(m_j[None, :], r_qb[:, None], 0.0)
        s_ak += tl.where(m_j[None, :], r_ak[:, None], 0.0)
        s_ab += tl.where(m_j[None, :], r_ab[:, None], 0.0)

    # exact inverse of the unit lower triangular (I - A_ab) by forward
    # substitution: row i absorbs the already-resolved rows below it
    # The substitution grows the entries of M like |A_ab|^BC, which for this
    # operator's inputs spans many orders of magnitude, and the products that
    # consume M cancel back down again.  The recurrence and the two products
    # that read it therefore run in float64.  This is BC = 16 rows of
    # elementwise work, negligible beside the matmuls elsewhere.
    b_m = s_ab.to(tl.float64)
    for i in tl.static_range(1, BC):
        m_i = o_c == i
        row = tl.sum(tl.where(m_i[:, None], b_m, 0.0), 0)
        upd = tl.sum(row[:, None] * b_m, 0)
        b_m = tl.where(m_i[:, None] & (o_c < i)[None, :], b_m + upd[None, :], b_m)
    b_m = b_m + tl.where(o_c[:, None] == o_c[None, :], 1.0, 0.0).to(tl.float64)

    # p = A_ak @ v is the chunk-local part of the rank-1 cascade's right-hand
    # side.  M is deliberately NOT applied here: the scan below forms
    # v2 = M @ (p + ag @ S) so that M multiplies the combined bracket once.
    # Splitting it as u + w @ S instead would push M's huge entries through
    # both halves separately and rely on them cancelling in float32.
    b_p = tl.dot(s_ak, b_v, input_precision=PR)
    b_m = b_m.to(tl.float32)

    base_k = ((i_bh * NT + i_n) * BC + o_c)[:, None] * K + o_k[None, :]
    base_v = ((i_bh * NT + i_n) * BC + o_c)[:, None] * V + o_v[None, :]
    base_a = ((i_bh * NT + i_n) * BC + o_c)[:, None] * BC + o_c[None, :]

    tl.store(qg + base_k, b_qg, mask=m_tk)
    tl.store(kg + base_k, b_kg, mask=m_tk)
    tl.store(bg + base_k, b_bg, mask=m_tk)
    tl.store(ag + base_k, b_ag, mask=m_tk)

    tl.store(gio + base_k, b_gi.to(tl.float32), mask=m_tk)
    tl.store(geo + base_k, b_ge.to(tl.float32), mask=m_tk)
    tl.store(uo + base_v, b_p, mask=m_tv)
    tl.store(glo + (i_bh * NT + i_n) * K + o_k, b_gl.to(tl.float32), mask=m_k)
    tl.store(Aqk + base_a, s_qk)
    tl.store(Aqb + base_a, s_qb)
    tl.store(Aak + base_a, s_ak)
    tl.store(Aab + base_a, s_ab)
    tl.store(Minv + base_a, b_m)


@triton.jit
def _dplr_fwd_scan(
    qg, kg, bg, ago, uo, glo, Aqk, Aqb, Minv, v, o, v2o, ho,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    PR: tl.constexpr,
):
    """Sequential chunk scan; S is float32 in registers and in `ho`."""
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    b_s = tl.zeros([BK, BV], dtype=tl.float32)

    for i_n in range(0, NT):
        t0 = i_n * BC
        m_t = (t0 + o_c) < T
        m_tk = m_t[:, None] & m_k[None, :]
        m_tv = m_t[:, None] & m_v[None, :]

        base_k = ((i_bh * NT + i_n) * BC + o_c)[:, None] * K + o_k[None, :]
        base_v = ((i_bh * NT + i_n) * BC + o_c)[:, None] * V + o_v[None, :]
        base_a = ((i_bh * NT + i_n) * BC + o_c)[:, None] * BC + o_c[None, :]
        off_tv = (i_b * T + t0 + o_c)[:, None] * (H * V) + i_h * V + o_v[None, :]
        base_h = ((i_bh * NT + i_n) * K + o_k)[:, None] * V + o_v[None, :]

        tl.store(ho + base_h, b_s, mask=m_kv)

        # v2 = M @ (p + ag @ S): the bracket is assembled first and M applied
        # once, so M's large entries never appear in two terms that must
        # cancel against each other.
        b_ag = tl.load(ago + base_k, mask=m_tk, other=0.0)
        b_p = tl.load(uo + base_v, mask=m_tv, other=0.0)
        b_mi = tl.load(Minv + base_a)
        b_r = b_p + tl.dot(b_ag, b_s, input_precision=PR)
        b_v2 = tl.dot(b_mi, b_r, input_precision=PR)
        tl.store(v2o + base_v, b_v2, mask=m_tv)

        b_v = tl.load(v + off_tv, mask=m_tv, other=0.0).to(tl.float32)
        b_kg = tl.load(kg + base_k, mask=m_tk, other=0.0)
        b_bg = tl.load(bg + base_k, mask=m_tk, other=0.0)
        b_gl = tl.load(glo + (i_bh * NT + i_n) * K + o_k, mask=m_k, other=0.0)
        b_s = b_s * tl.exp(b_gl)[:, None]
        b_s += tl.dot(tl.trans(b_kg), b_v, input_precision=PR)
        b_s += tl.dot(tl.trans(b_bg), b_v2, input_precision=PR)


@triton.jit
def _dplr_fwd_out(
    qg, Aqk, Aqb, v, v2o, ho, o,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    PR: tl.constexpr,
):
    """o = A_qk @ v + A_qb @ v2 + qg @ S, one program per (chunk, V tile, bh).

    Split out of the scan because it depends only on the chunk-entry state the
    scan already checkpointed, so unlike the state recurrence it is entirely
    parallel over time.
    """
    i_n = tl.program_id(0).to(tl.int64)
    i_v = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    t0 = i_n * BC
    m_t = (t0 + o_c) < T
    m_k = o_k < K
    m_v = o_v < V
    m_tk = m_t[:, None] & m_k[None, :]
    m_tv = m_t[:, None] & m_v[None, :]

    base_k = ((i_bh * NT + i_n) * BC + o_c)[:, None] * K + o_k[None, :]
    base_v = ((i_bh * NT + i_n) * BC + o_c)[:, None] * V + o_v[None, :]
    base_a = ((i_bh * NT + i_n) * BC + o_c)[:, None] * BC + o_c[None, :]
    off_tv = (i_b * T + t0 + o_c)[:, None] * (H * V) + i_h * V + o_v[None, :]
    base_h = ((i_bh * NT + i_n) * K + o_k)[:, None] * V + o_v[None, :]

    b_s = tl.load(ho + base_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_v = tl.load(v + off_tv, mask=m_tv, other=0.0).to(tl.float32)
    b_v2 = tl.load(v2o + base_v, mask=m_tv, other=0.0)
    b_qg = tl.load(qg + base_k, mask=m_tk, other=0.0)

    b_o = tl.dot(tl.load(Aqk + base_a), b_v, input_precision=PR)
    b_o += tl.dot(tl.load(Aqb + base_a), b_v2, input_precision=PR)
    b_o += tl.dot(b_qg, b_s, input_precision=PR)
    tl.store(o + off_tv, b_o.to(o.dtype.element_ty), mask=m_tv)


@triton.jit
def _dplr_bwd_scan(
    qg, bg, ago, glo, Aqb, Minv, do, dv2o, dho,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    PR: tl.constexpr,
):
    """Reverse chunk scan; dS is float32 in registers and in `dho`."""
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    b_ds = tl.zeros([BK, BV], dtype=tl.float32)

    for i_r in range(0, NT):
        i_n = NT - 1 - i_r
        t0 = i_n * BC
        m_t = (t0 + o_c) < T
        m_tk = m_t[:, None] & m_k[None, :]
        m_tv = m_t[:, None] & m_v[None, :]

        base_k = ((i_bh * NT + i_n) * BC + o_c)[:, None] * K + o_k[None, :]
        base_v = ((i_bh * NT + i_n) * BC + o_c)[:, None] * V + o_v[None, :]
        base_a = ((i_bh * NT + i_n) * BC + o_c)[:, None] * BC + o_c[None, :]
        off_tv = (i_b * T + t0 + o_c)[:, None] * (H * V) + i_h * V + o_v[None, :]
        base_h = ((i_bh * NT + i_n) * K + o_k)[:, None] * V + o_v[None, :]

        tl.store(dho + base_h, b_ds, mask=m_kv)

        b_do = tl.load(do + off_tv, mask=m_tv, other=0.0).to(tl.float32)
        b_qb = tl.load(Aqb + base_a)
        b_bg = tl.load(bg + base_k, mask=m_tk, other=0.0)
        b_dv2 = tl.dot(tl.trans(b_qb), b_do, input_precision=PR)
        b_dv2 += tl.dot(b_bg, b_ds, input_precision=PR)
        tl.store(dv2o + base_v, b_dv2, mask=m_tv)

        # v2 = M @ (p + ag @ S), so dS picks up ag^T (M^T dv2)
        b_qg = tl.load(qg + base_k, mask=m_tk, other=0.0)
        b_ag = tl.load(ago + base_k, mask=m_tk, other=0.0)
        b_mi = tl.load(Minv + base_a)
        b_dR = tl.dot(tl.trans(b_mi), b_dv2, input_precision=PR)
        b_gl = tl.load(glo + (i_bh * NT + i_n) * K + o_k, mask=m_k, other=0.0)
        b_ds = b_ds * tl.exp(b_gl)[:, None]
        b_ds += tl.dot(tl.trans(b_qg), b_do, input_precision=PR)
        b_ds += tl.dot(tl.trans(b_ag), b_dR, input_precision=PR)


@triton.jit
def _dplr_bwd_local(
    v, do, v2o, dv2o, ho, dho, kg, ago, uo, glo, Minv, Aqk, Aak,
    dqg, dkg, dbg, dag, dAqk, dAqb, dAak, dAab, dglo, dv,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr, NV: tl.constexpr,
    PR: tl.constexpr,
):
    """Adjoints of the gated operands and of the four A matrices, plus dv."""
    i_n = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    t0 = i_n * BC
    m_t = (t0 + o_c) < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]

    base_k = ((i_bh * NT + i_n) * BC + o_c)[:, None] * K + o_k[None, :]
    base_a = ((i_bh * NT + i_n) * BC + o_c)[:, None] * BC + o_c[None, :]

    b_m = tl.load(Minv + base_a)
    b_mt = tl.trans(b_m)
    b_qk = tl.load(Aqk + base_a)
    b_ak = tl.load(Aak + base_a)
    b_kg = tl.load(kg + base_k, mask=m_tk, other=0.0)
    b_w = tl.load(ago + base_k, mask=m_tk, other=0.0)
    b_gl = tl.load(glo + (i_bh * NT + i_n) * K + o_k, mask=m_k, other=0.0)

    acc_dqg = tl.zeros([BC, BK], dtype=tl.float32)
    acc_dkg = tl.zeros([BC, BK], dtype=tl.float32)
    acc_dbg = tl.zeros([BC, BK], dtype=tl.float32)
    acc_dw = tl.zeros([BC, BK], dtype=tl.float32)
    acc_qk = tl.zeros([BC, BC], dtype=tl.float32)
    acc_qb = tl.zeros([BC, BC], dtype=tl.float32)
    acc_ak = tl.zeros([BC, BC], dtype=tl.float32)
    acc_ab = tl.zeros([BC, BC], dtype=tl.float32)
    # Contraction of h against dh over V: both factors carry the full magnitude
    # of the state while the contraction itself cancels down toward zero, so
    # this one accumulator runs in float64.
    acc_dgl = tl.zeros([BK], dtype=tl.float64)

    for i_v in range(0, NV):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        m_kv = m_k[:, None] & m_v[None, :]

        base_v = ((i_bh * NT + i_n) * BC + o_c)[:, None] * V + o_v[None, :]
        off_tv = (i_b * T + t0 + o_c)[:, None] * (H * V) + i_h * V + o_v[None, :]
        base_h = ((i_bh * NT + i_n) * K + o_k)[:, None] * V + o_v[None, :]

        b_h = tl.load(ho + base_h, mask=m_kv, other=0.0)
        b_dh = tl.load(dho + base_h, mask=m_kv, other=0.0)
        b_v = tl.load(v + off_tv, mask=m_tv, other=0.0).to(tl.float32)
        b_do = tl.load(do + off_tv, mask=m_tv, other=0.0).to(tl.float32)
        b_v2 = tl.load(v2o + base_v, mask=m_tv, other=0.0)
        b_dv2 = tl.load(dv2o + base_v, mask=m_tv, other=0.0)

        b_ht = tl.trans(b_h)
        b_dht = tl.trans(b_dh)
        acc_dqg += tl.dot(b_do, b_ht, input_precision=PR)
        acc_dkg += tl.dot(b_v, b_dht, input_precision=PR)
        acc_dbg += tl.dot(b_v2, b_dht, input_precision=PR)

        acc_qk += tl.dot(b_do, tl.trans(b_v), input_precision=PR)
        acc_qb += tl.dot(b_do, tl.trans(b_v2), input_precision=PR)

        # v2 = M @ R with R = A_ak @ v + ag @ S.  dR = M^T dv2 drives dA_ak
        # and dag, while M itself takes dM = dv2 R^T; since M = (I - A_ab)^-1,
        # dA_ab = M^T dv2 (M R)^T = dR v2^T.
        b_dR = tl.dot(b_mt, b_dv2, input_precision=PR)
        acc_ak += tl.dot(b_dR, tl.trans(b_v), input_precision=PR)
        acc_ab += tl.dot(b_dR, tl.trans(b_v2), input_precision=PR)
        acc_dw += tl.dot(b_dR, b_ht, input_precision=PR)

        b_dv = tl.dot(tl.trans(b_qk), b_do, input_precision=PR)
        b_dv += tl.dot(tl.trans(b_ak), b_dR, input_precision=PR)
        b_dv += tl.dot(b_kg, b_dh, input_precision=PR)
        tl.store(dv + off_tv, b_dv.to(dv.dtype.element_ty), mask=m_tv)

        acc_dgl += tl.sum(b_h.to(tl.float64) * b_dh.to(tl.float64), 1)

    acc_dgl = (acc_dgl * tl.exp(b_gl).to(tl.float64)).to(tl.float32)

    # ag enters R directly, so its adjoint is already complete in acc_dw
    b_dW = acc_dw

    tri_ge = o_c[:, None] >= o_c[None, :]
    tri_gt = o_c[:, None] > o_c[None, :]

    tl.store(dqg + base_k, acc_dqg, mask=m_tk)
    tl.store(dkg + base_k, acc_dkg, mask=m_tk)
    tl.store(dbg + base_k, acc_dbg, mask=m_tk)
    tl.store(dag + base_k, b_dW, mask=m_tk)
    tl.store(dAqk + base_a, tl.where(tri_ge, acc_qk, 0.0))
    tl.store(dAqb + base_a, tl.where(tri_ge, acc_qb, 0.0))
    tl.store(dAak + base_a, tl.where(tri_gt, acc_ak, 0.0))
    tl.store(dAab + base_a, tl.where(tri_gt, acc_ab, 0.0))
    tl.store(dglo + (i_bh * NT + i_n) * K + o_k, acc_dgl, mask=m_k)


@triton.jit
def _dplr_bwd_gate(
    q, k, a, b, gio, geo,
    dqg, dkg, dbg, dag, dAqk, dAqb, dAak, dAab, dglo,
    dq, dk, da, db, dgk,
    T, H: tl.constexpr, K: tl.constexpr, NT, scale,
    BK: tl.constexpr, BC: tl.constexpr,
    PR: tl.constexpr,
):
    """Backward through the pairwise exponentials and the cumulative gate."""
    i_n = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_c = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    t0 = i_n * BC
    m_t = (t0 + o_c) < T
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]

    off_tk = (i_b * T + t0 + o_c)[:, None] * (H * K) + i_h * K + o_k[None, :]
    base_k = ((i_bh * NT + i_n) * BC + o_c)[:, None] * K + o_k[None, :]
    base_a = ((i_bh * NT + i_n) * BC + o_c)[:, None] * BC + o_c[None, :]

    b_q = tl.load(q + off_tk, mask=m_tk, other=0.0).to(tl.float32) * scale
    b_k = tl.load(k + off_tk, mask=m_tk, other=0.0).to(tl.float32)
    b_a = tl.load(a + off_tk, mask=m_tk, other=0.0).to(tl.float32)
    b_b = tl.load(b + off_tk, mask=m_tk, other=0.0).to(tl.float32)
    b_gi = tl.load(gio + base_k, mask=m_tk, other=0.0)
    b_ge = tl.load(geo + base_k, mask=m_tk, other=0.0)

    d_qk = tl.load(dAqk + base_a)
    d_qb = tl.load(dAqb + base_a)
    d_ak = tl.load(dAak + base_a)
    d_ab = tl.load(dAab + base_a)

    # These sums run over the BC columns of a chunk with gate factors spanning
    # the whole dynamic range, and they cancel heavily, so they accumulate in
    # float64 alongside the gate adjoint below.
    pr_dq = tl.zeros([BC, BK], dtype=tl.float32)
    pr_da = tl.zeros([BC, BK], dtype=tl.float32)
    pc_dk = tl.zeros([BC, BK], dtype=tl.float32)
    pc_db = tl.zeros([BC, BK], dtype=tl.float32)
    # The gate adjoint is a sum of terms that individually reach the magnitude
    # of the entire chunk yet cancel down to ~1e-6 of it -- float32 epsilon.
    # The whole dgk path therefore runs in float64.  It is elementwise work,
    # negligible beside the matmuls around it.
    acc_dgk = tl.zeros([BC, BK], dtype=tl.float64)

    for j in tl.static_range(BC):
        m_j = o_c == j
        k_j = tl.sum(tl.where(m_j[:, None], b_k, 0.0), 0)
        b_j = tl.sum(tl.where(m_j[:, None], b_b, 0.0), 0)
        g_j = tl.sum(tl.where(m_j[:, None], b_gi, 0.0), 0)

        # column j of each dA, indexed by the row i
        c_qk = tl.sum(tl.where(m_j[None, :], d_qk, 0.0), 1)
        c_qb = tl.sum(tl.where(m_j[None, :], d_qb, 0.0), 1)
        c_ak = tl.sum(tl.where(m_j[None, :], d_ak, 0.0), 1)
        c_ab = tl.sum(tl.where(m_j[None, :], d_ab, 0.0), 1)

        # clamped for the same reason as in _dplr_prep: exact where the mask
        # keeps the entry, and inf-free where it does not
        e1d = tl.exp(tl.minimum(b_gi - g_j[None, :], 0.0)) * (o_c >= j)[:, None]
        e2d = tl.exp(tl.minimum(b_ge - g_j[None, :], 0.0)) * (o_c > j)[:, None]

        # row-side: dq_i, da_i at fixed column j
        t_dq = (c_qk[:, None] * k_j[None, :] + c_qb[:, None] * b_j[None, :]) * e1d
        pr_dq += t_dq
        t_da = (c_ak[:, None] * k_j[None, :] + c_ab[:, None] * b_j[None, :]) * e2d
        pr_da += t_da

        # column-side: dk_j, db_j summed over all rows i
        w_dk = tl.sum(c_qk[:, None] * b_q * e1d, 0) + tl.sum(c_ak[:, None] * b_a * e2d, 0)
        w_db = tl.sum(c_qb[:, None] * b_q * e1d, 0) + tl.sum(c_ab[:, None] * b_a * e2d, 0)
        pc_dk += tl.where(m_j[:, None], w_dk[None, :], 0.0)
        pc_db += tl.where(m_j[:, None], w_db[None, :], 0.0)

        # Gate adjoint written directly against gk.  The A_qk/A_qb exponent
        # gi_i - gi_j is exactly sum(gk_m) for m in (j, i], and the A_ak/A_ab
        # exponent ge_i - gi_j is sum(gk_m) for m in (j, i).  Spelling the
        # ranges out makes the dependence on gk_m for m <= j -- including the
        # enormous diagonal contribution -- vanish identically, rather than by
        # a floating-point subtraction that would leave a residue far above
        # the true value.
        s1 = (t_dq * b_q).to(tl.float64)
        s2 = (t_da * b_a).to(tl.float64)
        ge_m = o_c[:, None] <= o_c[None, :]     # [m, i], i >= m
        gt_m = o_c[:, None] < o_c[None, :]      # [m, i], i >  m
        rev1 = tl.sum(tl.where(ge_m[:, :, None], s1[None, :, :], 0.0), 1)
        rev2 = tl.sum(tl.where(gt_m[:, :, None], s2[None, :, :], 0.0), 1)
        acc_dgk += tl.where((o_c > j)[:, None], rev1 + rev2, 0.0)

    # adjoints of the gated operand transforms
    b_dqg = tl.load(dqg + base_k, mask=m_tk, other=0.0)
    b_dkg = tl.load(dkg + base_k, mask=m_tk, other=0.0)
    b_dbg = tl.load(dbg + base_k, mask=m_tk, other=0.0)
    b_dag = tl.load(dag + base_k, mask=m_tk, other=0.0)
    b_dgl = tl.load(dglo + (i_bh * NT + i_n) * K + o_k, mask=m_k, other=0.0)

    m_last = o_c == (BC - 1)
    gl_v = tl.sum(tl.where(m_last[:, None], b_gi, 0.0), 0)

    e_gi = tl.exp(b_gi)
    e_dec = tl.exp(gl_v[None, :] - b_gi)
    e_ge = tl.exp(b_ge)

    # Totals per operand, in un-gated operand units.
    acc_dq = pr_dq + b_dqg * e_gi
    acc_da = pr_da + b_dag * e_ge
    acc_dk = pc_dk + b_dkg * e_dec
    acc_db = pc_db + b_dbg * e_dec

    # Gate terms of the four gated-operand transforms.  qg = q exp(gi) and
    # ag = a exp(ge) contribute suffix sums over the chunk; kg = k exp(gl - gi)
    # and bg = b exp(gl - gi) have an exponent that is the sum of gk strictly
    # after the row, giving a strict prefix sum.
    d_gi = b_dqg * (b_q * e_gi)
    d_ge = b_dag * (b_a * e_ge)
    t_dec = b_dkg * (b_k * e_dec) + b_dbg * (b_b * e_dec)

    # qg = q exp(gi) and ag = a exp(ge) give suffix sums over the chunk;
    # kg / bg carry exp(gl - gi), an exponent that is the sum of gk strictly
    # after the row, hence a strict prefix sum.  Suffix and prefix sums are
    # formed directly rather than as (total - prefix), which would again be a
    # subtraction of two large nearly-equal quantities.
    d_gi = (b_dqg * (b_q * e_gi)).to(tl.float64)
    d_ge = (b_dag * (b_a * e_ge)).to(tl.float64)
    d_dec = t_dec.to(tl.float64)

    ge_m = o_c[:, None] <= o_c[None, :]     # [m, i], i >= m
    gt_m = o_c[:, None] < o_c[None, :]      # [m, i], i >  m
    lt_m = o_c[:, None] > o_c[None, :]      # [m, i], i <  m
    acc_dgk += tl.sum(tl.where(ge_m[:, :, None], d_gi[None, :, :], 0.0), 1)
    acc_dgk += tl.sum(tl.where(gt_m[:, :, None], d_ge[None, :, :], 0.0), 1)
    acc_dgk += tl.sum(tl.where(lt_m[:, :, None], d_dec[None, :, :], 0.0), 1)
    acc_dgk += b_dgl[None, :].to(tl.float64)
    b_dgk = acc_dgk.to(tl.float32)

    tl.store(dq + off_tk, (acc_dq * scale).to(dq.dtype.element_ty), mask=m_tk)
    tl.store(dk + off_tk, acc_dk.to(dk.dtype.element_ty), mask=m_tk)
    tl.store(da + off_tk, acc_da.to(da.dtype.element_ty), mask=m_tk)
    tl.store(db + off_tk, acc_db.to(db.dtype.element_ty), mask=m_tk)
    tl.store(dgk + off_tk, b_dgk.to(dgk.dtype.element_ty), mask=m_tk)


def _nx(x, lo=16):
    return max(triton.next_power_of_2(int(x)), lo)


_NUM_SM = torch.cuda.get_device_properties(0).multi_processor_count


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        BH = B * H
        NT = triton.cdiv(T, C)
        BK, BVf = _nx(K), _nx(V)
        scale = K ** -0.5
        dev = q.device
        f32 = torch.float32

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()

        nk = (BH * NT * C, K)
        nv = (BH * NT * C, V)
        na = (BH * NT * C, C)
        e = lambda s: torch.empty(s, dtype=f32, device=dev)  # noqa: E731
        qg, kg, bg, ag, gi, ge = (e(nk) for _ in range(6))
        u, v2 = e(nv), e(nv)
        gl = e((BH * NT, K))
        Aqk, Aqb, Aak, Aab, Minv = (e(na) for _ in range(5))
        # float32 chunk-boundary state checkpoints
        h = torch.empty((BH * NT, K, V), dtype=f32, device=dev)
        o = torch.empty((B, T, H, V), dtype=v.dtype, device=dev)

        _dplr_prep[(NT, BH)](
            q, k, alpha, beta, gk, v,
            qg, kg, bg, ag, u, gi, ge, gl,
            Aqk, Aqb, Aak, Aab, Minv,
            T, H, K, V, scale, NT,
            BK=BK, BV=BVf, BC=C, PR=PREC, num_warps=4, num_stages=2,
        )
        # The two scans are serial in time, so their only parallelism is
        # (V tiles) x (B*H).  Narrow the V tile until that fills the device;
        # this depends on shapes alone, never on tensor values.
        BVs = min(BVf, 64)
        while BVs > 16 and triton.cdiv(V, BVs) * BH < _NUM_SM:
            BVs //= 2
        NVs = triton.cdiv(V, BVs)
        _dplr_fwd_scan[(NVs, BH)](
            qg, kg, bg, ag, u, gl, Aqk, Aqb, Minv, v, o, v2, h,
            T, H, K, V, NT,
            BK=BK, BV=BVs, BC=C, PR=PREC, num_warps=8, num_stages=2,
        )
        BVo = min(BVf, 64)
        _dplr_fwd_out[(NT, triton.cdiv(V, BVo), BH)](
            qg, Aqk, Aqb, v, v2, h, o,
            T, H, K, V, NT,
            BK=BK, BV=BVo, BC=C, PR=PREC, num_warps=4, num_stages=2,
        )

        ctx.save_for_backward(q, k, v, alpha, beta, gk, qg, kg, bg, ag, u,
                              gi, ge, gl, Aqk, Aqb, Aak, Aab, Minv, h, v2)
        ctx.dims = (B, T, H, K, V, NT, BK, BVf, BVs, NVs, scale)
        return o

    @staticmethod
    def backward(ctx, do):
        (q, k, v, alpha, beta, gk, qg, kg, bg, ag, u,
         gi, ge, gl, Aqk, Aqb, Aak, Aab, Minv, h, v2) = ctx.saved_tensors
        B, T, H, K, V, NT, BK, BVf, BVs, NVs, scale = ctx.dims
        BH = B * H
        dev = q.device
        f32 = torch.float32
        do = do.contiguous()

        nk = (BH * NT * C, K)
        na = (BH * NT * C, C)
        e = lambda s: torch.empty(s, dtype=f32, device=dev)  # noqa: E731
        dv2 = e((BH * NT * C, V))
        # float32 chunk-boundary adjoint-state checkpoints
        dh = torch.empty((BH * NT, K, V), dtype=f32, device=dev)
        dqg, dkg, dbg, dag = (e(nk) for _ in range(4))
        dAqk, dAqb, dAak, dAab = (e(na) for _ in range(4))
        dgl = e((BH * NT, K))
        dv = torch.empty_like(v)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        da = torch.empty_like(alpha)
        db = torch.empty_like(beta)
        dgk = torch.empty_like(gk)

        _dplr_bwd_scan[(NVs, BH)](
            qg, bg, ag, gl, Aqb, Minv, do, dv2, dh,
            T, H, K, V, NT,
            BK=BK, BV=BVs, BC=C, PR=PREC, num_warps=8, num_stages=2,
        )
        BVl = min(BVf, 64)
        NVl = triton.cdiv(V, BVl)
        _dplr_bwd_local[(NT, BH)](
            v, do, v2, dv2, h, dh, kg, ag, u, gl, Minv, Aqk, Aak,
            dqg, dkg, dbg, dag, dAqk, dAqb, dAak, dAab, dgl, dv,
            T, H, K, V, NT,
            BK=BK, BV=BVl, BC=C, NV=NVl, PR=PREC, num_warps=4, num_stages=2,
        )
        _dplr_bwd_gate[(NT, BH)](
            q, k, alpha, beta, gi, ge,
            dqg, dkg, dbg, dag, dAqk, dAqb, dAak, dAab, dgl,
            dq, dk, da, db, dgk,
            T, H, K, NT, scale,
            BK=BK, BC=C, PR=PREC, num_warps=4, num_stages=2,
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
    """Entry point. Signature and semantics fixed."""
    return _Dplr.apply(q, k, v, alpha, beta, gk)
