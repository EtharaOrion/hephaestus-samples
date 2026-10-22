"""Chunk-parallel DPLR (diagonal-plus-low-rank) generalized delta rule.

Per (batch, head) the recurrence

    S_t = diag(exp(gk_t)) S_{t-1} + k_t v_t^T + beta_t (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))

is reorganised into chunks of BT steps so that all intra-chunk work becomes
dense matrix products and only one [K, V] state crosses a chunk boundary.

Gate factorisation.  With gi the per-K-channel inclusive cumulative sum of gk
inside the chunk, ge = gi - gk its exclusive counterpart and gl the chunk
total, define

    qt = q e^{gi} / sqrt(K)      at = alpha e^{ge}
    kt = k e^{-gi}               bt = beta  e^{-gi}
    kh = k e^{gl - gi}           bh = beta  e^{gl - gi}

Every pairwise product that survives the causal mask carries a non-positive
exponent, so the four intra-chunk matrices of the DPLR cascade

    A_ab = tril(at bt^T, -1)     A_ak = tril(at kt^T, -1)
    A_qk = tril(qt kt^T,  0)     A_qb = tril(qt bt^T,  0)

stay bounded however negative the gates are.  A_ab is strictly lower
triangular, hence nilpotent, so the intra-chunk alpha-beta erasure operator is
inverted exactly by the doubling product

    M = (I - A_ab)^{-1} = prod_j (I + A_ab^{2^j}),   j < log2(BT)

which is a finite exact identity, not a truncated series.  Then

    U     = M (A_ak v + at S)                  (pseudo-values of the chunk)
    S_nxt = e^{gl} S + kh^T v + bh^T U
    o     = qt S + A_qk v + A_qb U

The backward pass mirrors this with the transposed operator M^T and a reverse
carry Z, giving dq, dk, dv, dalpha, dbeta directly; dgk follows from the
gate reparametrisation, where the loss depends on gk only through the six
scaled operands above, so

    dgk_t = sum_{s>=t} (q.dq - k.dk - beta.dbeta)_s + sum_{s>t} (alpha.dalpha)_s

The chunk-boundary states S and Z, their per-chunk checkpoints, and every
scratch buffer that carries them are float32 for the whole of their life.
"""

import torch
import triton
import triton.language as tl

BT = 16
LOG_BT = 4
CAP = 60.0
PREC = "ieee"


def _p2(x):
    return max(16, triton.next_power_of_2(x))


@triton.jit
def dplr_prep_kernel(
    q_ptr, k_ptr, a_ptr, b_ptr, g_ptr,
    qt_ptr, kt_ptr, bt_ptr, at_ptr, kh_ptr, bh_ptr, gi_ptr, ge_ptr, gl_ptr,
    T, H, K, scale,
    BTC: tl.constexpr, BK: tl.constexpr, CAPV: tl.constexpr,
):
    """Per-chunk cumulative gate and the six gate-scaled operand tiles."""
    nt = tl.program_id(0)
    bh = tl.program_id(1)
    NT = tl.num_programs(0)
    h = bh % H
    b = bh // H

    i = tl.arange(0, BTC)
    c = tl.arange(0, BK)
    t = nt * BTC + i
    mc = c < K
    mrow = i[:, None] >= 0
    mload = (t < T)[:, None] & mc[None, :]
    mst = mrow & mc[None, :]

    off = ((b.to(tl.int64) * T + t) * H + h)[:, None] * K + c[None, :]
    q = tl.load(q_ptr + off, mask=mload, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + off, mask=mload, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + off, mask=mload, other=0.0).to(tl.float32)
    bb = tl.load(b_ptr + off, mask=mload, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + off, mask=mload, other=0.0).to(tl.float32)

    gi = tl.cumsum(g, axis=0)
    ge = gi - g
    gl = tl.sum(g, axis=0)

    # Gate reference taken at the middle of the chunk rather than at its start.
    # Only differences of gi ever survive the causal mask, so any per-channel
    # offset r cancels in every matched pair; centring at r = gl/2 splits the
    # exponent range symmetrically and halves the largest magnitude reached by
    # e^{r-gi}, which is what overflows float32 when the chunk decay is deep.
    r = gl * 0.5
    fwd = tl.exp(tl.minimum(gi - r[None, :], CAPV))
    bwdf = tl.exp(tl.minimum(r[None, :] - gi, CAPV))
    dec = tl.exp(tl.minimum(gl[None, :] - gi, CAPV))

    n = bh.to(tl.int64) * NT + nt
    so = n * (BTC * K) + i[:, None] * K + c[None, :]
    tl.store(qt_ptr + so, q * scale * fwd, mask=mst)
    tl.store(kt_ptr + so, k * bwdf, mask=mst)
    tl.store(bt_ptr + so, bb * bwdf, mask=mst)
    tl.store(at_ptr + so, a * tl.exp(tl.minimum(ge - r[None, :], CAPV)), mask=mst)
    tl.store(kh_ptr + so, k * dec, mask=mst)
    tl.store(bh_ptr + so, bb * dec, mask=mst)
    tl.store(gi_ptr + so, gi, mask=mst)
    tl.store(ge_ptr + so, ge, mask=mst)
    tl.store(gl_ptr + n * K + c, gl, mask=mc)


@triton.jit
def dplr_wy_kernel(
    at_ptr, kt_ptr, bt_ptr, v_ptr, m_ptr, wa_ptr, u_ptr,
    T, H, K, V,
    BTC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    LB: tl.constexpr, PR: tl.constexpr,
):
    """Exact inverse of the intra-chunk alpha-beta erasure operator plus the
    local (state-independent) part of the pseudo-value block."""
    nt = tl.program_id(0)
    bh = tl.program_id(1)
    NT = tl.num_programs(0)
    h = bh % H
    b = bh // H

    i = tl.arange(0, BTC)
    j = tl.arange(0, BTC)
    c = tl.arange(0, BK)
    mc = c < K
    mrow = i[:, None] >= 0
    mst = mrow & mc[None, :]
    n = bh.to(tl.int64) * NT + nt
    so = n * (BTC * K) + i[:, None] * K + c[None, :]

    at = tl.load(at_ptr + so, mask=mst, other=0.0)
    bt = tl.load(bt_ptr + so, mask=mst, other=0.0)
    kt = tl.load(kt_ptr + so, mask=mst, other=0.0)

    slow = i[:, None] > j[None, :]
    A = tl.where(slow, tl.dot(at, tl.trans(bt), input_precision=PR), 0.0)
    Aak = tl.where(slow, tl.dot(at, tl.trans(kt), input_precision=PR), 0.0)

    # (I - A_ab)^{-1} by exact forward substitution, one row at a time:
    #   M[r,:] = e_r + sum_{s<r} A[r,s] M[s,:]
    # A_ab is strictly lower triangular so this terminates after BTC rows and is
    # algebraically exact.  The repeated-squaring form is exact too, but it
    # squares a matrix whose entries already span several decades on deep-decay
    # channels, and the rounding that introduces propagates into the chunk state.
    M = (i[:, None] == j[None, :]).to(tl.float32)
    for r in tl.static_range(1, BTC):
        Ar = tl.sum(tl.where(i[:, None] == r, A, 0.0), axis=0)
        contrib = tl.sum(Ar[:, None] * M, axis=0)
        M = M + tl.where(i[:, None] == r, contrib[None, :], 0.0)

    tl.store(wa_ptr + so, tl.dot(M, at, input_precision=PR), mask=mst)
    tl.store(m_ptr + n * (BTC * BTC) + i[:, None] * BTC + j[None, :], M)

    t = nt * BTC + i
    mt = t < T
    for vo in range(0, V, BV):
        vc = vo + tl.arange(0, BV)
        mv = vc < V
        voff = ((b.to(tl.int64) * T + t) * H + h)[:, None] * V + vc[None, :]
        v = tl.load(v_ptr + voff, mask=mt[:, None] & mv[None, :], other=0.0).to(tl.float32)
        # Solve (I - A_ab) U = A_ak v by the same substitution rather than
        # multiplying by the assembled inverse: forming M A_ak first rounds a
        # product of two badly scaled triangular factors before it ever meets v.
        Uloc = tl.dot(Aak, v, input_precision=PR)
        for r in tl.static_range(1, BTC):
            Arv = tl.sum(tl.where(i[:, None] == r, A, 0.0), axis=0)
            contribv = tl.sum(Arv[:, None] * Uloc, axis=0)
            Uloc = Uloc + tl.where(i[:, None] == r, contribv[None, :], 0.0)
        tl.store(u_ptr + n * (BTC * V) + i[:, None] * V + vc[None, :],
                 Uloc, mask=mrow & mv[None, :])


@triton.jit
def dplr_scan_kernel(
    wa_ptr, u_ptr, kh_ptr, bh_ptr, gl_ptr, v_ptr, h_ptr,
    T, H, K, V, NT,
    BTC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PR: tl.constexpr,
):
    """Forward chunk-boundary carry.  The state stays in float32 registers for
    the whole scan and is checkpointed to a float32 buffer at every chunk."""
    bh = tl.program_id(0)
    vt = tl.program_id(1)
    h = bh % H
    b = bh // H

    i = tl.arange(0, BTC)
    c = tl.arange(0, BK)
    vc = vt * BV + tl.arange(0, BV)
    mc = c < K
    mv = vc < V
    mrow = i[:, None] >= 0
    S = tl.zeros([BK, BV], dtype=tl.float32)

    for nt in range(0, NT):
        n = bh.to(tl.int64) * NT + nt
        so = n * (BTC * K) + i[:, None] * K + c[None, :]
        uo = n * (BTC * V) + i[:, None] * V + vc[None, :]
        t = nt * BTC + i
        mt = t < T

        tl.store(h_ptr + n * (BK * V) + c[:, None] * V + vc[None, :], S,
                 mask=mc[:, None] & mv[None, :])

        wa = tl.load(wa_ptr + so, mask=mrow & mc[None, :], other=0.0)
        u = tl.load(u_ptr + uo, mask=mrow & mv[None, :], other=0.0)
        gl = tl.load(gl_ptr + n * K + c, mask=mc, other=0.0)
        er = tl.exp(gl * 0.5)[:, None]
        u += tl.dot(wa, S * er, input_precision=PR)
        tl.store(u_ptr + uo, u, mask=mrow & mv[None, :])

        voff = ((b.to(tl.int64) * T + t) * H + h)[:, None] * V + vc[None, :]
        v = tl.load(v_ptr + voff, mask=mt[:, None] & mv[None, :], other=0.0).to(tl.float32)
        kh = tl.load(kh_ptr + so, mask=mrow & mc[None, :], other=0.0)
        bb = tl.load(bh_ptr + so, mask=mrow & mc[None, :], other=0.0)

        S = S * tl.exp(gl)[:, None] + tl.dot(tl.trans(kh), v, input_precision=PR) \
            + tl.dot(tl.trans(bb), u, input_precision=PR)


@triton.jit
def dplr_out_kernel(
    qt_ptr, kt_ptr, bt_ptr, v_ptr, u_ptr, h_ptr, gl_ptr, o_ptr,
    T, H, K, V,
    BTC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PR: tl.constexpr,
):
    nt = tl.program_id(0)
    bh = tl.program_id(1)
    vt = tl.program_id(2)
    NT = tl.num_programs(0)
    h = bh % H
    b = bh // H

    i = tl.arange(0, BTC)
    j = tl.arange(0, BTC)
    c = tl.arange(0, BK)
    vc = vt * BV + tl.arange(0, BV)
    mc = c < K
    mv = vc < V
    mrow = i[:, None] >= 0
    n = bh.to(tl.int64) * NT + nt
    so = n * (BTC * K) + i[:, None] * K + c[None, :]

    qt = tl.load(qt_ptr + so, mask=mrow & mc[None, :], other=0.0)
    kt = tl.load(kt_ptr + so, mask=mrow & mc[None, :], other=0.0)
    bt = tl.load(bt_ptr + so, mask=mrow & mc[None, :], other=0.0)
    hs = tl.load(h_ptr + n * (BK * V) + c[:, None] * V + vc[None, :],
                 mask=mc[:, None] & mv[None, :], other=0.0)
    glo = tl.load(gl_ptr + n * K + c, mask=mc, other=0.0)
    hs = hs * tl.exp(glo * 0.5)[:, None]
    u = tl.load(u_ptr + n * (BTC * V) + i[:, None] * V + vc[None, :],
                mask=mrow & mv[None, :], other=0.0)

    t = nt * BTC + i
    mt = t < T
    voff = ((b.to(tl.int64) * T + t) * H + h)[:, None] * V + vc[None, :]
    mvt = mt[:, None] & mv[None, :]
    v = tl.load(v_ptr + voff, mask=mvt, other=0.0).to(tl.float32)

    low = i[:, None] >= j[None, :]
    Aqk = tl.where(low, tl.dot(qt, tl.trans(kt), input_precision=PR), 0.0)
    Aqb = tl.where(low, tl.dot(qt, tl.trans(bt), input_precision=PR), 0.0)
    o = tl.dot(qt, hs, input_precision=PR)
    o += tl.dot(Aqk, v, input_precision=PR)
    o += tl.dot(Aqb, u, input_precision=PR)
    tl.store(o_ptr + voff, o.to(o_ptr.dtype.element_ty), mask=mvt)


@triton.jit
def dplr_bprep_kernel(
    m_ptr, bh_ptr, qt_ptr, bt_ptr, do_ptr, wb_ptr, wl_ptr,
    T, H, K, V,
    BTC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PR: tl.constexpr,
):
    """Transposed WY operator and the local part of the adjoint block W."""
    nt = tl.program_id(0)
    bh = tl.program_id(1)
    NT = tl.num_programs(0)
    h = bh % H
    b = bh // H

    i = tl.arange(0, BTC)
    j = tl.arange(0, BTC)
    c = tl.arange(0, BK)
    mc = c < K
    mrow = i[:, None] >= 0
    mst = mrow & mc[None, :]
    n = bh.to(tl.int64) * NT + nt
    so = n * (BTC * K) + i[:, None] * K + c[None, :]

    Mt = tl.trans(tl.load(m_ptr + n * (BTC * BTC) + i[:, None] * BTC + j[None, :]))
    bb = tl.load(bh_ptr + so, mask=mst, other=0.0)
    tl.store(wb_ptr + so, tl.dot(Mt, bb, input_precision=PR), mask=mst)

    qt = tl.load(qt_ptr + so, mask=mst, other=0.0)
    bt = tl.load(bt_ptr + so, mask=mst, other=0.0)
    low = i[:, None] >= j[None, :]
    Aqb_t = tl.trans(tl.where(low, tl.dot(qt, tl.trans(bt), input_precision=PR), 0.0))
    MA = tl.dot(Mt, Aqb_t, input_precision=PR)

    t = nt * BTC + i
    mt = t < T
    for vo in range(0, V, BV):
        vc = vo + tl.arange(0, BV)
        mv = vc < V
        voff = ((b.to(tl.int64) * T + t) * H + h)[:, None] * V + vc[None, :]
        do = tl.load(do_ptr + voff, mask=mt[:, None] & mv[None, :], other=0.0).to(tl.float32)
        tl.store(wl_ptr + n * (BTC * V) + i[:, None] * V + vc[None, :],
                 tl.dot(MA, do, input_precision=PR), mask=mrow & mv[None, :])


@triton.jit
def dplr_dscan_kernel(
    wb_ptr, wl_ptr, qt_ptr, at_ptr, gl_ptr, do_ptr, z_ptr,
    T, H, K, V, NT,
    BTC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PR: tl.constexpr,
):
    """Reverse chunk-boundary carry; the adjoint state is float32 throughout."""
    bh = tl.program_id(0)
    vt = tl.program_id(1)
    h = bh % H
    b = bh // H

    i = tl.arange(0, BTC)
    c = tl.arange(0, BK)
    vc = vt * BV + tl.arange(0, BV)
    mc = c < K
    mv = vc < V
    mrow = i[:, None] >= 0
    Z = tl.zeros([BK, BV], dtype=tl.float32)

    for r in range(0, NT):
        nt = NT - 1 - r
        n = bh.to(tl.int64) * NT + nt
        so = n * (BTC * K) + i[:, None] * K + c[None, :]
        wo = n * (BTC * V) + i[:, None] * V + vc[None, :]
        t = nt * BTC + i
        mt = t < T

        tl.store(z_ptr + n * (BK * V) + c[:, None] * V + vc[None, :], Z,
                 mask=mc[:, None] & mv[None, :])

        wb = tl.load(wb_ptr + so, mask=mrow & mc[None, :], other=0.0)
        W = tl.load(wl_ptr + wo, mask=mrow & mv[None, :], other=0.0)
        W += tl.dot(wb, Z, input_precision=PR)
        tl.store(wl_ptr + wo, W, mask=mrow & mv[None, :])

        voff = ((b.to(tl.int64) * T + t) * H + h)[:, None] * V + vc[None, :]
        do = tl.load(do_ptr + voff, mask=mt[:, None] & mv[None, :], other=0.0).to(tl.float32)
        qt = tl.load(qt_ptr + so, mask=mrow & mc[None, :], other=0.0)
        at = tl.load(at_ptr + so, mask=mrow & mc[None, :], other=0.0)
        gl = tl.load(gl_ptr + n * K + c, mask=mc, other=0.0)

        erb = tl.exp(gl * 0.5)[None, :]
        Z = Z * tl.exp(gl)[:, None] \
            + tl.dot(tl.trans(qt * erb), do, input_precision=PR) \
            + tl.dot(tl.trans(at * erb), W, input_precision=PR)


@triton.jit
def dplr_dgate_kernel(
    qt_ptr, kt_ptr, bt_ptr, at_ptr, gl_ptr,
    v_ptr, u_ptr, w_ptr, do_ptr, h_ptr, z_ptr, dg_ptr,
    T, H, K, V,
    BTC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, PR: tl.constexpr,
):
    """Gate gradient as the direct inner product

        dgk_m[c] = <Y_m[c,:], X_m[c,:]>,   Y_m = e^{gi_m} dL/dS_m,
                                           X_m = e^{-gi_{m-1}} S_{m-1}

    The two exponentials cancel analytically, so the quantity is formed on one
    scale rather than as a difference of large partial sums.  Y and X are
    carried explicitly across the chunk in float32 registers: Y is seeded with
    its full suffix (two dense matmuls) and walked down, X starts at the chunk
    entry state and is walked up, each by one rank-1 update per step.  This
    keeps the reduction over V a single sum of same-scale terms, which is what
    the deep-decay channels need.
    """
    nt = tl.program_id(0)
    bh = tl.program_id(1)
    NT = tl.num_programs(0)
    h = bh % H
    b = bh // H

    i = tl.arange(0, BTC)
    j = tl.arange(0, BTC)
    c = tl.arange(0, BK)
    vc = tl.arange(0, BV)
    mc = c < K
    mv = vc < V
    mrow = i[:, None] >= 0
    n = bh.to(tl.int64) * NT + nt
    so = n * (BTC * K) + i[:, None] * K + c[None, :]
    t = nt * BTC + i
    mt = t < T

    qt = tl.load(qt_ptr + so, mask=mrow & mc[None, :], other=0.0)
    kt = tl.load(kt_ptr + so, mask=mrow & mc[None, :], other=0.0)
    bt = tl.load(bt_ptr + so, mask=mrow & mc[None, :], other=0.0)
    at = tl.load(at_ptr + so, mask=mrow & mc[None, :], other=0.0)
    gl = tl.load(gl_ptr + n * K + c, mask=mc, other=0.0)
    er = tl.exp(gl * 0.5)[:, None]

    SH = (j[None, :] == i[:, None] + 1).to(tl.float32)
    yB = tl.dot(SH, at, input_precision=PR)

    voff = ((b.to(tl.int64) * T + t) * H + h)[:, None] * V + vc[None, :]
    mvt = mt[:, None] & mv[None, :]
    v = tl.load(v_ptr + voff, mask=mvt, other=0.0).to(tl.float32)
    do = tl.load(do_ptr + voff, mask=mvt, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + n * (BTC * V) + i[:, None] * V + vc[None, :],
                mask=mrow & mv[None, :], other=0.0)
    W = tl.load(w_ptr + n * (BTC * V) + i[:, None] * V + vc[None, :],
                mask=mrow & mv[None, :], other=0.0)
    rB = tl.dot(SH, W, input_precision=PR)

    Xe = tl.load(h_ptr + n * (BK * V) + c[:, None] * V + vc[None, :],
                 mask=mc[:, None] & mv[None, :], other=0.0) * er
    Ye = tl.load(z_ptr + n * (BK * V) + c[:, None] * V + vc[None, :],
                 mask=mc[:, None] & mv[None, :], other=0.0) * er

    # Expanding <Ye + suffix, Xe + prefix> keeps each of the four pieces as a
    # cumulative sum of dense tile products.  Walking Y down by subtracting its
    # own terms instead would cancel: Y_m is genuinely tiny next to the full
    # suffix once the chunk decay is deep, and float32 has no digits to spare.
    excl = (i[:, None] > j[None, :]).to(tl.float32)   # strict prefix
    suf = (i[:, None] <= j[None, :]).to(tl.float32)   # inclusive suffix
    gtm = j[None, :] > i[:, None]
    ltm = i[:, None] > j[None, :]

    dgk = tl.sum(Ye * Xe, axis=1)[None, :]
    pre = kt * tl.trans(tl.dot(Ye, tl.trans(v), input_precision=PR)) \
        + bt * tl.trans(tl.dot(Ye, tl.trans(u), input_precision=PR))
    sfx = qt * tl.trans(tl.dot(Xe, tl.trans(do), input_precision=PR)) \
        + yB * tl.trans(tl.dot(Xe, tl.trans(rB), input_precision=PR))
    dgk += tl.dot(excl, pre, input_precision=PR)
    dgk += tl.dot(suf, sfx, input_precision=PR)

    # <suffix Y, prefix X>: the double sum over the rectangle {j >= m, l < m}.
    # It is accumulated by growing the prefix operand one step at a time with a
    # rank-1 update and reducing the suffix directly, so only like-signed
    # partial sums are combined.  Writing the same quantity as an incremental
    # difference of two rectangle sums would subtract two large nearly-equal
    # numbers at every step and lose the result where the decay is deep.
    for pi in tl.static_range(2):
        yP = qt if pi == 0 else yB
        rP = do if pi == 0 else rB
        for qi in tl.static_range(2):
            xQ = kt if qi == 0 else bt
            sQ = v if qi == 0 else u
            D = tl.dot(rP, tl.trans(sQ), input_precision=PR)
            G = tl.zeros([BTC, BK], dtype=tl.float32)
            for mm in tl.static_range(BTC):
                fm = tl.sum(tl.where((i >= mm)[:, None], yP * G, 0.0), axis=0)
                dgk += tl.where(i[:, None] == mm, fm[None, :], 0.0)
                colm = tl.sum(tl.where(j[None, :] == mm, D, 0.0), axis=1)
                xrow = tl.sum(tl.where(i[:, None] == mm, xQ, 0.0), axis=0)
                G += colm[:, None] * xrow[None, :]

    ooff = ((b.to(tl.int64) * T + t) * H + h)[:, None] * K + c[None, :]
    tl.store(dg_ptr + ooff, dgk, mask=mt[:, None] & mc[None, :])


@triton.jit
def dplr_dbwd_kernel(
    qt_ptr, kt_ptr, bt_ptr, at_ptr, kh_ptr, bh_ptr,
    gi_ptr, ge_ptr, gl_ptr, v_ptr, u_ptr, w_ptr, do_ptr, h_ptr, z_ptr,
    dq_ptr, dk_ptr, dv_ptr, da_ptr, db_ptr,
    T, H, K, V, scale,
    BTC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    CAPV: tl.constexpr, PR: tl.constexpr,
):
    """The five non-gate gradients for one chunk."""
    nt = tl.program_id(0)
    bh = tl.program_id(1)
    NT = tl.num_programs(0)
    h = bh % H
    b = bh // H

    i = tl.arange(0, BTC)
    j = tl.arange(0, BTC)
    c = tl.arange(0, BK)
    mc = c < K
    mrow = i[:, None] >= 0
    mst = mrow & mc[None, :]
    n = bh.to(tl.int64) * NT + nt
    so = n * (BTC * K) + i[:, None] * K + c[None, :]
    t = nt * BTC + i
    mt = t < T
    m = mt[:, None] & mc[None, :]

    qt = tl.load(qt_ptr + so, mask=mst, other=0.0)
    kt = tl.load(kt_ptr + so, mask=mst, other=0.0)
    bt = tl.load(bt_ptr + so, mask=mst, other=0.0)
    at = tl.load(at_ptr + so, mask=mst, other=0.0)
    kh = tl.load(kh_ptr + so, mask=mst, other=0.0)
    bhh = tl.load(bh_ptr + so, mask=mst, other=0.0)
    gi = tl.load(gi_ptr + so, mask=mst, other=0.0)
    ge = tl.load(ge_ptr + so, mask=mst, other=0.0)
    gl = tl.load(gl_ptr + n * K + c, mask=mc, other=0.0)

    low = i[:, None] >= j[None, :]
    slow = i[:, None] > j[None, :]
    Aqk_t = tl.trans(tl.where(low, tl.dot(qt, tl.trans(kt), input_precision=PR), 0.0))
    Aak_t = tl.trans(tl.where(slow, tl.dot(at, tl.trans(kt), input_precision=PR), 0.0))
    dec = tl.exp(tl.minimum(gl[None, :] - gi, CAPV))
    # Undo the chunk-centring offset when pairing a centred operand with the
    # un-centred boundary state, and when emitting gradients in input space.
    er = tl.exp(gl * 0.5)[None, :]

    dq = tl.zeros([BTC, BK], dtype=tl.float32)
    dk = tl.zeros([BTC, BK], dtype=tl.float32)
    da = tl.zeros([BTC, BK], dtype=tl.float32)
    db = tl.zeros([BTC, BK], dtype=tl.float32)
    dks = tl.zeros([BTC, BK], dtype=tl.float32)
    dbs = tl.zeros([BTC, BK], dtype=tl.float32)
    dqs = tl.zeros([BTC, BK], dtype=tl.float32)
    das = tl.zeros([BTC, BK], dtype=tl.float32)

    for vo in range(0, V, BV):
        vc = vo + tl.arange(0, BV)
        mv = vc < V
        voff = ((b.to(tl.int64) * T + t) * H + h)[:, None] * V + vc[None, :]
        mvt = mt[:, None] & mv[None, :]
        v = tl.load(v_ptr + voff, mask=mvt, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + voff, mask=mvt, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + n * (BTC * V) + i[:, None] * V + vc[None, :],
                    mask=mrow & mv[None, :], other=0.0)
        W = tl.load(w_ptr + n * (BTC * V) + i[:, None] * V + vc[None, :],
                    mask=mrow & mv[None, :], other=0.0)
        hs = tl.load(h_ptr + n * (BK * V) + c[:, None] * V + vc[None, :],
                     mask=mc[:, None] & mv[None, :], other=0.0)
        zs = tl.load(z_ptr + n * (BK * V) + c[:, None] * V + vc[None, :],
                     mask=mc[:, None] & mv[None, :], other=0.0)

        dov = tl.where(low, tl.dot(do, tl.trans(v), input_precision=PR), 0.0)
        dou = tl.where(low, tl.dot(do, tl.trans(u), input_precision=PR), 0.0)
        wv = tl.where(slow, tl.dot(W, tl.trans(v), input_precision=PR), 0.0)
        wu = tl.where(slow, tl.dot(W, tl.trans(u), input_precision=PR), 0.0)

        dqs += tl.dot(do, tl.trans(hs), input_precision=PR)
        dq += tl.dot(dov, kt, input_precision=PR)
        dq += tl.dot(dou, bt, input_precision=PR)

        das += tl.dot(W, tl.trans(hs), input_precision=PR)
        da += tl.dot(wv, kt, input_precision=PR)
        da += tl.dot(wu, bt, input_precision=PR)

        dvv = tl.dot(Aqk_t, do, input_precision=PR)
        dvv += tl.dot(Aak_t, W, input_precision=PR)
        dvv += tl.dot(kh, zs, input_precision=PR)
        tl.store(dv_ptr + voff, dvv.to(dv_ptr.dtype.element_ty), mask=mvt)

        dk += tl.dot(tl.trans(dov), qt, input_precision=PR)
        dk += tl.dot(tl.trans(wv), at, input_precision=PR)
        dks += tl.dot(v, tl.trans(zs), input_precision=PR)
        db += tl.dot(tl.trans(dou), qt, input_precision=PR)
        db += tl.dot(tl.trans(wu), at, input_precision=PR)
        dbs += tl.dot(u, tl.trans(zs), input_precision=PR)

    r = gl[None, :] * 0.5
    fwdf = tl.exp(tl.minimum(gi - r, CAPV))
    negf = tl.exp(tl.minimum(r - gi, CAPV))
    ooff = ((b.to(tl.int64) * T + t) * H + h)[:, None] * K + c[None, :]
    tl.store(dq_ptr + ooff,
             ((dq * fwdf + dqs * tl.exp(gi)) * scale).to(dq_ptr.dtype.element_ty), mask=m)
    tl.store(dk_ptr + ooff, (dk * negf + dks * dec).to(dk_ptr.dtype.element_ty), mask=m)
    tl.store(da_ptr + ooff,
             (da * tl.exp(tl.minimum(ge - r, CAPV))
              + das * tl.exp(ge)).to(da_ptr.dtype.element_ty), mask=m)
    tl.store(db_ptr + ooff, (db * negf + dbs * dec).to(db_ptr.dtype.element_ty), mask=m)



def _fwd(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    dev = q.device
    NT = triton.cdiv(T, BT)
    BK = _p2(K)
    N = B * H * NT
    scale = K ** -0.5

    f32 = torch.float32
    sh = (N, BT, K)
    qt = torch.empty(sh, device=dev, dtype=f32)
    kt = torch.empty(sh, device=dev, dtype=f32)
    bt = torch.empty(sh, device=dev, dtype=f32)
    at = torch.empty(sh, device=dev, dtype=f32)
    kh = torch.empty(sh, device=dev, dtype=f32)
    bh = torch.empty(sh, device=dev, dtype=f32)
    gi = torch.empty(sh, device=dev, dtype=f32)
    ge = torch.empty(sh, device=dev, dtype=f32)
    gl = torch.empty((N, K), device=dev, dtype=f32)

    dplr_prep_kernel[(NT, B * H)](
        q, k, alpha, beta, gk, qt, kt, bt, at, kh, bh, gi, ge, gl,
        T, H, K, scale, BTC=BT, BK=BK, CAPV=CAP, num_warps=4,
    )

    M = torch.empty((N, BT, BT), device=dev, dtype=f32)
    wa = torch.empty(sh, device=dev, dtype=f32)
    u = torch.empty((N, BT, V), device=dev, dtype=f32)
    BVw = min(_p2(V), 64)
    dplr_wy_kernel[(NT, B * H)](
        at, kt, bt, v, M, wa, u, T, H, K, V,
        BTC=BT, BK=BK, BV=BVw, LB=LOG_BT, PR=PREC, num_warps=4,
    )

    hst = torch.empty((N, BK, V), device=dev, dtype=f32)
    BVs = min(_p2(V), 16)
    dplr_scan_kernel[(B * H, triton.cdiv(V, BVs))](
        wa, u, kh, bh, gl, v, hst, T, H, K, V, NT,
        BTC=BT, BK=BK, BV=BVs, PR=PREC, num_warps=2,
    )

    o = torch.empty((B, T, H, V), device=dev, dtype=v.dtype)
    BVo = min(_p2(V), 64)
    dplr_out_kernel[(NT, B * H, triton.cdiv(V, BVo))](
        qt, kt, bt, v, u, hst, gl, o, T, H, K, V,
        BTC=BT, BK=BK, BV=BVo, PR=PREC, num_warps=4,
    )
    return o, (qt, kt, bt, at, kh, bh, gi, ge, gl, M, u, hst)


def _bwd(do, saved, q, k, v, alpha, beta):
    qt, kt, bt, at, kh, bh, gi, ge, gl, M, u, hst = saved
    B, T, H, K = q.shape
    V = v.shape[-1]
    dev = q.device
    NT = triton.cdiv(T, BT)
    BK = _p2(K)
    N = B * H * NT
    scale = K ** -0.5
    f32 = torch.float32

    do = do.contiguous()
    wb = torch.empty((N, BT, K), device=dev, dtype=f32)
    W = torch.empty((N, BT, V), device=dev, dtype=f32)
    BVb = min(_p2(V), 64)
    dplr_bprep_kernel[(NT, B * H)](
        M, bh, qt, bt, do, wb, W, T, H, K, V,
        BTC=BT, BK=BK, BV=BVb, PR=PREC, num_warps=4,
    )

    z = torch.empty((N, BK, V), device=dev, dtype=f32)
    BVs = min(_p2(V), 16)
    dplr_dscan_kernel[(B * H, triton.cdiv(V, BVs))](
        wb, W, qt, at, gl, do, z, T, H, K, V, NT,
        BTC=BT, BK=BK, BV=BVs, PR=PREC, num_warps=2,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    da = torch.empty_like(alpha)
    db = torch.empty_like(beta)
    BVg = min(_p2(V), 16)
    dg = torch.empty((B, T, H, K), device=dev, dtype=f32)
    dplr_dbwd_kernel[(NT, B * H)](
        qt, kt, bt, at, kh, bh, gi, ge, gl, v, u, W, do, hst, z,
        dq, dk, dv, da, db, T, H, K, V, scale,
        BTC=BT, BK=BK, BV=BVg, CAPV=CAP, PR=PREC, num_warps=4, num_stages=1,
    )
    BVd = _p2(V)
    dplr_dgate_kernel[(NT, B * H)](
        qt, kt, bt, at, gl, v, u, W, do, hst, z, dg, T, H, K, V,
        BTC=BT, BK=BK, BV=BVd, PR=PREC, num_warps=8, num_stages=1,
    )
    return dq, dk, dv, da, db, dg


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        o, saved = _fwd(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, *saved)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta = ctx.saved_tensors[:5]
        saved = ctx.saved_tensors[5:]
        dq, dk, dv, da, db, dg = _bwd(do, saved, q, k, v, alpha, beta)
        return dq, dk, dv, da, db, dg


def chunked_dplr_delta(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]    DPLR erasure basis
    beta: torch.Tensor,  # [B, T, H, K]    DPLR write basis
    gk: torch.Tensor,  # [B, T, H, K]    per-K-channel log-decay (<= 0)
) -> torch.Tensor:  # [B, T, H, V]
    return _Dplr.apply(q, k, v, alpha, beta, gk)
