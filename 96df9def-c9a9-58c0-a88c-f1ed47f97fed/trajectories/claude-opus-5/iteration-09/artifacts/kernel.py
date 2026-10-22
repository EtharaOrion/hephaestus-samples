"""Chunk-parallel DPLR (diagonal-plus-low-rank) generalized delta rule.

For one (batch, head) the recurrence implemented here is

    S_t = diag(exp(gk_t)) S_{t-1} + k_t v_t^T + beta_t (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))

with S in R^{K x V} and gk a per-K-channel log decay.

Chunkwise form
--------------
Time is cut into chunks of BT steps.  Inside a chunk let g be the inclusive
per-channel cumulative gate, gp = g - gk the exclusive one and gtot the chunk
total.  Define the gated projections

    qt = q * scale * exp(g)     at = alpha * exp(gp)
    bh = beta  * exp(-g)        kh = k * exp(-g)
    bb = beta  * exp(gtot - g)  kb = k * exp(gtot - g)

and the four masked intra-chunk matrices of the DPLR cascade

    A_ab = tril(at @ bh^T, -1)   A_ak = tril(at @ kh^T, -1)
    A_qb = tril(qt @ bh^T,  0)   A_qk = tril(qt @ kh^T,  0)

A_ab is strictly lower triangular, hence nilpotent, and it is the intra-chunk
alpha/beta erasure cascade.  Inverting (I - A_ab) exactly (repeated squaring,
no truncation) gives the chunk's rank-1 read vectors and then the outputs and
the next boundary state:

    U  = (I - A_ab)^{-1} (A_ak V + at S0)
    O  = qt S0 + A_qb U + A_qk V
    S1 = diag(exp(gtot)) S0 + bb^T U + kb^T V

Backward
--------
The adjoint recurrence Z_{t-1} = M_t^T Z_t + q_{t-1} do_{t-1}^T has the same
shape with time reversed, so it factors through a strictly *upper* triangular
nilpotent B_ba = triu(bb @ ab^T, 1) with ab = alpha * exp(gp - gtot) and
qb = q * scale * exp(g - gtot):

    W  = (I - B_ba)^{-1} (triu(bb @ qb^T, 0) DO + bb Cin)
    C0 = diag(exp(gtot)) Cin + at^T W + qt^T DO

from which the five matrix gradients are masked-matmul contractions of the
same tiles.  The gate gradient closes analytically:

    dgk_t = revcumsum_{s>=t}(q dq - k dk - beta dbeta + alpha dalpha)_s
            - (alpha dalpha)_t

Precision
---------
The boundary state carried between chunks -- S in the forward scan, C in the
backward scan, and the checkpoint buffers that hand them from forward to
backward -- lives in float32 for its whole life, as do U and W.  Only the
inputs and the returned gradients use the caller's dtype.

Every exponent is clamped symmetrically to +-EXP_CAP.  The clamp is a
branchless min/max on a value, not a control-flow decision, and it only
engages where the paired factor of the product has already underflowed, so it
bounds the result without changing it.
"""

import torch
import triton
import triton.language as tl

BT = 16           # time steps per chunk
LOG_BT = 4        # log2(BT): squarings needed to invert the nilpotent exactly
EXP_CAP = tl.constexpr(80.0)
PREC = tl.constexpr("ieee")


@triton.jit
def _ex(x):
    return tl.exp(tl.minimum(tl.maximum(x, -EXP_CAP), EXP_CAP))


@triton.jit
def _mm(a, b, DT: tl.constexpr):
    """Bulk matmul: narrow operands, fp32 accumulation.

    Used for the wide contractions along the V axis, where the operands are
    single tiles of data rather than products of the intra-chunk cascade and a
    narrow mantissa costs a few units in the last place.
    """
    return tl.dot(a.to(DT), b.to(DT), input_precision=PREC)


@triton.jit
def _mmh(a, b):
    """Matmul kept at fp32 operand width.

    The [BT, BT] intra-chunk matrices and the inverse of (I - A_ab) sit inside
    the cascade: their error is amplified by every later stage, and the tiles
    they span can cover the whole chunk's decay range.  Reduced-mantissa
    variants were measured here and are not accurate enough: tf32's 10 bits and
    a split-bf16 product's ~16 both leave the inverted cascade outside
    tolerance, so these sites stay at full fp32 width.  They are small next to
    the V-axis work, which runs narrow.
    """
    return tl.dot(a, b, input_precision="ieee")


@triton.jit
def _mmx(a, b):
    """Matmul at the widest precision, for the inverse's repeated squaring.

    Errors introduced here are raised to a power by the remaining squarings,
    so this one site does not tolerate a reduced mantissa.
    """
    return tl.dot(a, b, input_precision="ieee")


@triton.jit
def _nilpotent_inverse(N, BT_: tl.constexpr, LOG: tl.constexpr):
    """(I - N)^{-1} for triangular nilpotent N, by repeated squaring.

    Uses prod_{j<LOG} (I + N^(2^j)); since N^BT == 0 the product terminates
    exactly, so this is the true inverse and not a truncated Neumann series.
    """
    i = tl.arange(0, BT_)
    M = tl.where(i[:, None] == i[None, :], 1.0, 0.0)
    P = N
    for _ in tl.static_range(LOG):
        M = M + _mmx(P, M)
        P = _mmx(P, P)
    return M


@triton.jit
def dplr_cumsum(g_ptr, gc_ptr, gt_ptr, T, H: tl.constexpr, K: tl.constexpr,
                NT, BT_: tl.constexpr, BK: tl.constexpr):
    """Inclusive per-channel cumulative gate inside each chunk, plus its total.

    Reads the caller's [B, T, H, K] layout, writes float32 [B*H, T, K] and the
    per-chunk totals [B*H, NT, K].
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT_ + tl.arange(0, BT_)
    o_k = tl.arange(0, BK)
    mk = o_k < K
    m = (o_t < T)[:, None] & mk[None, :]

    p = (i_b * T) * H * K + i_h * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    g = tl.load(g_ptr + p, mask=m, other=0.0).to(tl.float32)
    cs = tl.cumsum(g, axis=0)
    base = i_bh * T * K
    tl.store(gc_ptr + base + o_t[:, None].to(tl.int64) * K + o_k[None, :], cs, mask=m)
    # chunk total: masked-out rows contributed zero, so the plain sum is the
    # cumulative value at the last valid step of the chunk.
    tl.store(gt_ptr + (i_bh * NT + i_t) * K + o_k, tl.sum(g, axis=0), mask=mk)


@triton.jit
def dplr_wy_fwd(a_ptr, b_ptr, k_ptr, v_ptr, g_ptr, gk_ptr, gt_ptr,
                w_ptr, u_ptr, T, H: tl.constexpr, K: tl.constexpr,
                V: tl.constexpr, NT, BT_: tl.constexpr, BK: tl.constexpr,
                BV: tl.constexpr, LOG: tl.constexpr, DT: tl.constexpr):
    """Invert the intra-chunk erasure cascade: w = Minv at, u = Minv A_ak V."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT_ + tl.arange(0, BT_)
    o_k = tl.arange(0, BK)
    mt = o_t < T
    m = mt[:, None] & (o_k < K)[None, :]

    pk = (i_b * T) * H * K + i_h * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    pg = i_bh * T * K + o_t[:, None].to(tl.int64) * K + o_k[None, :]
    g = tl.load(g_ptr + pg, mask=m, other=0.0)
    gp = g - tl.load(gk_ptr + pk, mask=m, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + pk, mask=m, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + pk, mask=m, other=0.0).to(tl.float32)
    kk = tl.load(k_ptr + pk, mask=m, other=0.0).to(tl.float32)

    # The pairwise exponent of A_ab[i, j] is gp_i - g_j, which is bounded, but
    # splitting it as exp(gp_i) * exp(-g_j) puts the whole chunk's decay into
    # one factor and overflows a narrow dtype.  Splitting about the chunk
    # midpoint gm instead keeps both factors within exp(|gtot| / 2) while the
    # product is unchanged.
    gm = 0.5 * tl.load(gt_ptr + (i_bh * NT + i_t) * K + o_k, mask=o_k < K, other=0.0)
    at = a * _ex(gp)
    asx = a * _ex(gp - gm[None, :])
    bh = b * _ex(gm[None, :] - g)
    kh = kk * _ex(gm[None, :] - g)

    i = tl.arange(0, BT_)
    strict = i[:, None] > i[None, :]
    A_ab = tl.where(strict, _mmh(asx, tl.trans(bh)), 0.0)
    A_ak = tl.where(strict, _mmh(asx, tl.trans(kh)), 0.0)

    Minv = _nilpotent_inverse(A_ab, BT_, LOG)
    tl.store(w_ptr + pg, _mmh(Minv, at), mask=m)

    MA = _mmh(Minv, A_ak)
    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        mv = mt[:, None] & (o_v < V)[None, :]
        pv = (i_b * T) * H * V + i_h * V + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        vv = tl.load(v_ptr + pv, mask=mv, other=0.0).to(tl.float32)
        u = _mmh(MA, vv)
        tl.store(u_ptr + i_bh * T * V + o_t[:, None].to(tl.int64) * V + o_v[None, :], u, mask=mv)


@triton.jit
def dplr_state_fwd(k_ptr, b_ptr, v_ptr, g_ptr, gt_ptr, w_ptr, u_ptr, s_ptr,
                   T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
                   BT_: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, DT: tl.constexpr):
    """Sequential scan over chunks carrying the boundary state in float32.

    Per chunk it finalizes U = w S0 + u0, checkpoints S0, then advances
    S1 = diag(exp(gtot)) S0 + bb^T U + kb^T V.
    """
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_i = tl.arange(0, BT_)
    mk = o_k < K
    mv = o_v < V
    S = tl.zeros([BK, BV], dtype=tl.float32)

    for i_t in range(0, NT):
        o_t = i_t * BT_ + o_i
        mt = o_t < T
        mkk = mt[:, None] & mk[None, :]
        mvv = mt[:, None] & mv[None, :]
        pk = (i_b * T) * H * K + i_h * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        pg = i_bh * T * K + o_t[:, None].to(tl.int64) * K + o_k[None, :]
        pu = i_bh * T * V + o_t[:, None].to(tl.int64) * V + o_v[None, :]

        tl.store(s_ptr + ((i_bh * NT + i_t) * K + o_k[:, None].to(tl.int64)) * V + o_v[None, :],
                 S, mask=mk[:, None] & mv[None, :])

        w = tl.load(w_ptr + pg, mask=mkk, other=0.0)
        u = tl.load(pu + u_ptr, mask=mvv, other=0.0)
        u += _mmh(w, S)
        tl.store(u_ptr + pu, u, mask=mvv)

        g = tl.load(g_ptr + pg, mask=mkk, other=0.0)
        gtot = tl.load(gt_ptr + (i_bh * NT + i_t) * K + o_k, mask=mk, other=0.0)
        dec = _ex(gtot - g)
        bb = tl.load(b_ptr + pk, mask=mkk, other=0.0).to(tl.float32) * dec
        kb = tl.load(k_ptr + pk, mask=mkk, other=0.0).to(tl.float32) * dec
        pv = (i_b * T) * H * V + i_h * V + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        vv = tl.load(v_ptr + pv, mask=mvv, other=0.0).to(tl.float32)

        S = S * _ex(gtot)[:, None] + _mmh(tl.trans(bb), u) \
            + _mmh(tl.trans(kb), vv)


@triton.jit
def dplr_out_fwd(q_ptr, k_ptr, b_ptr, v_ptr, g_ptr, gk_ptr, gt_ptr, u_ptr, s_ptr, o_ptr,
                 scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
                 BT_: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, DT: tl.constexpr):
    """O = qt S0 + A_qb U + A_qk V for one chunk."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT_ + tl.arange(0, BT_)
    o_k = tl.arange(0, BK)
    mt = o_t < T
    mkk = mt[:, None] & (o_k < K)[None, :]

    pk = (i_b * T) * H * K + i_h * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    pg = i_bh * T * K + o_t[:, None].to(tl.int64) * K + o_k[None, :]
    g = tl.load(g_ptr + pg, mask=mkk, other=0.0)
    gm = 0.5 * tl.load(gt_ptr + (i_bh * NT + i_t) * K + o_k, mask=o_k < K, other=0.0)
    hb = _ex(gm[None, :] - g)
    qr = tl.load(q_ptr + pk, mask=mkk, other=0.0).to(tl.float32) * scale
    qt = qr * _ex(g)                       # pairs with the chunk-entry state
    qs = qr * _ex(g - gm[None, :])         # pairs with the midpoint-split tiles
    bh = tl.load(b_ptr + pk, mask=mkk, other=0.0).to(tl.float32) * hb
    kh = tl.load(k_ptr + pk, mask=mkk, other=0.0).to(tl.float32) * hb

    i = tl.arange(0, BT_)
    li = i[:, None] >= i[None, :]
    A_qb = tl.where(li, _mmh(qs, tl.trans(bh)), 0.0)
    A_qk = tl.where(li, _mmh(qs, tl.trans(kh)), 0.0)

    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        mv = o_v < V
        mvv = mt[:, None] & mv[None, :]
        pv = (i_b * T) * H * V + i_h * V + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        vv = tl.load(v_ptr + pv, mask=mvv, other=0.0).to(tl.float32)
        uu = tl.load(u_ptr + i_bh * T * V + o_t[:, None].to(tl.int64) * V + o_v[None, :],
                     mask=mvv, other=0.0)
        S0 = tl.load(s_ptr + ((i_bh * NT + i_t) * K + o_k[:, None].to(tl.int64)) * V + o_v[None, :],
                     mask=(o_k < K)[:, None] & mv[None, :], other=0.0)
        # A_qb @ u closes the intra-chunk cascade, so it keeps full width; the
        # state and raw-value terms do not.
        o = _mmh(qt, S0) + _mmh(A_qb, uu) \
            + _mm(A_qk, vv, DT)
        tl.store(o_ptr + pv, o.to(o_ptr.dtype.element_ty), mask=mvv)


@triton.jit
def dplr_wy_bwd(a_ptr, b_ptr, q_ptr, do_ptr, g_ptr, gk_ptr, gt_ptr,
                wq_ptr, wd_ptr, scale, T, H: tl.constexpr, K: tl.constexpr,
                V: tl.constexpr, NT, BT_: tl.constexpr, BK: tl.constexpr,
                BV: tl.constexpr, LOG: tl.constexpr, DT: tl.constexpr):
    """Reverse-time counterpart of dplr_wy_fwd: invert triu(bb ab^T, 1)."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT_ + tl.arange(0, BT_)
    o_k = tl.arange(0, BK)
    mt = o_t < T
    mkk = mt[:, None] & (o_k < K)[None, :]

    pk = (i_b * T) * H * K + i_h * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    pg = i_bh * T * K + o_t[:, None].to(tl.int64) * K + o_k[None, :]
    g = tl.load(g_ptr + pg, mask=mkk, other=0.0)
    gp = g - tl.load(gk_ptr + pk, mask=mkk, other=0.0).to(tl.float32)
    gtot = tl.load(gt_ptr + (i_bh * NT + i_t) * K + o_k, mask=o_k < K, other=0.0)

    bb = tl.load(b_ptr + pk, mask=mkk, other=0.0).to(tl.float32) * _ex(gtot[None, :] - g)
    ab = tl.load(a_ptr + pk, mask=mkk, other=0.0).to(tl.float32) * _ex(gp - gtot[None, :])
    qb = tl.load(q_ptr + pk, mask=mkk, other=0.0).to(tl.float32) * scale * _ex(g - gtot[None, :])

    i = tl.arange(0, BT_)
    B_ba = tl.where(i[:, None] < i[None, :], _mmh(bb, tl.trans(ab)), 0.0)
    B_bq = tl.where(i[:, None] <= i[None, :], _mmh(bb, tl.trans(qb)), 0.0)

    Tb = _nilpotent_inverse(B_ba, BT_, LOG)
    tl.store(wq_ptr + pg, _mmh(Tb, bb), mask=mkk)

    TQ = _mmh(Tb, B_bq)
    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        mvv = mt[:, None] & (o_v < V)[None, :]
        pdo = (i_b * T) * H * V + i_h * V + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        do = tl.load(do_ptr + pdo, mask=mvv, other=0.0).to(tl.float32)
        wd = _mmh(TQ, do)
        tl.store(wd_ptr + i_bh * T * V + o_t[:, None].to(tl.int64) * V + o_v[None, :],
                 wd, mask=mvv)


@triton.jit
def dplr_state_bwd(q_ptr, a_ptr, do_ptr, g_ptr, gk_ptr, gt_ptr, wq_ptr, wd_ptr,
                   c_ptr, cb_ptr, scale, T, H: tl.constexpr, K: tl.constexpr,
                   V: tl.constexpr, NT, BT_: tl.constexpr, BK: tl.constexpr,
                   BV: tl.constexpr, DT: tl.constexpr):
    """Reverse sequential scan of the adjoint state, carried in float32."""
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_i = tl.arange(0, BT_)
    mk = o_k < K
    mv = o_v < V
    C = tl.zeros([BK, BV], dtype=tl.float32)

    for j in range(0, NT):
        i_t = NT - 1 - j
        o_t = i_t * BT_ + o_i
        mt = o_t < T
        mkk = mt[:, None] & mk[None, :]
        mvv = mt[:, None] & mv[None, :]
        pk = (i_b * T) * H * K + i_h * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
        pg = i_bh * T * K + o_t[:, None].to(tl.int64) * K + o_k[None, :]
        pw = i_bh * T * V + o_t[:, None].to(tl.int64) * V + o_v[None, :]

        tl.store(c_ptr + ((i_bh * NT + i_t) * K + o_k[:, None].to(tl.int64)) * V + o_v[None, :],
                 C, mask=mk[:, None] & mv[None, :])

        wq = tl.load(wq_ptr + pg, mask=mkk, other=0.0)
        W = tl.load(wd_ptr + pw, mask=mvv, other=0.0) + _mmh(wq, C)
        tl.store(wd_ptr + pw, W, mask=mvv)

        g = tl.load(g_ptr + pg, mask=mkk, other=0.0)
        gp = g - tl.load(gk_ptr + pk, mask=mkk, other=0.0).to(tl.float32)
        gtot = tl.load(gt_ptr + (i_bh * NT + i_t) * K + o_k, mask=mk, other=0.0)
        at = tl.load(a_ptr + pk, mask=mkk, other=0.0).to(tl.float32) * _ex(gp)
        qt = tl.load(q_ptr + pk, mask=mkk, other=0.0).to(tl.float32) * scale * _ex(g)
        pdo = (i_b * T) * H * V + i_h * V + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        do = tl.load(do_ptr + pdo, mask=mvv, other=0.0).to(tl.float32)

        C = C * _ex(gtot)[:, None] + _mmh(tl.trans(at), W) \
            + _mmh(tl.trans(qt), do)
        # C is now the adjoint leaving this chunk backwards; it anchors dgk at
        # the chunk's first step.
        tl.store(cb_ptr + ((i_bh * NT + i_t) * K + o_k[:, None].to(tl.int64)) * V + o_v[None, :],
                 C, mask=mk[:, None] & mv[None, :])


@triton.jit
def dplr_grads(q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, do_ptr, g_ptr, gk_ptr, gt_ptr,
               u_ptr, w_ptr, s_ptr, c_ptr,
               dq_ptr, dk_ptr, dv_ptr, da_ptr, db_ptr, ip_ptr, ad_ptr,
               scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
               BT_: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, DT: tl.constexpr):
    """The five matrix gradients for one chunk, plus the per-channel inner
    products that the gate gradient reverse-scans."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_t * BT_ + tl.arange(0, BT_)
    o_k = tl.arange(0, BK)
    mt = o_t < T
    mk = o_k < K
    mkk = mt[:, None] & mk[None, :]

    pk = (i_b * T) * H * K + i_h * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    pg = i_bh * T * K + o_t[:, None].to(tl.int64) * K + o_k[None, :]

    g = tl.load(g_ptr + pg, mask=mkk, other=0.0)
    gp = g - tl.load(gk_ptr + pk, mask=mkk, other=0.0).to(tl.float32)
    gtot = tl.load(gt_ptr + (i_bh * NT + i_t) * K + o_k, mask=mk, other=0.0)

    qr = tl.load(q_ptr + pk, mask=mkk, other=0.0).to(tl.float32)
    kr = tl.load(k_ptr + pk, mask=mkk, other=0.0).to(tl.float32)
    ar = tl.load(a_ptr + pk, mask=mkk, other=0.0).to(tl.float32)
    br = tl.load(b_ptr + pk, mask=mkk, other=0.0).to(tl.float32)

    eg = _ex(g)
    negg = _ex(-g)
    dec = _ex(gtot[None, :] - g)
    bh = br * negg
    kh = kr * negg
    ab = ar * _ex(gp - gtot[None, :])
    qb = qr * scale * _ex(g - gtot[None, :])
    qt = qr * scale * eg
    at = ar * _ex(gp)
    kb = kr * dec

    i = tl.arange(0, BT_)
    li = i[:, None] >= i[None, :]
    sl = i[:, None] > i[None, :]
    ui = i[:, None] <= i[None, :]
    su = i[:, None] < i[None, :]

    dq = tl.zeros([BT_, BK], dtype=tl.float32)
    dk = tl.zeros([BT_, BK], dtype=tl.float32)
    da = tl.zeros([BT_, BK], dtype=tl.float32)
    db = tl.zeros([BT_, BK], dtype=tl.float32)

    # The V axis is tiled so the live fp32 working set fits in shared memory;
    # the four K-side gradients accumulate across tiles, dv is complete per tile.
    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        mv = o_v < V
        mvv = mt[:, None] & mv[None, :]
        pv = (i_b * T) * H * V + i_h * V + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        pw = i_bh * T * V + o_t[:, None].to(tl.int64) * V + o_v[None, :]
        pst = ((i_bh * NT + i_t) * K + o_k[:, None].to(tl.int64)) * V + o_v[None, :]
        mst = mk[:, None] & mv[None, :]

        vv = tl.load(v_ptr + pv, mask=mvv, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + pv, mask=mvv, other=0.0).to(tl.float32)
        uu = tl.load(u_ptr + pw, mask=mvv, other=0.0)
        ww = tl.load(w_ptr + pw, mask=mvv, other=0.0)
        S0 = tl.load(s_ptr + pst, mask=mst, other=0.0)
        Ci = tl.load(c_ptr + pst, mask=mst, other=0.0)

        # Four of the eight time-by-time Gram matrices needed here are exact
        # transposes of the other four -- an upper-masked G^T is the transpose
        # of the lower-masked G -- so each is formed once and reused.
        G_du = _mmh(do, tl.trans(uu))
        G_dv = _mmh(do, tl.trans(vv))
        G_wu = _mmh(ww, tl.trans(uu))
        G_wv = _mmh(ww, tl.trans(vv))

        dq += (_mmh(do, tl.trans(S0))
               + _mmh(tl.where(li, G_du, 0.0), bh)
               + _mmh(tl.where(li, G_dv, 0.0), kh))

        dk += (_mmh(vv, tl.trans(Ci))
               + _mmh(tl.where(su, tl.trans(tl.where(sl, G_wv, 0.0)), 0.0), ab)
               + _mmh(tl.where(ui, tl.trans(tl.where(li, G_dv, 0.0)), 0.0), qb))

        db += (_mmh(uu, tl.trans(Ci))
               + _mmh(tl.where(su, tl.trans(tl.where(sl, G_wu, 0.0)), 0.0), ab)
               + _mmh(tl.where(ui, tl.trans(tl.where(li, G_du, 0.0)), 0.0), qb))

        da += (_mmh(ww, tl.trans(S0))
               + _mmh(tl.where(sl, G_wu, 0.0), bh)
               + _mmh(tl.where(sl, G_wv, 0.0), kh))

        dv = (_mm(kb, Ci, DT)
              + _mmh(tl.where(su, _mmh(kb, tl.trans(ab)), 0.0), ww)
              + _mm(tl.where(ui, _mm(kb, tl.trans(qb), DT), 0.0), do, DT))
        tl.store(dv_ptr + pv, dv.to(dv_ptr.dtype.element_ty), mask=mvv)

    dq = dq * scale * eg
    dk = dk * dec
    db = db * dec
    da = da * _ex(gp)

    tl.store(dq_ptr + pk, dq.to(dq_ptr.dtype.element_ty), mask=mkk)
    tl.store(dk_ptr + pk, dk.to(dk_ptr.dtype.element_ty), mask=mkk)
    tl.store(da_ptr + pk, da.to(da_ptr.dtype.element_ty), mask=mkk)
    tl.store(db_ptr + pk, db.to(db_ptr.dtype.element_ty), mask=mkk)

    # per-channel inner products feeding the gate gradient; the alpha term is
    # kept separately because dgk subtracts it back at its own timestep.
    ad = ar * da
    tl.store(ip_ptr + pg, qr * dq - kr * dk - br * db + ad, mask=mkk)
    tl.store(ad_ptr + pg, ad, mask=mkk)


@triton.jit
def dplr_dgk(ip_ptr, ad_ptr, s_ptr, cb_ptr, dg_ptr, T, H: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, NT, BT_: tl.constexpr,
             BK: tl.constexpr, BV: tl.constexpr):
    """Gate gradient, from the state/adjoint contraction directly.

    dgk_t[c] = exp(gk_t[c]) * ( S_{t-1}[c, :] . Z_t[c, :] ), where Z is the
    adjoint of the state.  Writing S_{t-1} = diag(exp(gp_t)) P_{t-1} and
    Z_t = diag(exp(gtot - g_t)) R_t in the chunk-local variables, the gate
    factors collapse to a single bounded exp(gtot):

        dgk_t[c] = exp(gtot[c]) * ( P_{t-1}[c, :] . R_t[c, :] )

    P_{t-1} is the chunk-entry state S0 plus the rank-1 writes before t, and
    R_t is the chunk-exit adjoint Cin plus the reads from t onwards, so the
    contraction expands into a constant term, a prefix term, a suffix term and
    a prefix-by-suffix cross term.  Each is a matmul or a cumulative sum over
    the chunk.

    Expressing dgk instead as a running sum of (q dq - k dk - beta dbeta +
    alpha dalpha) is algebraically equal but numerically useless whenever the
    recurrence amplifies: those four products are individually many orders of
    magnitude larger than their sum, so the answer is lost to cancellation
    before it is ever formed.  The contraction above never forms them.
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_k = tl.arange(0, BK)
    o_i = tl.arange(0, BT_)
    o_t = i_t * BT_ + o_i
    mk = o_k < K
    mt = o_t < T
    m = mt[:, None] & mk[None, :]

    pk = (i_b * T) * H * K + i_h * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    pg = i_bh * T * K + o_t[:, None].to(tl.int64) * K + o_k[None, :]
    g = tl.load(g_ptr + pg, mask=m, other=0.0)
    gp = g - tl.load(gk_ptr + pk, mask=m, other=0.0).to(tl.float32)
    gtot = tl.load(gt_ptr + (i_bh * NT + i_t) * K + o_k, mask=mk, other=0.0)

    negg = _ex(-g)
    bh = tl.load(b_ptr + pk, mask=m, other=0.0).to(tl.float32) * negg
    kh = tl.load(k_ptr + pk, mask=m, other=0.0).to(tl.float32) * negg
    ab = tl.load(a_ptr + pk, mask=m, other=0.0).to(tl.float32) * _ex(gp - gtot[None, :])
    qb = tl.load(q_ptr + pk, mask=m, other=0.0).to(tl.float32) * scale * _ex(g - gtot[None, :])
    bh = tl.where(mt[:, None], bh, 0.0)
    kh = tl.where(mt[:, None], kh, 0.0)
    ab = tl.where(mt[:, None], ab, 0.0)
    qb = tl.where(mt[:, None], qb, 0.0)

    sc = tl.zeros([BK], dtype=tl.float32)
    yW = tl.zeros([BT_, BK], dtype=tl.float32)   # S0 . W_s,  summed over s > t
    yD = tl.zeros([BT_, BK], dtype=tl.float32)   # S0 . do_s, summed over s >= t
    yU = tl.zeros([BT_, BK], dtype=tl.float32)   # Cin . u_j, summed over j < t
    yV = tl.zeros([BT_, BK], dtype=tl.float32)   # Cin . v_j, summed over j < t
    G_uW = tl.zeros([BT_, BT_], dtype=tl.float32)
    G_uD = tl.zeros([BT_, BT_], dtype=tl.float32)
    G_vW = tl.zeros([BT_, BT_], dtype=tl.float32)
    G_vD = tl.zeros([BT_, BT_], dtype=tl.float32)

    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        mv = o_v < V
        mvv = mt[:, None] & mv[None, :]
        pv = (i_b * T) * H * V + i_h * V + o_t[:, None].to(tl.int64) * (H * V) + o_v[None, :]
        pw = i_bh * T * V + o_t[:, None].to(tl.int64) * V + o_v[None, :]
        pst = ((i_bh * NT + i_t) * K + o_k[:, None].to(tl.int64)) * V + o_v[None, :]
        mst = mk[:, None] & mv[None, :]

        vv = tl.load(v_ptr + pv, mask=mvv, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + pv, mask=mvv, other=0.0).to(tl.float32)
        uu = tl.load(u_ptr + pw, mask=mvv, other=0.0)
        ww = tl.load(w_ptr + pw, mask=mvv, other=0.0)
        S0 = tl.load(s_ptr + pst, mask=mst, other=0.0)
        Ci = tl.load(c_ptr + pst, mask=mst, other=0.0)

        sc += tl.sum(S0 * Ci, axis=1)
        # the chunk-entry state paired with the adjoint's reads, and the
        # state's writes paired with the chunk-exit adjoint
        yW += tl.trans(_mmh(S0, tl.trans(ww)))
        yD += tl.trans(_mmh(S0, tl.trans(do)))
        yU += tl.trans(_mmh(Ci, tl.trans(uu)))
        yV += tl.trans(_mmh(Ci, tl.trans(vv)))

        G_uW += _mmh(uu, tl.trans(ww))
        G_uD += _mmh(uu, tl.trans(do))
        G_vW += _mmh(vv, tl.trans(ww))
        G_vD += _mmh(vv, tl.trans(do))

    ii = tl.arange(0, BT_)
    # Suffix over s > t for the alpha/W reads, s >= t for the do reads, and
    # prefix over j < t for the writes.
    y1 = yW * ab
    y2 = yD * qb
    y3 = yU * bh
    y4 = yV * kh
    dg = (sc[None, :]
          + (tl.sum(y1, axis=0)[None, :] - tl.cumsum(y1, axis=0))
          + (tl.sum(y2, axis=0)[None, :] - tl.cumsum(y2, axis=0) + y2)
          + (tl.cumsum(y3, axis=0) - y3)
          + (tl.cumsum(y4, axis=0) - y4))

    # Cross term: sum over j < t of sum over s > t (or s >= t) of
    # write_j . read_s.  Stepping t by one adds the j = t row of the Gram
    # matrix and drops one column, so the whole profile follows from two
    # statically masked matmuls per pair plus a running sum.  That running sum
    # starts and ends at exactly zero, so each row accumulates from whichever
    # end of the chunk is nearer -- a choice made on the row index only.
    half = (ii < (BT_ // 2))[:, None]
    dg += _cross(G_uW, bh, ab, ii, BT_, half, 1)
    dg += _cross(G_uD, bh, qb, ii, BT_, half, 0)
    dg += _cross(G_vW, kh, ab, ii, BT_, half, 1)
    dg += _cross(G_vD, kh, qb, ii, BT_, half, 0)

    dg = dg * _ex(gtot)[None, :]
    pd = (i_b * T) * H * K + i_h * K + o_t[:, None].to(tl.int64) * (H * K) + o_k[None, :]
    tl.store(dg_ptr + pd, dg.to(dg_ptr.dtype.element_ty), mask=m)


def _dt_of(x):
    return {torch.bfloat16: tl.bfloat16,
            torch.float16: tl.float16}.get(x.dtype, tl.float32)


def _nx(x):
    return max(16, triton.next_power_of_2(x))


def _fwd(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH, NT = B * H, triton.cdiv(T, BT)
    BK, BV = _nx(K), _nx(V)
    BVt = min(BV, 64)
    DT = _dt_of(q)
    dev = q.device
    f32 = torch.float32

    g = torch.empty(BH, T, K, device=dev, dtype=f32)
    gt = torch.empty(BH, NT, K, device=dev, dtype=f32)
    dplr_cumsum[(NT, BH)](gk, g, gt, T, H, K, NT, BT_=BT, BK=BK,
                          num_warps=4, num_stages=2)

    w = torch.empty(BH, T, K, device=dev, dtype=f32)
    u = torch.empty(BH, T, V, device=dev, dtype=f32)
    dplr_wy_fwd[(NT, BH)](alpha, beta, k, v, g, gk, gt, w, u, T, H, K, V, NT,
                          BT_=BT, BK=BK, BV=BVt, LOG=LOG_BT,
                          DT=DT, num_warps=4, num_stages=2)

    S = torch.empty(BH, NT, K, V, device=dev, dtype=f32)
    BVs = min(BV, 16)
    dplr_state_fwd[(triton.cdiv(V, BVs), BH)](
        k, beta, v, g, gt, w, u, S, T, H, K, V, NT,
        BT_=BT, BK=BK, BV=BVs, DT=DT, num_warps=8, num_stages=2)

    o = torch.empty(B, T, H, V, device=dev, dtype=v.dtype)
    dplr_out_fwd[(NT, BH)](q, k, beta, v, g, gk, gt, u, S, o, K ** -0.5,
                           T, H, K, V, NT, BT_=BT, BK=BK, BV=BVt,
                           DT=DT, num_warps=4, num_stages=2)
    return o, g, gt, u, S


def _bwd(do, q, k, v, alpha, beta, gk, g, gt, u, S):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH, NT = B * H, triton.cdiv(T, BT)
    BK, BV = _nx(K), _nx(V)
    BVt = min(BV, 64)
    dev = q.device
    f32 = torch.float32
    scale = K ** -0.5
    do = do.contiguous()
    DT = _dt_of(q)

    wq = torch.empty(BH, T, K, device=dev, dtype=f32)
    W = torch.empty(BH, T, V, device=dev, dtype=f32)
    dplr_wy_bwd[(NT, BH)](alpha, beta, q, do, g, gk, gt, wq, W, scale,
                          T, H, K, V, NT, BT_=BT, BK=BK, BV=BVt, LOG=LOG_BT,
                          DT=DT, num_warps=4, num_stages=2)

    C = torch.empty(BH, NT, K, V, device=dev, dtype=f32)
    Cb = torch.empty(BH, NT, K, V, device=dev, dtype=f32)
    BVs = min(BV, 16)
    dplr_state_bwd[(triton.cdiv(V, BVs), BH)](
        q, alpha, do, g, gk, gt, wq, W, C, Cb, scale, T, H, K, V, NT,
        BT_=BT, BK=BK, BV=BVs, DT=DT, num_warps=8, num_stages=2)

    dq = torch.empty(B, T, H, K, device=dev, dtype=q.dtype)
    dk = torch.empty(B, T, H, K, device=dev, dtype=k.dtype)
    dv = torch.empty(B, T, H, V, device=dev, dtype=v.dtype)
    da = torch.empty(B, T, H, K, device=dev, dtype=alpha.dtype)
    db = torch.empty(B, T, H, K, device=dev, dtype=beta.dtype)
    ip = torch.empty(BH, T, K, device=dev, dtype=f32)
    ad = torch.empty(BH, T, K, device=dev, dtype=f32)
    dplr_grads[(NT, BH)](q, k, v, alpha, beta, do, g, gk, gt, u, W, S, C,
                         dq, dk, dv, da, db, ip, ad, scale, T, H, K, V, NT,
                         BT_=BT, BK=BK, BV=min(BV, 64), DT=DT, num_warps=4, num_stages=2)

    dgk = torch.empty(B, T, H, K, device=dev, dtype=gk.dtype)
    dplr_dgk[(NT, BH)](ip, ad, S, Cb, dgk, T, H, K, V, NT,
                       BT_=BT, BK=BK, BV=min(BV, 64),
                       num_warps=4, num_stages=2)
    return dq, dk, dv, da, db, dgk


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta = alpha.contiguous(), beta.contiguous()
        gk = gk.contiguous()
        o, g, gt, u, S = _fwd(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, g, gt, u, S)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta, gk, g, gt, u, S = ctx.saved_tensors
        return _bwd(do, q, k, v, alpha, beta, gk, g, gt, u, S)


def chunked_dplr_delta(
    q: torch.Tensor,      # [B, T, H, K]
    k: torch.Tensor,      # [B, T, H, K]
    v: torch.Tensor,      # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]    DPLR erasure basis
    beta: torch.Tensor,   # [B, T, H, K]    DPLR write basis
    gk: torch.Tensor,     # [B, T, H, K]    per-K-channel log-decay (<= 0)
) -> torch.Tensor:        # [B, T, H, V]
    return _Dplr.apply(q, k, v, alpha, beta, gk)
