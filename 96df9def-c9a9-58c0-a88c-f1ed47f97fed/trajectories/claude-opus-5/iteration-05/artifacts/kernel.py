"""
Chunk-parallel DPLR (diagonal-plus-low-rank) generalized delta rule -- forward
and backward, implemented as a set of Triton kernels.

Recurrence (per batch/head, state S in R^{K x V}):

    S_t = diag(exp(gk_t)) S_{t-1} + k_t v_t^T + beta_t (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))

Chunked factorization used here (chunk length BT, per-K-channel gate):

    G   = inclusive cumsum of gk inside the chunk,  gl = G[-1],  Gp = G - gk

    Because gk <= 0 the cumsum G is non-increasing, so every exponent below is
    non-positive and every factor lands in (0, 1]:

    qt  = q*scale*exp(G)    at = alpha*exp(Gp)
    kh  = k*exp(gl - G)     bh = beta*exp(gl - G)

    The four intra-chunk matrices need the pairwise ratio exp(G[t] - G[s]),
    which is what a k*exp(-G) column factor would supply -- but exp(-G) alone
    overflows float32 on a long chunk of shallow decay, and forming the ratio
    as a product of a huge and a tiny number destroys the mantissa long before
    that.  They are built instead one source row s at a time, from the
    difference of the two cumsums directly, so the exponent is never positive:

    A_qk[t,s] = <q[t], k[s]> exp(G[t] - G[s])          s <= t
    A_qb[t,s] = <q[t], beta[s]> exp(G[t] - G[s])       s <= t
    A_ak[t,s] = <alpha[t], k[s]> exp(Gp[t] - G[s])     s <  t
    A_ab[t,s] = <alpha[t], beta[s]> exp(Gp[t] - G[s])  s <  t
    A_ab = tril(at bt^T, -1)  A_ak = tril(at kt^T, -1)
    A_qb = tril(qt bt^T,  0)  A_qk = tril(qt kt^T,  0)
    M    = (I - A_ab)^{-1}          (exact: A_ab is nilpotent, doubling series)
    W    = M at             Uk = M (A_ak v)
    U    = Uk + W S
    O    = qt S + A_qb U + A_qk v
    S'   = S*exp(gl) + bh^T U + kh^T v

The chunk-boundary state (S in the forward scan, Z = dS in the backward scan)
is created, carried, checkpointed and consumed in float32 everywhere.
"""

import torch
import triton
import triton.language as tl

BT = 16           # chunk length (semantic)
BTD = 16          # tile width; Triton needs >= 16 on a dot contraction
LOG_BT = 4        # log2(BTD)
BV_SCAN = 64      # V tile of the sequential state scans
BV_BWD = 64       # V tile of the backward per-chunk kernel
EXP_CAP = 80.0    # uniform clamp on exponent arguments (fp32 exp overflows ~88.7)
DOT_PREC = "ieee"


@triton.jit
def _exp(x, EXP_CAP: tl.constexpr = 80.0):
    return tl.math.exp2(tl.minimum(x, EXP_CAP) * 1.4426950408889634)


@triton.jit
def _tri_inv(N, BTD: tl.constexpr, LOG_BT: tl.constexpr, PREC: tl.constexpr):
    """(I - N)^{-1} for strictly lower triangular N: prod_j (I + N^(2^j))."""
    i = tl.arange(0, BTD)
    P = (i[:, None] == i[None, :]).to(tl.float32) + N
    R = tl.dot(N, N, input_precision=PREC)
    for _ in tl.static_range(LOG_BT - 1):
        P = P + tl.dot(P, R, input_precision=PREC)
        R = tl.dot(R, R, input_precision=PREC)
    return P


@triton.jit
def _build_A(q, k, a, b, G, gl, g, ot, BTD: tl.constexpr, BK: tl.constexpr,
             PREC: tl.constexpr):
    """The four intra-chunk matrices as dense matmuls.

    A_qk[t,s] = <q[t], k[s]> exp(G[t]-G[s]) for s <= t, and likewise A_qb with
    beta, while A_ak / A_ab use exp(Gp[t]-G[s]) on s < t with Gp = G - gk.
    The ratio is factorised through the chunk-centred exponent so that the row
    factor exp(Gc) and the column factor exp(-Gc) each span only half the
    chunk's total decay.
    """
    Gc = G - 0.5 * gl[None, :]
    ec = _exp(-Gc)
    qc = q * _exp(Gc)
    ac = a * _exp(Gc - g)
    kc = k * ec
    bc = b * ec
    mi = ot[:, None] >= ot[None, :]
    ms = ot[:, None] > ot[None, :]
    return (tl.where(mi, tl.dot(qc, tl.trans(kc), input_precision=PREC), 0.0),
            tl.where(mi, tl.dot(qc, tl.trans(bc), input_precision=PREC), 0.0),
            tl.where(ms, tl.dot(ac, tl.trans(kc), input_precision=PREC), 0.0),
            tl.where(ms, tl.dot(ac, tl.trans(bc), input_precision=PREC), 0.0),
            qc, ac, kc, bc)


@triton.jit
def dplr_intra_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, g_ptr,
    W_ptr, Uk_ptr, M_ptr, Aqb_ptr, Aqk_ptr,
    T, K, V, H, NT, scale,
    BT: tl.constexpr, BTD: tl.constexpr, LOG_BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    PREC: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    ot = tl.arange(0, BTD)
    ok = tl.arange(0, BK)
    ov = tl.arange(0, BV)
    t = i_t * BT + ot
    mt = (t < T) & (ot < BT)
    mk = ok < K
    mv = ov < V
    m2 = mt[:, None] & mk[None, :]
    mvv = mt[:, None] & mv[None, :]

    base = (i_b.to(tl.int64) * T) * H + i_h
    row = (base + t.to(tl.int64) * H)
    offk = row[:, None] * K + ok[None, :]
    offv = row[:, None] * V + ov[None, :]

    q = tl.load(q_ptr + offk, mask=m2, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + offk, mask=m2, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offk, mask=m2, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offk, mask=m2, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offk, mask=m2, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + offv, mask=mvv, other=0.0).to(tl.float32)

    G = tl.cumsum(g, axis=0)
    gl = tl.sum(g, axis=0)
    Gp = G - g
    at = a * _exp(Gp)
    A_qk, A_qb, A_ak, A_ab, _q, _a, _k, _b = _build_A(
        q * scale, k, a, b, G, gl, g, ot, BTD, BK, PREC)

    M = _tri_inv(A_ab, BTD, LOG_BT, PREC)
    W = tl.dot(M, at, input_precision=PREC)
    Uk = tl.dot(M, tl.dot(A_ak, v, input_precision=PREC), input_precision=PREC)

    pc = i_bh.to(tl.int64) * NT + i_t
    tl.store(W_ptr + pc * (BTD * K) + ot[:, None] * K + ok[None, :], W, mask=m2)
    tl.store(Uk_ptr + pc * (BTD * V) + ot[:, None] * V + ov[None, :], Uk, mask=mvv)
    tl.store(M_ptr + pc * (BTD * BTD) + ot[:, None] * BTD + ot[None, :], M)
    tl.store(Aqb_ptr + pc * (BTD * BTD) + ot[:, None] * BTD + ot[None, :], A_qb)
    tl.store(Aqk_ptr + pc * (BTD * BTD) + ot[:, None] * BTD + ot[None, :], A_qk)


@triton.jit
def dplr_scan_kernel(
    k_ptr, v_ptr, b_ptr, g_ptr, W_ptr, Uk_ptr, U_ptr, S_ptr, Sl_ptr,
    T, K, V, H, NT,
    BT: tl.constexpr, BTD: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, PREC: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    ot = tl.arange(0, BTD)
    ok = tl.arange(0, BK)
    ov = i_v * BV + tl.arange(0, BV)
    mk = ok < K
    mv = ov < V
    mkv = mk[:, None] & mv[None, :]
    base = (i_b.to(tl.int64) * T) * H + i_h

    S = tl.zeros([BK, BV], dtype=tl.float32)
    cS = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        t = i_t * BT + ot
        mt = (t < T) & (ot < BT)
        m2 = mt[:, None] & mk[None, :]
        mvv = mt[:, None] & mv[None, :]
        row = base + t.to(tl.int64) * H
        offk = row[:, None] * K + ok[None, :]
        offv = row[:, None] * V + ov[None, :]

        k = tl.load(k_ptr + offk, mask=m2, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offk, mask=m2, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offk, mask=m2, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + offv, mask=mvv, other=0.0).to(tl.float32)
        G = tl.cumsum(g, axis=0)
        gl = tl.sum(g, axis=0)

        pc = i_bh.to(tl.int64) * NT + i_t
        W = tl.load(W_ptr + pc * (BTD * K) + ot[:, None] * K + ok[None, :],
                    mask=m2, other=0.0)
        Uk = tl.load(Uk_ptr + pc * (BTD * V) + ot[:, None] * V + ov[None, :],
                     mask=mvv, other=0.0)

        # Checkpoint the chunk-entry state as an unevaluated float32 pair
        # (S, cS): S holds the leading bits and cS the residual the running
        # compensation has accumulated.  Consumers contract against both, so
        # the state carry keeps roughly twice the float32 significand while
        # every stored word stays float32.
        tl.store(S_ptr + pc * (K * V) + ok[:, None] * V + ov[None, :], S, mask=mkv)
        tl.store(Sl_ptr + pc * (K * V) + ok[:, None] * V + ov[None, :], cS, mask=mkv)

        U = Uk + tl.dot(W, S, input_precision=PREC) \
               + tl.dot(W, cS, input_precision=PREC)
        tl.store(U_ptr + pc * (BTD * V) + ot[:, None] * V + ov[None, :], U, mask=mvv)

        # Advance the boundary state with a running compensation.  The state
        # is float32 and grows to many orders of magnitude above the per-chunk
        # increments; without carrying the residual, the bits dropped at each
        # of the ~T/BT carries accumulate into the output.
        eh = _exp(gl[None, :] - G)
        dS = tl.dot(tl.trans(b * eh), U, input_precision=PREC)
        dS += tl.dot(tl.trans(k * eh), v, input_precision=PREC)
        E = _exp(gl)[:, None]
        Sd = S * E
        cS = cS * E + (tl.math.fma(S, E, -Sd))
        tot = Sd + dS
        cS += tl.where(tl.abs(Sd) >= tl.abs(dS),
                       (Sd - tot) + dS, (dS - tot) + Sd)
        S = tot


@triton.jit
def dplr_out_kernel(
    q_ptr, v_ptr, g_ptr, U_ptr, S_ptr, Sl_ptr, Aqb_ptr, Aqk_ptr, o_ptr,
    T, K, V, H, NT, scale,
    BT: tl.constexpr, BTD: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_v = tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    ot = tl.arange(0, BTD)
    ok = tl.arange(0, BK)
    ov = i_v * BV + tl.arange(0, BV)
    t = i_t * BT + ot
    mt = (t < T) & (ot < BT)
    mk = ok < K
    mv = ov < V
    m2 = mt[:, None] & mk[None, :]
    mvv = mt[:, None] & mv[None, :]

    base = (i_b.to(tl.int64) * T) * H + i_h
    row = base + t.to(tl.int64) * H
    offk = row[:, None] * K + ok[None, :]
    offv = row[:, None] * V + ov[None, :]

    q = tl.load(q_ptr + offk, mask=m2, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offk, mask=m2, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + offv, mask=mvv, other=0.0).to(tl.float32)
    G = tl.cumsum(g, axis=0)
    gl = tl.sum(g, axis=0)
    qt = q * scale * _exp(G)

    pc = i_bh.to(tl.int64) * NT + i_t
    S = tl.load(S_ptr + pc * (K * V) + ok[:, None] * V + ov[None, :],
                mask=mk[:, None] & mv[None, :], other=0.0)
    Sl = tl.load(Sl_ptr + pc * (K * V) + ok[:, None] * V + ov[None, :],
                 mask=mk[:, None] & mv[None, :], other=0.0)
    U = tl.load(U_ptr + pc * (BTD * V) + ot[:, None] * V + ov[None, :],
                mask=mvv, other=0.0)
    A_qb = tl.load(Aqb_ptr + pc * (BTD * BTD) + ot[:, None] * BTD + ot[None, :])
    A_qk = tl.load(Aqk_ptr + pc * (BTD * BTD) + ot[:, None] * BTD + ot[None, :])

    o = tl.dot(qt, S, input_precision=PREC)
    o += tl.dot(qt, Sl, input_precision=PREC)
    o += tl.dot(A_qb, U, input_precision=PREC)
    o += tl.dot(A_qk, v, input_precision=PREC)
    tl.store(o_ptr + offv, o.to(o_ptr.dtype.element_ty), mask=mvv)


@triton.jit
def dplr_dscan_kernel(
    do_ptr, q_ptr, b_ptr, g_ptr, W_ptr, Aqb_ptr, dU_ptr, Z_ptr,
    T, K, V, H, NT, scale,
    BT: tl.constexpr, BTD: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, PREC: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    ot = tl.arange(0, BTD)
    ok = tl.arange(0, BK)
    ov = i_v * BV + tl.arange(0, BV)
    mk = ok < K
    mv = ov < V
    mkv = mk[:, None] & mv[None, :]
    base = (i_b.to(tl.int64) * T) * H + i_h

    Z = tl.zeros([BK, BV], dtype=tl.float32)
    cZ = tl.zeros([BK, BV], dtype=tl.float32)
    for i_r in range(NT):
        i_t = NT - 1 - i_r
        t = i_t * BT + ot
        mt = (t < T) & (ot < BT)
        m2 = mt[:, None] & mk[None, :]
        mvv = mt[:, None] & mv[None, :]
        row = base + t.to(tl.int64) * H
        offk = row[:, None] * K + ok[None, :]
        offv = row[:, None] * V + ov[None, :]

        q = tl.load(q_ptr + offk, mask=m2, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offk, mask=m2, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offk, mask=m2, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + offv, mask=mvv, other=0.0).to(tl.float32)
        G = tl.cumsum(g, axis=0)
        gl = tl.sum(g, axis=0)
        qt = q * scale * _exp(G)
        bh = b * _exp(gl[None, :] - G)

        pc = i_bh.to(tl.int64) * NT + i_t
        W = tl.load(W_ptr + pc * (BTD * K) + ot[:, None] * K + ok[None, :],
                    mask=m2, other=0.0)
        A_qb = tl.load(Aqb_ptr + pc * (BTD * BTD) + ot[:, None] * BTD + ot[None, :])

        # checkpoint the chunk-exit state gradient, float32 (compensated value)
        Zc = Z + cZ
        tl.store(Z_ptr + pc * (K * V) + ok[:, None] * V + ov[None, :], Zc, mask=mkv)

        dU = tl.dot(tl.trans(A_qb), do, input_precision=PREC)
        dU += tl.dot(bh, Zc, input_precision=PREC)
        tl.store(dU_ptr + pc * (BTD * V) + ot[:, None] * V + ov[None, :], dU, mask=mvv)

        # Same compensated carry as the forward scan: the state gradient
        # spans many orders of magnitude across the reverse sweep, and the
        # bits a plain float32 accumulation drops are the answer in the rows
        # where the incoming and outgoing contributions nearly cancel.
        dS = tl.dot(tl.trans(qt), do, input_precision=PREC)
        dS += tl.dot(tl.trans(W), dU, input_precision=PREC)
        Eg = _exp(gl)[:, None]
        Zd = Z * Eg
        cZ = cZ * Eg + tl.math.fma(Z, Eg, -Zd)
        tot = Zd + dS
        cZ += tl.where(tl.abs(Zd) >= tl.abs(dS),
                       (Zd - tot) + dS, (dS - tot) + Zd)
        Z = tot


@triton.jit
def dplr_dintra_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, g_ptr, do_ptr,
    W_ptr, Uk_ptr, U_ptr, M_ptr, S_ptr, Sl_ptr, Z_ptr, dU_ptr,
    dq_ptr, dk_ptr, dv_ptr, da_ptr, db_ptr, dg_ptr,
    T, K, V, H, NT, scale, NV,
    BT: tl.constexpr, BTD: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    ot = tl.arange(0, BTD)
    ok = tl.arange(0, BK)
    t = i_t * BT + ot
    mt = (t < T) & (ot < BT)
    mk = ok < K
    m2 = mt[:, None] & mk[None, :]
    base = (i_b.to(tl.int64) * T) * H + i_h
    row = base + t.to(tl.int64) * H
    offk = row[:, None] * K + ok[None, :]
    pc = i_bh.to(tl.int64) * NT + i_t

    q = tl.load(q_ptr + offk, mask=m2, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + offk, mask=m2, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offk, mask=m2, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offk, mask=m2, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offk, mask=m2, other=0.0).to(tl.float32)

    G = tl.cumsum(g, axis=0)
    gl = tl.sum(g, axis=0)
    Gp = G - g
    eq = _exp(G)
    ea = _exp(Gp)
    eh = _exp(gl[None, :] - G)
    qt = q * scale * eq
    at = a * ea
    kh = k * eh
    bh = b * eh
    E = _exp(gl)

    ms = ot[:, None] > ot[None, :]
    mi = ot[:, None] >= ot[None, :]
    A_qk, A_qb_u, A_ak, A_ab_u, qc, ac, kc, bc = _build_A(
        q * scale, k, a, b, G, gl, g, ot, BTD, BK, PREC)
    M = tl.load(M_ptr + pc * (BTD * BTD) + ot[:, None] * BTD + ot[None, :])
    W = tl.load(W_ptr + pc * (BTD * K) + ot[:, None] * K + ok[None, :],
                mask=m2, other=0.0)
    Mt = tl.trans(M)

    dW = tl.zeros([BTD, BK], dtype=tl.float32)
    dqt_s = tl.zeros([BTD, BK], dtype=tl.float32)
    dbh = tl.zeros([BTD, BK], dtype=tl.float32)
    dkh = tl.zeros([BTD, BK], dtype=tl.float32)
    dAab = tl.zeros([BTD, BTD], dtype=tl.float32)
    dAak = tl.zeros([BTD, BTD], dtype=tl.float32)
    dAqb = tl.zeros([BTD, BTD], dtype=tl.float32)
    dAqk = tl.zeros([BTD, BTD], dtype=tl.float32)
    dgl_dec = tl.zeros([BK], dtype=tl.float32)
    cgl = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(NV):
        ov = i_v * BV + tl.arange(0, BV)
        mv = ov < V
        mvv = mt[:, None] & mv[None, :]
        mkv = mk[:, None] & mv[None, :]
        offv = row[:, None] * V + ov[None, :]

        v = tl.load(v_ptr + offv, mask=mvv, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + offv, mask=mvv, other=0.0).to(tl.float32)
        U = tl.load(U_ptr + pc * (BTD * V) + ot[:, None] * V + ov[None, :],
                    mask=mvv, other=0.0)
        Uk = tl.load(Uk_ptr + pc * (BTD * V) + ot[:, None] * V + ov[None, :],
                     mask=mvv, other=0.0)
        dU = tl.load(dU_ptr + pc * (BTD * V) + ot[:, None] * V + ov[None, :],
                     mask=mvv, other=0.0)
        S = tl.load(S_ptr + pc * (K * V) + ok[:, None] * V + ov[None, :],
                    mask=mkv, other=0.0)
        Slo = tl.load(Sl_ptr + pc * (K * V) + ok[:, None] * V + ov[None, :],
                      mask=mkv, other=0.0)
        Z = tl.load(Z_ptr + pc * (K * V) + ok[:, None] * V + ov[None, :],
                    mask=mkv, other=0.0)
        # <Z, S> summed over V, compensated: Z and S are both large and the
        # products alternate sign, so the reduction cancels heavily.
        p = Z * S
        inc_d = tl.sum(p, axis=1) + tl.sum(tl.math.fma(Z, S, -p), axis=1) \
            + tl.sum(Z * Slo, axis=1)
        tot_d = dgl_dec + inc_d
        cgl += tl.where(tl.abs(dgl_dec) >= tl.abs(inc_d),
                        (dgl_dec - tot_d) + inc_d, (inc_d - tot_d) + dgl_dec)
        dgl_dec = tot_d

        Slt = tl.trans(S)
        Slt2 = tl.trans(Slo)
        dW += tl.dot(dU, Slt, input_precision=PREC)
        dW += tl.dot(dU, Slt2, input_precision=PREC)
        dqt_s += tl.dot(do, Slt, input_precision=PREC)
        dqt_s += tl.dot(do, Slt2, input_precision=PREC)

        dY = tl.dot(Mt, dU, input_precision=PREC)
        dAab += tl.dot(dY, tl.trans(Uk), input_precision=PREC)
        dAak += tl.dot(dY, tl.trans(v), input_precision=PREC)
        dAqb += tl.dot(do, tl.trans(U), input_precision=PREC)
        dAqk += tl.dot(do, tl.trans(v), input_precision=PREC)

        dv = tl.dot(tl.trans(A_qk), do, input_precision=PREC)
        dv += tl.dot(tl.trans(A_ak), dY, input_precision=PREC)
        dv += tl.dot(kh, Z, input_precision=PREC)
        tl.store(dv_ptr + offv, dv, mask=mvv)

        dbh += tl.dot(U, tl.trans(Z), input_precision=PREC)
        dkh += tl.dot(v, tl.trans(Z), input_precision=PREC)

    dat_W = tl.dot(Mt, dW, input_precision=PREC)
    dAab = tl.where(ms, dAab + tl.dot(dat_W, tl.trans(W), input_precision=PREC), 0.0)
    dAak = tl.where(ms, dAak, 0.0)
    dAqb = tl.where(mi, dAqb, 0.0)
    dAqk = tl.where(mi, dAqk, 0.0)

    # Per-pair row/column contractions.  Each A matrix couples a row family
    # (qt or at) to a column family (kt or bt) through an exponent difference,
    # so its whole contribution to the gate gradient is banded.
    # Contract the four A-gradients back onto the inputs.  Each A couples a
    # row family (qc or ac) to a column family (kc or bc) through the centred
    # exponent, so the contraction is four pairs of matmuls.
    rw1 = tl.dot(dAqk, kc, input_precision=PREC)
    rw2 = tl.dot(dAqb, bc, input_precision=PREC)
    rw3 = tl.dot(dAak, kc, input_precision=PREC)
    rw4 = tl.dot(dAab, bc, input_precision=PREC)
    cl1 = tl.dot(tl.trans(dAqk), qc, input_precision=PREC)
    cl2 = tl.dot(tl.trans(dAqb), qc, input_precision=PREC)
    cl3 = tl.dot(tl.trans(dAak), ac, input_precision=PREC)
    cl4 = tl.dot(tl.trans(dAab), ac, input_precision=PREC)

    Gc = G - 0.5 * gl[None, :]
    ecr = _exp(Gc)
    eca = _exp(Gc - g)
    ecc = _exp(-Gc)

    dq = (dqt_s * eq + (rw1 + rw2) * ecr) * scale
    da = dat_W * ea + (rw3 + rw4) * eca
    dk = (cl1 + cl3) * ecc + dkh * eh
    db = (cl2 + cl4) * ecc + dbh * eh

    # Gate gradient.  Pair (t, s) feeds gate rows s < u <= t for the q-rows and
    # s < u < t for the alpha-rows.  Collapsing that band into
    # (row suffix) - (column suffix) is exact on paper but loses everything to
    # cancellation here, because each suffix carries the magnitude of the whole
    # tile while the difference is small.  Walking the source row s and adding
    # only the rows the band covers keeps every partial sum the size of the
    # result it builds.
    # The same walk also yields dk and dbeta exactly: column s of the
    # contraction is the sum over t of the very products the band needs, so
    # both come out of one pass with the pairwise factor exp(G[t] - G[s])
    # formed from the difference of the cumsums rather than as a ratio.
    dg_band = tl.zeros([BTD, BK], dtype=tl.float32)
    cg = tl.zeros([BTD, BK], dtype=tl.float32)
    Gp = G - g
    for s in tl.static_range(BTD):
        sel = (ot == s).to(tl.float32)
        Gs = tl.sum(G * sel[:, None], axis=0)
        ks = tl.sum(k * sel[:, None], axis=0)
        bs = tl.sum(b * sel[:, None], axis=0)
        wq = _exp(G - Gs[None, :])
        wa = _exp(Gp - Gs[None, :])
        c1 = tl.sum(dAqk * sel[None, :], axis=1)
        c2 = tl.sum(dAqb * sel[None, :], axis=1)
        c3 = tl.sum(dAak * sel[None, :], axis=1)
        c4 = tl.sum(dAab * sel[None, :], axis=1)

        qs = q * scale
        fq = qs * wq * (c1[:, None] * ks[None, :] + c2[:, None] * bs[None, :])
        fa = a * wa * (c3[:, None] * ks[None, :] + c4[:, None] * bs[None, :])
        sq = tl.flip(tl.cumsum(tl.flip(fq, 0), axis=0), 0)
        sa = tl.flip(tl.cumsum(tl.flip(fa, 0), axis=0), 0) - fa
        inc = tl.where(ot[:, None] > s, sq + sa, 0.0)
        tot = dg_band + inc
        cg += tl.where(tl.abs(dg_band) >= tl.abs(inc),
                       (dg_band - tot) + inc, (inc - tot) + dg_band)
        dg_band = tot

    # Terms that couple to the carried state: the read of the incoming state
    # (a suffix over rows), the writes through kh/bh, which carry exp(gl - G)
    # and so land on a strict prefix, and the whole-chunk decay exp(gl).
    # dqt_s and dat_W are contractions against the incoming state, so these
    # rows can be orders of magnitude above the total; accumulate their
    # suffix and the kh/bh prefix with the same compensation the band uses.
    srq = dqt_s * (q * scale * eq)
    sra = dat_W * at
    hw = dkh * kh + dbh * bh

    dg = dg_band

    inc = tl.flip(tl.cumsum(tl.flip(srq, 0), axis=0), 0)
    tot = dg + inc
    cg += tl.where(tl.abs(dg) >= tl.abs(inc), (dg - tot) + inc, (inc - tot) + dg)
    dg = tot

    inc = tl.flip(tl.cumsum(tl.flip(sra, 0), axis=0), 0) - sra
    tot = dg + inc
    cg += tl.where(tl.abs(dg) >= tl.abs(inc), (dg - tot) + inc, (inc - tot) + dg)
    dg = tot

    inc = tl.cumsum(hw, axis=0) - hw
    tot = dg + inc
    cg += tl.where(tl.abs(dg) >= tl.abs(inc), (dg - tot) + inc, (inc - tot) + dg)
    dg = tot

    inc = tl.broadcast_to(((dgl_dec + cgl) * E)[None, :], [BTD, BK])
    tot = dg + inc
    cg += tl.where(tl.abs(dg) >= tl.abs(inc), (dg - tot) + inc, (inc - tot) + dg)
    dg = tot + cg

    tl.store(dq_ptr + offk, dq, mask=m2)
    tl.store(dk_ptr + offk, dk, mask=m2)
    tl.store(da_ptr + offk, da, mask=m2)
    tl.store(db_ptr + offk, db, mask=m2)
    tl.store(dg_ptr + offk, dg, mask=m2)


def _pow2(x):
    return 1 << (x - 1).bit_length()


def _launch_fwd(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H
    NT = triton.cdiv(T, BT)
    BK = max(_pow2(K), 16)
    BVf = max(_pow2(V), 16)
    scale = K ** -0.5
    dev = q.device

    W = torch.empty(BH, NT, BTD, K, dtype=torch.float32, device=dev)
    Uk = torch.empty(BH, NT, BTD, V, dtype=torch.float32, device=dev)
    M = torch.empty(BH, NT, BTD, BTD, dtype=torch.float32, device=dev)
    Aqb = torch.empty(BH, NT, BTD, BTD, dtype=torch.float32, device=dev)
    Aqk = torch.empty(BH, NT, BTD, BTD, dtype=torch.float32, device=dev)
    dplr_intra_kernel[(NT, BH)](
        q, k, v, alpha, beta, gk, W, Uk, M, Aqb, Aqk,
        T, K, V, H, NT, scale,
        BT=BT, BTD=BTD, LOG_BT=LOG_BT, BK=BK, BV=BVf, PREC=DOT_PREC,
        num_warps=4, num_stages=2,
    )

    U = torch.empty(BH, NT, BTD, V, dtype=torch.float32, device=dev)
    S = torch.empty(BH, NT, K, V, dtype=torch.float32, device=dev)
    Sl = torch.empty(BH, NT, K, V, dtype=torch.float32, device=dev)
    BVs = min(BV_SCAN, BVf)
    dplr_scan_kernel[(BH, triton.cdiv(V, BVs))](
        k, v, beta, gk, W, Uk, U, S, Sl,
        T, K, V, H, NT,
        BT=BT, BTD=BTD, BK=BK, BV=BVs, PREC=DOT_PREC,
        num_warps=4, num_stages=2,
    )

    # Output tensor only; it is the operator's result in the input dtype, not
    # part of the recurrent state, which is float32 end to end.
    o = v.new_empty(B, T, H, V)
    BVo = min(64, BVf)
    dplr_out_kernel[(NT, BH, triton.cdiv(V, BVo))](
        q, v, gk, U, S, Sl, Aqb, Aqk, o,
        T, K, V, H, NT, scale,
        BT=BT, BTD=BTD, BK=BK, BV=BVo, PREC=DOT_PREC,
        num_warps=4, num_stages=2,
    )
    return o, W, Uk, M, Aqb, Aqk, U, S, Sl


def _launch_bwd(do, q, k, v, alpha, beta, gk, W, Uk, M, Aqb, Aqk, U, S, Sl):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H
    NT = triton.cdiv(T, BT)
    BK = max(_pow2(K), 16)
    BVf = max(_pow2(V), 16)
    scale = K ** -0.5
    dev = q.device

    dU = torch.empty(BH, NT, BTD, V, dtype=torch.float32, device=dev)
    Z = torch.empty(BH, NT, K, V, dtype=torch.float32, device=dev)
    BVs = min(BV_SCAN, BVf)
    dplr_dscan_kernel[(BH, triton.cdiv(V, BVs))](
        do, q, beta, gk, W, Aqb, dU, Z,
        T, K, V, H, NT, scale,
        BT=BT, BTD=BTD, BK=BK, BV=BVs, PREC=DOT_PREC,
        num_warps=4, num_stages=2,
    )

    dq = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dk = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dv = torch.empty(B, T, H, V, dtype=torch.float32, device=dev)
    da = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    db = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    dg = torch.empty(B, T, H, K, dtype=torch.float32, device=dev)
    BVb = min(BV_BWD, BVf)
    dplr_dintra_kernel[(NT, BH)](
        q, k, v, alpha, beta, gk, do,
        W, Uk, U, M, S, Sl, Z, dU,
        dq, dk, dv, da, db, dg,
        T, K, V, H, NT, scale, triton.cdiv(V, BVb),
        BT=BT, BTD=BTD, BK=BK, BV=BVb, PREC=DOT_PREC,
        num_warps=4, num_stages=2,
    )
    return dq, dk, dv, da, db, dg


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        o, W, Uk, M, Aqb, Aqk, U, S, Sl = _launch_fwd(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, W, Uk, M, Aqb, Aqk,
                              U, S, Sl)
        return o

    @staticmethod
    def backward(ctx, do):
        (q, k, v, alpha, beta, gk, W, Uk, M, Aqb, Aqk, U, S,
         Sl) = ctx.saved_tensors
        dq, dk, dv, da, db, dg = _launch_bwd(
            do.contiguous(), q, k, v, alpha, beta, gk, W, Uk, M, Aqb, Aqk,
            U, S, Sl
        )
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                da.to(alpha.dtype), db.to(beta.dtype), dg.to(gk.dtype))


def chunked_dplr_delta(
    q: torch.Tensor,      # [B, T, H, K]
    k: torch.Tensor,      # [B, T, H, K]
    v: torch.Tensor,      # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]
    beta: torch.Tensor,   # [B, T, H, K]
    gk: torch.Tensor,     # [B, T, H, K]
) -> torch.Tensor:        # [B, T, H, V]
    return _Dplr.apply(q, k, v, alpha, beta, gk)
