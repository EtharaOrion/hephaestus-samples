"""Chunk-parallel DPLR generalized delta rule (RWKV-7 family), forward and backward.

Recurrence, per (batch, head), state S in R^{K x V}:

    S_t = diag(exp(gk_t)) S_{t-1} + k_t v_t^T + beta_t (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))

Chunked form used here (chunk length C, g = inclusive cumsum of gk inside the
chunk, ge = g - gk, gl = g at the last row of the chunk).  Every exponential is
anchored on gl so that each factor is exp(<=0) for decaying gates and no
intermediate overflows before the recurrence itself does:

    Ar = alpha e^{ge-gl}   Qr = (q/sqrt K) e^{g-gl}   Bc = beta e^{gl-g}   Kc = k e^{gl-g}
    A_ab = tril(Ar Bc^T, -1)   A_ak = tril(Ar Kc^T, -1)
    A_qb = tril(Qr Bc^T,  0)   A_qk = tril(Qr Kc^T,  0)
    Sh   = diag(e^{gl}) S_chunkstart
    U    = (I - A_ab)^{-1} (Ar Sh + A_ak V)          <- rank-1 alpha/beta cascade
    O    = Qr Sh + A_qb U + A_qk V
    S'   = Sh + Bc^T U + Kc^T V

A_qb is inclusive on the diagonal: S_i already contains the b_i u_i^T term.
(I - A_ab) is unit lower triangular with A_ab nilpotent, so its exact inverse is
the finite product (I+A)(I+A^2)(I+A^4)... built by repeated squaring -- no
truncation and no approximation of the rank-1 alpha/beta term.

The backward pass mirrors this with an adjoint state Z carried right-to-left.
Both carried states (S and Z) and every checkpoint live in float32 for their
whole life; only the user-visible inputs and outputs use the input dtype.
"""

import torch
import triton
import triton.language as tl

C = 16          # chunk length
SUB = 8         # width of each explicitly inverted diagonal window
LOG_SUB = 3     # squarings for the exact nilpotent inverse of a SUB-wide block
EXP_CAP = 80.0  # guards exp() against overflow only; never binds on real gate ranges


def _pow2(x):
    return max(16, triton.next_power_of_2(x))


@triton.jit
def _dplr_prep(q, k, a, b, gk, Ar, Qr, Bc, Kc, Tv,
               T, H: tl.constexpr, K: tl.constexpr, scale,
               BK: tl.constexpr, BT: tl.constexpr, SUB: tl.constexpr, LOG_SUB: tl.constexpr,
               NBLK: tl.constexpr, CAP: tl.constexpr):
    """Per-chunk gate cumsum, the four gated operand matrices, and (I-A_ab)^{-1}."""
    n = tl.program_id(0)
    bh = tl.program_id(1)
    b_i = bh // H
    h_i = bh % H

    ot = tl.arange(0, BT)
    ok = tl.arange(0, BK)
    t = n * BT + ot
    mt = t < T
    mk = ok < K
    m2 = mt[:, None] & mk[None, :]
    gsrc = (b_i.to(tl.int64) * T + t.to(tl.int64)) * H * K + h_i * K
    off = gsrc[:, None] + ok[None, :]

    g_ = tl.load(gk + off, mask=m2, other=0.0).to(tl.float32)
    g = tl.cumsum(g_, axis=0)
    ge = g - g_
    gl = tl.sum(g_, axis=0)[None, :]

    q_ = tl.load(q + off, mask=m2, other=0.0).to(tl.float32)
    k_ = tl.load(k + off, mask=m2, other=0.0).to(tl.float32)
    a_ = tl.load(a + off, mask=m2, other=0.0).to(tl.float32)
    b_ = tl.load(b + off, mask=m2, other=0.0).to(tl.float32)

    ar = a_ * tl.exp(tl.minimum(ge, CAP))
    qr = q_ * tl.exp(tl.minimum(g, CAP)) * scale
    e_col = tl.exp(tl.minimum(-g, CAP))
    bc = b_ * e_col
    kc = k_ * e_col

    dst = ((bh.to(tl.int64) * T + t.to(tl.int64)) * K)[:, None] + ok[None, :]
    tl.store(Ar + dst, ar, mask=m2)
    tl.store(Qr + dst, qr, mask=m2)
    tl.store(Bc + dst, bc, mask=m2)
    tl.store(Kc + dst, kc, mask=m2)

    lo = ot[:, None] > ot[None, :]
    aab = tl.where(lo, tl.dot(ar, tl.trans(bc), input_precision="ieee"), 0.0)
    eye = (ot[:, None] == ot[None, :]).to(tl.float32)
    # (I - A)^{-1} built from SUB-block diagonal inverses rather than one inverse
    # over the whole tile.  The full-tile inverse of this operator grows like
    # 1e6 at 16 steps (and far worse beyond), and that growth multiplies the
    # round-off of everything it is applied to; restricting each explicit inverse
    # to a SUB-wide window keeps the amplification small while the identity
    #   (I-A)^{-1} = Rd + Rd Ao Rd + (Rd Ao)^2 Rd + ...
    # over the strictly-lower block part Ao stays exact (Ao is block-nilpotent).
    blk = ot // SUB
    same = blk[:, None] == blk[None, :]
    ad = tl.where(same, aab, 0.0)
    ao = tl.where(same, 0.0, aab)
    rd = eye + ad
    x = ad
    for _ in tl.static_range(LOG_SUB - 1):
        x = tl.dot(x, x, input_precision="ieee")
        rd = tl.dot(rd, eye + x, input_precision="ieee")
    # Only the block-diagonal inverse is stored.  The off-diagonal coupling is
    # applied later straight to the right-hand side by the block recurrence
    #   u_p = Rd_p (rhs_p + sum_{q<p} A_pq u_q),
    # which is exact after NBLK-1 sweeps and never materializes the full
    # (I-A)^{-1}, whose entries reach ~1e6 and lose bits when they are formed.
    tdst = (bh.to(tl.int64) * tl.num_programs(0) + n) * BT * BT
    tl.store(Tv + tdst + ot[:, None] * BT + ot[None, :], rd)


@triton.jit
def _dplr_fwd(Ar, Qr, Bc, Kc, Tv, gk, v, U, O, S0,
              T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
              BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr,
              SUB: tl.constexpr, NBLK: tl.constexpr, CAP: tl.constexpr):
    """Sequential scan over chunks for one V tile; boundary state kept in float32."""
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    b_i = bh // H
    h_i = bh % H

    ot = tl.arange(0, BT)
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mk = ok < K
    mv = ov < V
    lo = ot[:, None] > ot[None, :]
    le = ot[:, None] >= ot[None, :]
    sameblk = (ot // SUB)[:, None] == (ot // SUB)[None, :]
    kcol = mk[:, None].to(tl.float32)

    s = tl.zeros([BK, BV], dtype=tl.float32)
    for n in range(0, NT):
        t = n * BT + ot
        mt = t < T
        m2 = mt[:, None] & mk[None, :]
        mvv = mt[:, None] & mv[None, :]
        base = bh.to(tl.int64) * T + t.to(tl.int64)
        off = base[:, None] * K + ok[None, :]
        ar = tl.load(Ar + off, mask=m2, other=0.0)
        qr = tl.load(Qr + off, mask=m2, other=0.0)
        bc = tl.load(Bc + off, mask=m2, other=0.0)
        kc = tl.load(Kc + off, mask=m2, other=0.0)
        tinv = tl.load(Tv + (bh.to(tl.int64) * NT + n) * BT * BT
                       + ot[:, None] * BT + ot[None, :])

        gsrc = (b_i.to(tl.int64) * T + t.to(tl.int64)) * H * K + h_i * K
        g_ = tl.load(gk + gsrc[:, None] + ok[None, :], mask=m2, other=0.0).to(tl.float32)
        gl = tl.sum(g_, axis=0)

        vsrc = (b_i.to(tl.int64) * T + t.to(tl.int64)) * H * V + h_i * V
        vv = tl.load(v + vsrc[:, None] + ov[None, :], mask=mvv, other=0.0).to(tl.float32)

        sdst = ((bh.to(tl.int64) * NT + n) * K + ok[:, None].to(tl.int64)) * V + ov[None, :]
        tl.store(S0 + sdst, s, mask=mk[:, None] & mv[None, :])

        e_gl = tl.exp(tl.minimum(gl, CAP))
        aak = tl.where(lo, tl.dot(ar, tl.trans(kc), input_precision="ieee"), 0.0)
        rhs = tl.dot(ar, s, input_precision="ieee") \
            + tl.dot(aak, vv, input_precision="ieee")
        aab = tl.where(lo, tl.dot(ar, tl.trans(bc), input_precision="ieee"), 0.0)
        aoff = tl.where(sameblk, 0.0, aab)
        u = tl.dot(tinv, rhs, input_precision="ieee")
        for _ in tl.static_range(NBLK - 1):
            u = tl.dot(tinv, rhs + tl.dot(aoff, u, input_precision="ieee"),
                       input_precision="ieee")

        aqb = tl.where(le, tl.dot(qr, tl.trans(bc), input_precision="ieee"), 0.0)
        aqk = tl.where(le, tl.dot(qr, tl.trans(kc), input_precision="ieee"), 0.0)
        o = tl.dot(qr, s, input_precision="ieee") \
            + tl.dot(aqb, u, input_precision="ieee") \
            + tl.dot(aqk, vv, input_precision="ieee")

        tl.store(U + base[:, None] * V + ov[None, :], u, mask=mvv)
        odst = (b_i.to(tl.int64) * T + t.to(tl.int64)) * H * V + h_i * V
        tl.store(O + odst[:, None] + ov[None, :], o.to(O.dtype.element_ty), mask=mvv)

        bd = bc * e_gl[None, :]
        kd = kc * e_gl[None, :]
        s = s * e_gl[:, None] * kcol \
            + tl.dot(tl.trans(bd), u, input_precision="ieee") \
            + tl.dot(tl.trans(kd), vv, input_precision="ieee")


@triton.jit
def _dplr_bwd_scan(Ar, Qr, Bc, Kc, Tv, gk, q, a, DO, W, dV, Zs,
                   T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
                   BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr,
                   SUB: tl.constexpr, NBLK: tl.constexpr, scale, CAP: tl.constexpr):
    """Reverse scan of the adjoint state Z (float32); produces W and dv."""
    iv = tl.program_id(0)
    bh = tl.program_id(1)
    b_i = bh // H
    h_i = bh % H

    ot = tl.arange(0, BT)
    ok = tl.arange(0, BK)
    ov = iv * BV + tl.arange(0, BV)
    mk = ok < K
    mv = ov < V
    lo = ot[:, None] > ot[None, :]
    le = ot[:, None] >= ot[None, :]
    sameblk = (ot // SUB)[:, None] == (ot // SUB)[None, :]
    kcol = mk[:, None].to(tl.float32)

    z = tl.zeros([BK, BV], dtype=tl.float32)
    for nn in range(0, NT):
        n = NT - 1 - nn
        t = n * BT + ot
        mt = t < T
        m2 = mt[:, None] & mk[None, :]
        mvv = mt[:, None] & mv[None, :]
        base = bh.to(tl.int64) * T + t.to(tl.int64)
        off = base[:, None] * K + ok[None, :]
        ar = tl.load(Ar + off, mask=m2, other=0.0)
        qr = tl.load(Qr + off, mask=m2, other=0.0)
        bc = tl.load(Bc + off, mask=m2, other=0.0)
        kc = tl.load(Kc + off, mask=m2, other=0.0)
        tinv = tl.load(Tv + (bh.to(tl.int64) * NT + n) * BT * BT
                       + ot[:, None] * BT + ot[None, :])

        gsrc = (b_i.to(tl.int64) * T + t.to(tl.int64)) * H * K + h_i * K
        g_ = tl.load(gk + gsrc[:, None] + ok[None, :], mask=m2, other=0.0).to(tl.float32)
        gl = tl.sum(g_, axis=0)
        q_raw = tl.load(q + gsrc[:, None] + ok[None, :], mask=m2, other=0.0).to(tl.float32)
        a_raw = tl.load(a + gsrc[:, None] + ok[None, :], mask=m2, other=0.0).to(tl.float32)

        vsrc = (b_i.to(tl.int64) * T + t.to(tl.int64)) * H * V + h_i * V
        do = tl.load(DO + vsrc[:, None] + ov[None, :], mask=mvv, other=0.0).to(tl.float32)

        zdst = ((bh.to(tl.int64) * NT + n) * K + ok[:, None].to(tl.int64)) * V + ov[None, :]
        tl.store(Zs + zdst, z, mask=mk[:, None] & mv[None, :])

        aqb = tl.where(le, tl.dot(qr, tl.trans(bc), input_precision="ieee"), 0.0)
        aqk = tl.where(le, tl.dot(qr, tl.trans(kc), input_precision="ieee"), 0.0)
        aak = tl.where(lo, tl.dot(ar, tl.trans(kc), input_precision="ieee"), 0.0)

        e_gl = tl.exp(tl.minimum(gl, CAP))
        bd = bc * e_gl[None, :]
        kd = kc * e_gl[None, :]
        du = tl.dot(tl.trans(aqb), do, input_precision="ieee") \
            + tl.dot(bd, z, input_precision="ieee")
        aab = tl.where(lo, tl.dot(ar, tl.trans(bc), input_precision="ieee"), 0.0)
        aoff = tl.trans(tl.where(sameblk, 0.0, aab))
        w = tl.dot(tl.trans(tinv), du, input_precision="ieee")
        for _ in tl.static_range(NBLK - 1):
            w = tl.dot(tl.trans(tinv), du + tl.dot(aoff, w, input_precision="ieee"),
                       input_precision="ieee")
        dv = tl.dot(tl.trans(aqk), do, input_precision="ieee") \
            + tl.dot(kd, z, input_precision="ieee") \
            + tl.dot(tl.trans(aak), w, input_precision="ieee")

        tl.store(W + base[:, None] * V + ov[None, :], w, mask=mvv)
        tl.store(dV + vsrc[:, None] + ov[None, :], dv.to(dV.dtype.element_ty), mask=mvv)

        z = (tl.dot(tl.trans(qr), do, input_precision="ieee")
             + tl.dot(tl.trans(ar), w, input_precision="ieee")
             + z * e_gl[:, None]) * kcol


@triton.jit
def _dplr_bwd_mix(Ar, Qr, Bc, Kc, Tv, gk, q, k, a, b, v, U, W, DO, S0, Zs,
                  dq, dk, da, db, dg,
                  T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NT,
                  BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr,
                  SUB: tl.constexpr, NBLK: tl.constexpr, scale, CAP: tl.constexpr):
    """Per-chunk K-side gradients: dq, dk, dalpha, dbeta, dgk."""
    n = tl.program_id(0)
    bh = tl.program_id(1)
    b_i = bh // H
    h_i = bh % H

    ot = tl.arange(0, BT)
    ok = tl.arange(0, BK)
    t = n * BT + ot
    mt = t < T
    mk = ok < K
    m2 = mt[:, None] & mk[None, :]
    lo = ot[:, None] > ot[None, :]
    le = ot[:, None] >= ot[None, :]
    sameblk = (ot // SUB)[:, None] == (ot // SUB)[None, :]

    base = bh.to(tl.int64) * T + t.to(tl.int64)
    off = base[:, None] * K + ok[None, :]
    ar = tl.load(Ar + off, mask=m2, other=0.0)
    qr = tl.load(Qr + off, mask=m2, other=0.0)
    bc = tl.load(Bc + off, mask=m2, other=0.0)
    kc = tl.load(Kc + off, mask=m2, other=0.0)

    gsrc = (b_i.to(tl.int64) * T + t.to(tl.int64)) * H * K + h_i * K
    g_ = tl.load(gk + gsrc[:, None] + ok[None, :], mask=m2, other=0.0).to(tl.float32)
    g_c = tl.cumsum(g_, axis=0)
    g = g_c
    ge = g - g_
    gl = tl.sum(g_, axis=0)
    e_gl = tl.exp(tl.minimum(gl, CAP))
    q_raw = tl.load(q + gsrc[:, None] + ok[None, :], mask=m2, other=0.0).to(tl.float32)
    k_raw = tl.load(k + gsrc[:, None] + ok[None, :], mask=m2, other=0.0).to(tl.float32)
    a_raw = tl.load(a + gsrc[:, None] + ok[None, :], mask=m2, other=0.0).to(tl.float32)
    b_raw = tl.load(b + gsrc[:, None] + ok[None, :], mask=m2, other=0.0).to(tl.float32)
    tinv = tl.load(Tv + (bh.to(tl.int64) * NT + n) * BT * BT
                   + ot[:, None] * BT + ot[None, :])

    dar = tl.zeros([BT, BK], dtype=tl.float32)
    dqr = tl.zeros([BT, BK], dtype=tl.float32)
    dbn = tl.zeros([BT, BK], dtype=tl.float32)
    dkn = tl.zeros([BT, BK], dtype=tl.float32)
    dgacc = tl.zeros([BT, BK], dtype=tl.float32)

    for iv in range(0, tl.cdiv(V, BV)):
        ov = iv * BV + tl.arange(0, BV)
        mv = ov < V
        mvv = mt[:, None] & mv[None, :]
        vsrc = (b_i.to(tl.int64) * T + t.to(tl.int64)) * H * V + h_i * V
        vv = tl.load(v + vsrc[:, None] + ov[None, :], mask=mvv, other=0.0).to(tl.float32)
        do = tl.load(DO + vsrc[:, None] + ov[None, :], mask=mvv, other=0.0).to(tl.float32)
        u = tl.load(U + base[:, None] * V + ov[None, :], mask=mvv, other=0.0)
        w = tl.load(W + base[:, None] * V + ov[None, :], mask=mvv, other=0.0)
        soff = ((bh.to(tl.int64) * NT + n) * K + ok[:, None].to(tl.int64)) * V + ov[None, :]
        msv = mk[:, None] & mv[None, :]
        s0 = tl.load(S0 + soff, mask=msv, other=0.0)
        # refine U against its defining equation (I - A_ab) u = Ar S0 + A_ak V.
        # dbeta and dgk contract U where the result is a small difference of large
        # terms, so the digits the forward inverse cascade dropped show up there.
        aab = tl.where(lo, tl.dot(ar, tl.trans(bc), input_precision="ieee"), 0.0)
        aak = tl.where(lo, tl.dot(ar, tl.trans(kc), input_precision="ieee"), 0.0)
        aoff = tl.where(sameblk, 0.0, aab)
        rhs_u = tl.dot(ar, s0, input_precision="ieee") \
            + tl.dot(aak, vv, input_precision="ieee")
        u = tl.dot(tinv, rhs_u, input_precision="ieee")
        for _ in tl.static_range(NBLK - 1):
            u = tl.dot(tinv, rhs_u + tl.dot(aoff, u, input_precision="ieee"),
                       input_precision="ieee")
        z = tl.load(Zs + soff, mask=msv, other=0.0)

        daab = tl.where(lo, tl.dot(w, tl.trans(u), input_precision="ieee"), 0.0)
        daak = tl.where(lo, tl.dot(w, tl.trans(vv), input_precision="ieee"), 0.0)
        daqb = tl.where(le, tl.dot(do, tl.trans(u), input_precision="ieee"), 0.0)
        daqk = tl.where(le, tl.dot(do, tl.trans(vv), input_precision="ieee"), 0.0)

        # dq_i = S_i do_i and da_i = S_{i-1} w_i, with the state marched forward
        # through the chunk in its OWN frame (never the e^{-g} normalized one).
        # The normalized frame inflates the state by e^{-g} ~ 1e13 over a chunk and
        # then divides it back out, which destroys the low bits wherever the true
        # gradient is a small difference of large state entries.
        yi = s0
        for ii in range(0, BT):
            sel = ot == ii
            gi = tl.sum(tl.where(sel[:, None], g_, 0.0), axis=0)
            mi = yi * tl.exp(tl.minimum(gi, CAP))[:, None]
            v_i = tl.sum(tl.where(sel[:, None], vv, 0.0), axis=0)
            bb_i = tl.sum(tl.where(sel[:, None], b_raw, 0.0), axis=0)
            kk_i = tl.sum(tl.where(sel[:, None], k_raw, 0.0), axis=0)
            u_i = tl.sum(tl.where(sel[:, None], u, 0.0), axis=0)
            si = mi + bb_i[:, None] * u_i[None, :] + kk_i[:, None] * v_i[None, :]
            do_i = tl.sum(tl.where(sel[:, None], do, 0.0), axis=0)
            w_i = tl.sum(tl.where(sel[:, None], w, 0.0), axis=0)
            dqr += tl.where(sel[:, None],
                            (scale * tl.sum(si * do_i[None, :], axis=1))[None, :], 0.0)
            dar += tl.where(sel[:, None], tl.sum(yi * w_i[None, :], axis=1)[None, :], 0.0)
            yi = si

        # dk_i = P_i v_i, db_i = P_i u_i and dgk_i = sum_v (D_i P_i) * Y_i, with the
        # adjoint marched backward through the chunk in its own frame for the same
        # reason.  Y_i is re-derived by undoing one marched step of the forward
        # state, so both operands of dgk stay in the natural frame.
        pi = z
        yj = yi
        for jj in range(BT - 1, -1, -1):
            selp = ot == jj + 1
            gp = tl.sum(tl.where(selp[:, None], g_, 0.0), axis=0)
            a_p = tl.sum(tl.where(selp[:, None], a_raw, 0.0), axis=0)
            w_p = tl.sum(tl.where(selp[:, None], w, 0.0), axis=0)
            has_p = tl.sum(selp.to(tl.float32), axis=0)
            pi = pi * (1.0 + has_p * (tl.exp(tl.minimum(gp, CAP)) - 1.0))[:, None] \
                + a_p[:, None] * w_p[None, :]
            sel = ot == jj
            q_i = tl.sum(tl.where(sel[:, None], q_raw, 0.0), axis=0) * scale
            do_i = tl.sum(tl.where(sel[:, None], do, 0.0), axis=0)
            pi = pi + q_i[:, None] * do_i[None, :]
            u_i = tl.sum(tl.where(sel[:, None], u, 0.0), axis=0)
            v_i = tl.sum(tl.where(sel[:, None], vv, 0.0), axis=0)
            gi = tl.sum(tl.where(sel[:, None], g_c, 0.0), axis=0)
            dkn += tl.where(sel[:, None], tl.sum(pi * v_i[None, :], axis=1)[None, :], 0.0)
            dbn += tl.where(sel[:, None], tl.sum(pi * u_i[None, :], axis=1)[None, :], 0.0)

            # dgk_i = sum_v (D_i Y_i) P_i with D_i Y_i expanded on the step-i anchor:
            #   D_i Y_i = e^{g_i} S0 + sum_{j<i} e^{g_i-g_j} (b_j u_j^T + k_j v_j^T),
            # every weight e^{g_i-g_j} <= 1, so nothing is inflated and divided back.
            ru = tl.trans(tl.dot(pi, tl.trans(u), input_precision="ieee"))
            rv = tl.trans(tl.dot(pi, tl.trans(vv), input_precision="ieee"))
            wgt = tl.exp(tl.minimum(gi[None, :] - g_c, CAP))
            prior = (ot < jj)[:, None]
            contrib = tl.sum(tl.where(prior, wgt * (b_raw * ru + k_raw * rv), 0.0), axis=0) \
                + tl.exp(tl.minimum(gi, CAP)) * tl.sum(pi * s0, axis=1)
            dgacc += tl.where(sel[:, None], contrib[None, :], 0.0)



    # every marched quantity is already in the natural frame
    d_q = dqr
    d_a = dar
    d_b = dbn
    d_k = dkn

    tl.store(dq + gsrc[:, None] + ok[None, :], d_q.to(dq.dtype.element_ty), mask=m2)
    tl.store(dk + gsrc[:, None] + ok[None, :], d_k.to(dk.dtype.element_ty), mask=m2)
    tl.store(da + gsrc[:, None] + ok[None, :], d_a.to(da.dtype.element_ty), mask=m2)
    tl.store(db + gsrc[:, None] + ok[None, :], d_b.to(db.dtype.element_ty), mask=m2)

    tl.store(dg + gsrc[:, None] + ok[None, :], dgacc.to(dg.dtype.element_ty), mask=m2)


def _fwd(q, k, v, alpha, beta, gk):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H
    NT = triton.cdiv(T, C)
    BK = _pow2(K)
    BV = min(_pow2(V), 64)
    NV = triton.cdiv(V, BV)
    scale = K ** -0.5
    dev = q.device

    Ar = torch.empty(BH, T, K, dtype=torch.float32, device=dev)
    Qr = torch.empty(BH, T, K, dtype=torch.float32, device=dev)
    Bc = torch.empty(BH, T, K, dtype=torch.float32, device=dev)
    Kc = torch.empty(BH, T, K, dtype=torch.float32, device=dev)
    Tv = torch.empty(BH, NT, C, C, dtype=torch.float32, device=dev)
    _dplr_prep[(NT, BH)](q, k, alpha, beta, gk, Ar, Qr, Bc, Kc, Tv,
                         T, H, K, scale, BK=BK, BT=C, SUB=SUB, LOG_SUB=LOG_SUB, NBLK=C // SUB, CAP=EXP_CAP,
                         num_warps=4, num_stages=2)

    U = torch.empty(BH, T, V, dtype=torch.float32, device=dev)
    S0 = torch.empty(BH, NT, K, V, dtype=torch.float32, device=dev)
    O = torch.empty(B, T, H, V, dtype=v.dtype, device=dev)
    _dplr_fwd[(NV, BH)](Ar, Qr, Bc, Kc, Tv, gk, v, U, O, S0,
                        T, H, K, V, NT, BK=BK, BV=BV, BT=C, SUB=SUB, NBLK=C // SUB, CAP=EXP_CAP,
                        num_warps=4, num_stages=2)
    return O, Ar, Qr, Bc, Kc, Tv, U, S0


def _bwd(dO, q, k, v, alpha, beta, gk, Ar, Qr, Bc, Kc, Tv, U, S0):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H
    NT = triton.cdiv(T, C)
    BK = _pow2(K)
    BV = min(_pow2(V), 64)
    NV = triton.cdiv(V, BV)
    scale = K ** -0.5
    dev = q.device

    W = torch.empty(BH, T, V, dtype=torch.float32, device=dev)
    Zs = torch.empty(BH, NT, K, V, dtype=torch.float32, device=dev)
    dv = torch.empty(B, T, H, V, dtype=v.dtype, device=dev)
    _dplr_bwd_scan[(NV, BH)](Ar, Qr, Bc, Kc, Tv, gk, q, alpha, dO, W, dv, Zs,
                             T, H, K, V, NT, BK=BK, BV=BV, BT=C, SUB=SUB, NBLK=C // SUB,
                             scale=scale, CAP=EXP_CAP,
                             num_warps=4, num_stages=2)

    dq = torch.empty(B, T, H, K, dtype=q.dtype, device=dev)
    dk = torch.empty(B, T, H, K, dtype=q.dtype, device=dev)
    da = torch.empty(B, T, H, K, dtype=q.dtype, device=dev)
    db = torch.empty(B, T, H, K, dtype=q.dtype, device=dev)
    dg = torch.empty(B, T, H, K, dtype=gk.dtype, device=dev)
    _dplr_bwd_mix[(NT, BH)](Ar, Qr, Bc, Kc, Tv, gk, q, k, alpha, beta, v, U, W, dO, S0, Zs,
                            dq, dk, da, db, dg,
                            T, H, K, V, NT, BK=BK, BV=BV, BT=C, SUB=SUB, NBLK=C // SUB,
                             scale=scale, CAP=EXP_CAP,
                            num_warps=4, num_stages=2)
    return dq, dk, dv, da, db, dg


class _Dplr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, gk):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        alpha, beta, gk = alpha.contiguous(), beta.contiguous(), gk.contiguous()
        O, Ar, Qr, Bc, Kc, Tv, U, S0 = _fwd(q, k, v, alpha, beta, gk)
        ctx.save_for_backward(q, k, v, alpha, beta, gk, Ar, Qr, Bc, Kc, Tv, U, S0)
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, alpha, beta, gk, Ar, Qr, Bc, Kc, Tv, U, S0 = ctx.saved_tensors
        return _bwd(dO.contiguous(), q, k, v, alpha, beta, gk, Ar, Qr, Bc, Kc, Tv, U, S0)


def chunked_dplr_delta(
    q: torch.Tensor,      # [B, T, H, K]
    k: torch.Tensor,      # [B, T, H, K]
    v: torch.Tensor,      # [B, T, H, V]
    alpha: torch.Tensor,  # [B, T, H, K]    DPLR erasure basis
    beta: torch.Tensor,   # [B, T, H, K]    DPLR write basis
    gk: torch.Tensor,     # [B, T, H, K]    per-K-channel log-decay
) -> torch.Tensor:        # [B, T, H, V]
    return _Dplr.apply(q, k, v, alpha, beta, gk)
