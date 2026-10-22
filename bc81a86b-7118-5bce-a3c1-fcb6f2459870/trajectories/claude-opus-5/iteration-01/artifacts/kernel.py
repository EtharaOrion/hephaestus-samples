"""Chunk-parallel gated delta rule, forward and backward, in Triton.

Every tensor is contiguous in [B, T, H, D] layout, so element (b, t, h, d) sits
at (((b * T + t) * H + h) * D + d).

For a chunk of BT timesteps, with `gam` the inclusive cumulative sum of the log
decay inside the chunk and s = K ** -0.5:

    A_ij  = beta_i * exp(gam_i - gam_j) * (k_i . k_j)          for i > j
    X     = (I + A)^-1
    W     = X @ (beta * exp(gam) * K)
    U     = X @ (beta * V)
    D     = U - W @ S_c
    S_c+1 = exp(gam_last) * S_c + Kd^T @ D,   Kd_j = exp(gam_last - gam_j) k_j
    O     = (s * exp(gam) * Q) @ S_c + tril(P) @ D,
            P_ij = s * exp(gam_i - gam_j) * (q_i . k_j)        for i >= j

The recurrent state S and its adjoint live in float32 buffers throughout.
"""

import torch
import triton
import triton.language as tl

BT = 64
BVB_TILE = 32


def _warps(KP, wide):
    # 8 warps fault on a shared-memory read when the head tile is under 64 wide
    return 8 if (wide and KP >= 64) else 4


def _pow2(n):
    return 1 << (n - 1).bit_length()


@triton.jit
def _cumsum_kernel(g, gc, T, H: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p = (i_b * T).to(tl.int64) * H + i_h + o_t * H
    b_g = tl.load(g + p, mask=m_t, other=0.0).to(tl.float32)
    tl.store(gc + p, tl.cumsum(b_g, axis=0), mask=m_t)


@triton.jit
def _tri_inv(b_A, BT: tl.constexpr, BS: tl.constexpr, PREC: tl.constexpr):
    """Inverse of the unit lower triangular matrix I + b_A, b_A strictly lower.

    The BS-wide diagonal blocks are inverted by forward substitution, every
    block at once, and the block-strictly-lower remainder E of I - X (I + A) is
    removed by a Neumann sum that terminates exactly, since E is nilpotent of
    index BT // BS.
    """
    o = tl.arange(0, BT)
    blk = o // BS
    m_diag = blk[:, None] == blk[None, :]
    m_I = (o[:, None] == o[None, :]).to(tl.float32)
    b_N = tl.where(m_diag, -b_A, 0.0)
    b_X = b_N
    for i in range(2, BS):
        m_row = (o % BS) == i
        b_a = tl.sum(tl.where(m_row[:, None], b_N, 0.0), 0)
        b_a += tl.sum(b_a[:, None] * b_X, 0)
        b_X = tl.where(m_row[:, None] & m_diag, b_a[None, :], b_X)
    b_X += m_I

    b_E = -tl.dot(b_X, b_A, input_precision=PREC)
    b_E = tl.where(m_diag, 0.0, b_E)
    # sum_{j<BT//BS} E^j, factored as prod_i (I + E^(2^i)); E^(BT//BS) is zero
    b_F = b_E + m_I
    n = BS
    while n * 2 < BT:
        b_E = tl.dot(b_E, b_E, input_precision=PREC)
        b_F = tl.dot(b_F, b_E + m_I, input_precision=PREC)
        n = n * 2
    return tl.dot(b_F, b_X, input_precision=PREC)


@triton.jit
def _wy_kernel(
    k, v, beta, gc, Ai, w, u,
    T, NT, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    KP: tl.constexpr, VP: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = tl.arange(0, KP)
    m_k = o_k < K
    o_v = tl.arange(0, VP)
    m_v = o_v < V

    base = (i_b * T).to(tl.int64)
    b_g = tl.load(gc + base * HV + i_hv + o_t * HV, mask=m_t, other=0.0)
    b_beta = tl.load(beta + base * HV + i_hv + o_t * HV, mask=m_t, other=0.0).to(tl.float32)

    p_k = k + (base * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)

    b_A = tl.dot(b_k, tl.trans(b_k), input_precision=PREC)
    b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :]) * b_beta[:, None]
    b_A = tl.where((o_t[:, None] > o_t[None, :]) & m_t[:, None] & m_t[None, :], b_A, 0.0)

    b_X = _tri_inv(b_A, BT, BS, PREC)

    o_i = tl.arange(0, BT)
    p_Ai = Ai + ((i_b * NT + i_t) * HV + i_hv).to(tl.int64) * (BT * BT) \
        + o_i[:, None] * BT + o_i[None, :]
    tl.store(p_Ai, b_X)

    b_Xc = b_X.to(b_k.dtype)
    b_kb = (b_k * (b_beta * tl.exp(b_g))[:, None]).to(b_k.dtype)
    b_w = tl.dot(b_Xc, b_kb, input_precision=PREC)
    p_w = w + (base * HV + i_hv) * K + o_t[:, None] * (HV * K) + o_k[None, :]
    tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_t[:, None] & m_k[None, :])

    p_v = v + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
    b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0)
    b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
    b_u = tl.dot(b_Xc, b_vb, input_precision=PREC)
    p_u = u + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
    tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


@triton.jit
def _state_kernel(
    k, w, u, gc, h, d,
    T, NT, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    KP: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, KP)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    b_h = tl.zeros([KP, BV], dtype=tl.float32)

    base = (i_b * T).to(tl.int64)
    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_h = h + ((i_b * NT + i_t) * HV + i_hv).to(tl.int64) * (K * V) \
            + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h, b_h, mask=m_k[:, None] & m_v[None, :])

        p_w = w + (base * HV + i_hv) * K + o_t[:, None] * (HV * K) + o_k[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k[None, :], other=0.0)
        p_u = u + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_u = tl.load(p_u, mask=m_t[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_d = b_u - tl.dot(b_w, b_h.to(b_w.dtype), input_precision=PREC)
        p_d = d + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_d, b_d.to(p_d.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        b_g = tl.load(gc + base * HV + i_hv + o_t * HV, mask=m_t, other=0.0)
        last = tl.minimum((i_t + 1) * BT, T) - 1
        b_gl = tl.load(gc + base * HV + i_hv + last * HV)

        p_k = k + (base * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_kd = (b_k * tl.where(m_t, tl.exp(b_gl - b_g), 0.0)[:, None]).to(b_k.dtype)
        b_h = b_h * tl.exp(b_gl) + tl.dot(tl.trans(b_kd), b_d.to(b_kd.dtype), input_precision=PREC)


@triton.jit
def _out_kernel(
    q, k, gc, h, d, o,
    scale, T, NT, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    KP: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = tl.arange(0, KP)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    base = (i_b * T).to(tl.int64)
    b_g = tl.load(gc + base * HV + i_hv + o_t * HV, mask=m_t, other=0.0)

    p_q = q + (base * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
    p_k = k + (base * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)

    b_qg = (b_q * (scale * tl.exp(b_g))[:, None]).to(b_q.dtype)
    b_p = tl.dot(b_q, tl.trans(b_k), input_precision=PREC) * scale * tl.exp(b_g[:, None] - b_g[None, :])
    b_p = tl.where((o_t[:, None] >= o_t[None, :]) & m_t[:, None] & m_t[None, :], b_p, 0.0)

    p_h = h + ((i_b * NT + i_t) * HV + i_hv).to(tl.int64) * (K * V) \
        + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
    p_d = d + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
    b_d = tl.load(p_d, mask=m_t[:, None] & m_v[None, :], other=0.0)

    b_o = tl.dot(b_qg, b_h.to(b_qg.dtype), input_precision=PREC) + tl.dot(b_p.to(b_d.dtype), b_d, input_precision=PREC)
    p_o = o + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


@triton.jit
def _dstate_kernel(
    q, k, w, gc, do, dh, dd,
    scale, T, NT, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    KP: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, KP)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    b_dh = tl.zeros([KP, BV], dtype=tl.float32)

    base = (i_b * T).to(tl.int64)
    for i_s in range(NT):
        i_t = NT - 1 - i_s
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        b_g = tl.load(gc + base * HV + i_hv + o_t * HV, mask=m_t, other=0.0)
        last = tl.minimum((i_t + 1) * BT, T) - 1
        b_gl = tl.load(gc + base * HV + i_hv + last * HV)

        p_q = q + (base * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        p_k = k + (base * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0.0)
        p_do = do + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_t[:, None] & m_v[None, :], other=0.0)

        b_p = tl.dot(b_q, tl.trans(b_k), input_precision=PREC) * scale * tl.exp(b_g[:, None] - b_g[None, :])
        b_p = tl.where((o_t[:, None] >= o_t[None, :]) & m_t[:, None] & m_t[None, :], b_p, 0.0)
        b_kd = (b_k * tl.where(m_t, tl.exp(b_gl - b_g), 0.0)[:, None]).to(b_k.dtype)

        b_dd = tl.dot(tl.trans(b_p).to(b_do.dtype), b_do, input_precision=PREC) + tl.dot(b_kd, b_dh.to(b_kd.dtype), input_precision=PREC)
        p_dd = dd + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_dd, b_dd, mask=m_t[:, None] & m_v[None, :])

        p_w = w + (base * HV + i_hv) * K + o_t[:, None] * (HV * K) + o_k[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_qg = (b_q * tl.where(m_t, scale * tl.exp(b_g), 0.0)[:, None]).to(b_q.dtype)
        b_dh = b_dh * tl.exp(b_gl) + tl.dot(tl.trans(b_qg), b_do, input_precision=PREC) \
            - tl.dot(tl.trans(b_w), b_dd.to(b_w.dtype), input_precision=PREC)
        p_dhs = dh + ((i_b * NT + i_t) * HV + i_hv).to(tl.int64) * (K * V) \
            + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dhs, b_dh, mask=m_k[:, None] & m_v[None, :])


@triton.jit
def _bwd_kernel(
    q, k, v, beta, gc, Ai, w, h, dh, dd, d, do,
    dq, dk, dv, dbeta, dg,
    scale, T, NT, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    KP: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, PREC: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = tl.arange(0, KP)
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]
    m_tt = m_t[:, None] & m_t[None, :]
    m_lo = (o_t[:, None] > o_t[None, :]) & m_tt
    m_le = (o_t[:, None] >= o_t[None, :]) & m_tt

    base = (i_b * T).to(tl.int64)
    b_g = tl.load(gc + base * HV + i_hv + o_t * HV, mask=m_t, other=0.0)
    last = tl.minimum((i_t + 1) * BT, T) - 1
    b_gl = tl.load(gc + base * HV + i_hv + last * HV)
    b_beta = tl.load(beta + base * HV + i_hv + o_t * HV, mask=m_t, other=0.0).to(tl.float32)

    p_q = q + (base * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_tk, other=0.0)
    p_k = k + (base * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_k = tl.load(p_k, mask=m_tk, other=0.0)
    o_i = tl.arange(0, BT)
    p_Ai = Ai + ((i_b * NT + i_t) * HV + i_hv).to(tl.int64) * (BT * BT) \
        + o_i[:, None] * BT + o_i[None, :]
    b_X = tl.load(p_Ai)

    dt = b_k.dtype
    b_Xt = tl.trans(b_X).to(dt)
    b_exp = tl.exp(b_g)
    b_edl = tl.where(m_t, tl.exp(b_gl - b_g), 0.0)
    b_dg = tl.zeros([BT], dtype=tl.float32)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)

    b_dw = tl.zeros([BT, KP], dtype=tl.float32)
    b_dXv = tl.zeros([BT, BT], dtype=tl.float32)
    b_dqg = tl.zeros([BT, KP], dtype=tl.float32)
    b_dP = tl.zeros([BT, BT], dtype=tl.float32)
    b_dkd = tl.zeros([BT, KP], dtype=tl.float32)
    b_hlast = tl.zeros([], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        m_kv = m_k[:, None] & m_v[None, :]

        p_v = v + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        p_d = d + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_d = tl.load(p_d, mask=m_tv, other=0.0)
        p_do = do + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_tv, other=0.0)
        p_dd = dd + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_ddc = tl.load(p_dd, mask=m_tv, other=0.0).to(dt)
        p_h = h + ((i_b * NT + i_t) * HV + i_hv).to(tl.int64) * (K * V) \
            + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h, mask=m_kv, other=0.0)
        p_dhn = dh + ((i_b * NT + i_t + 1) * HV + i_hv).to(tl.int64) * (K * V) \
            + o_k[:, None] * V + o_v[None, :]
        b_dhn = tl.load(p_dhn, mask=m_kv & (i_t + 1 < NT), other=0.0)

        b_vb = (b_v * b_beta[:, None]).to(dt)
        b_dw -= tl.dot(b_ddc, tl.trans(b_h).to(dt), input_precision=PREC)
        b_dXv += tl.dot(b_ddc, tl.trans(b_vb), input_precision=PREC)
        b_dqg += tl.dot(b_do, tl.trans(b_h).to(dt), input_precision=PREC)
        b_dP += tl.dot(b_do, tl.trans(b_d), input_precision=PREC)
        b_dkd += tl.dot(b_d, tl.trans(b_dhn).to(dt), input_precision=PREC)
        b_hlast += tl.sum(b_dhn * b_h)

        # U = X (beta V)
        b_XtdD = tl.dot(b_Xt, b_ddc, input_precision=PREC)
        b_dbeta += tl.sum(b_XtdD * b_v.to(tl.float32), 1)
        p_dv = dv + (base * HV + i_hv) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_dv, (b_XtdD * b_beta[:, None]).to(p_dv.dtype.element_ty), mask=m_tv)

    # D = U - W S_c
    b_kb = (b_k * (b_beta * b_exp)[:, None]).to(dt)
    b_dX = b_dXv + tl.dot(b_dw.to(dt), tl.trans(b_kb), input_precision=PREC)

    # X = (I + A)^-1  =>  dA = -X^T dX X^T, strictly lower part only
    b_dA = -tl.dot(b_Xt, tl.dot(b_dX.to(dt), b_Xt, input_precision=PREC).to(dt),
                   input_precision=PREC)
    b_dA = tl.where(m_lo, b_dA, 0.0)

    b_ed = tl.exp(b_g[:, None] - b_g[None, :])
    b_kk = tl.dot(b_k, tl.trans(b_k), input_precision=PREC)
    b_A = tl.where(m_lo, b_kk * b_ed * b_beta[:, None], 0.0)
    b_dbeta += tl.sum(tl.where(m_lo, b_dA * b_kk * b_ed, 0.0), 1)
    b_c = b_dA * b_A
    b_dg += tl.sum(b_c, 1) - tl.sum(b_c, 0)
    b_wA = tl.where(m_lo, b_dA * b_ed * b_beta[:, None], 0.0).to(dt)
    b_dk = tl.dot(b_wA, b_k, input_precision=PREC) \
        + tl.dot(tl.trans(b_wA), b_k, input_precision=PREC)

    # W = X (beta exp(gam) K)
    b_dkb = tl.dot(b_Xt, b_dw.to(dt), input_precision=PREC)
    b_kdot = tl.sum(b_dkb * b_k.to(tl.float32), 1)
    b_dk += b_dkb * (b_beta * b_exp)[:, None]
    b_dbeta += b_kdot * b_exp
    b_dg += b_kdot * b_beta * b_exp

    # O = (s exp(gam) q) S_c + tril(P) D
    b_dq = b_dqg * (scale * b_exp)[:, None]
    b_dg += tl.sum(b_dqg * b_q.to(tl.float32) * (scale * b_exp)[:, None], 1)

    b_dP = tl.where(m_le, b_dP, 0.0)
    b_P = tl.where(m_le, tl.dot(b_q, tl.trans(b_k), input_precision=PREC) * scale * b_ed, 0.0)
    b_cP = b_dP * b_P
    b_dg += tl.sum(b_cP, 1) - tl.sum(b_cP, 0)
    b_wP = tl.where(m_le, b_dP * scale * b_ed, 0.0).to(dt)
    b_dq += tl.dot(b_wP, b_k, input_precision=PREC)
    b_dk += tl.dot(tl.trans(b_wP), b_q, input_precision=PREC)

    # S_c+1 = exp(gam_last) S_c + Kd^T D
    b_dk += b_dkd * b_edl[:, None]
    b_row = tl.sum(b_dkd * b_k.to(tl.float32), 1) * b_edl
    b_dg -= b_row
    b_glast = tl.sum(b_row, 0) + b_hlast * tl.exp(b_gl)
    b_dg += tl.where(o_t == last, b_glast, 0.0)

    # gam is the inclusive cumsum of g over the chunk, so dg_t sums forward
    b_dg = tl.where(m_t, b_dg, 0.0)
    b_dgr = tl.sum(tl.where(o_t[:, None] <= o_t[None, :], b_dg[None, :], 0.0), 1)

    p_dq = dq + (base * HV + i_hv) * K + o_t[:, None] * (HV * K) + o_k[None, :]
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_tk)
    p_dk = dk + (base * HV + i_hv) * K + o_t[:, None] * (HV * K) + o_k[None, :]
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)
    tl.store(dbeta + base * HV + i_hv + o_t * HV,
             b_dbeta.to(dbeta.dtype.element_ty), mask=m_t)
    tl.store(dg + base * HV + i_hv + o_t * HV, b_dgr, mask=m_t)



@triton.jit
def _group_reduce_kernel(
    src, out, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    KP: tl.constexpr, R: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1)
    o_k = tl.arange(0, KP)
    m_k = o_k < K
    acc = tl.zeros([KP], dtype=tl.float32)
    for r in range(R):
        acc += tl.load(src + (i_n * HV + i_h * R + r) * K + o_k, mask=m_k, other=0.0)
    tl.store(out + (i_n * H + i_h) * K + o_k, acc.to(out.dtype.element_ty), mask=m_k)


def _fwd(q, k, v, g, beta):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    NT = triton.cdiv(T, BT)
    KP, VP = _pow2(K), _pow2(V)
    BS = 16 if BT > 16 else BT
    PREC = "ieee"
    NS = 1 if q.element_size() > 2 else 2

    gc = torch.empty(B, T, HV, dtype=torch.float32, device=q.device)
    _cumsum_kernel[(NT, B * HV)](g, gc, T, H=HV, BT=BT)

    Ai = torch.empty(B * NT * HV, BT, BT, dtype=torch.float32, device=q.device)
    w = q.new_empty(B, T, HV, K)
    u = q.new_empty(B, T, HV, V)
    _wy_kernel[(NT, B * HV)](
        k, v, beta, gc, Ai, w, u, T, NT,
        H=H, HV=HV, K=K, V=V, KP=KP, VP=VP, BT=BT, BS=BS, PREC=PREC,
        num_warps=_warps(KP, True), num_stages=NS,
    )

    BV = min(64 if KP <= 128 else 32, _pow2(V))
    NV = triton.cdiv(V, BV)
    h = torch.empty(B * NT * HV, K, V, dtype=torch.float32, device=q.device)
    d = q.new_empty(B, T, HV, V)
    _state_kernel[(NV, B * HV)](
        k, w, u, gc, h, d, T, NT,
        H=H, HV=HV, K=K, V=V, KP=KP, BT=BT, BV=BV, PREC=PREC,
        num_warps=_warps(KP, False), num_stages=NS,
    )

    o = q.new_empty(B, T, HV, V)
    _out_kernel[(NT, NV, B * HV)](
        q, k, gc, h, d, o, K ** -0.5, T, NT,
        H=H, HV=HV, K=K, V=V, KP=KP, BT=BT, BV=BV, PREC=PREC,
        num_warps=_warps(KP, False), num_stages=NS,
    )
    return o, gc, Ai, w, h, d


def _bwd(q, k, v, g, beta, gc, Ai, w, h, d, do):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    NT = triton.cdiv(T, BT)
    KP = _pow2(K)
    scale = K ** -0.5
    PREC = "ieee"
    NS = 1 if q.element_size() > 2 else 2

    BV = min(64 if KP <= 128 else 32, _pow2(V))
    BVB = min(BVB_TILE, _pow2(V))
    NV = triton.cdiv(V, BV)
    dh = torch.empty(B * NT * HV, K, V, dtype=torch.float32, device=q.device)
    dd = torch.empty(B, T, HV, V, dtype=torch.float32, device=q.device)
    _dstate_kernel[(NV, B * HV)](
        q, k, w, gc, do, dh, dd, scale, T, NT,
        H=H, HV=HV, K=K, V=V, KP=KP, BT=BT, BV=BV, PREC=PREC,
        num_warps=_warps(KP, False), num_stages=NS,
    )

    R = HV // H
    if R == 1:
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dqg, dkg = dq, dk
    else:
        dqg = torch.empty(B, T, HV, K, dtype=torch.float32, device=q.device)
        dkg = torch.empty(B, T, HV, K, dtype=torch.float32, device=q.device)
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta)
    dg = torch.empty(B, T, HV, dtype=torch.float32, device=q.device)
    _bwd_kernel[(NT, B * HV)](
        q, k, v, beta, gc, Ai, w, h, dh, dd, d, do,
        dqg, dkg, dv, dbeta, dg, scale, T, NT,
        H=H, HV=HV, K=K, V=V, KP=KP, BT=BT, BV=BVB, PREC=PREC,
        num_warps=_warps(KP, True), num_stages=NS,
    )

    if R > 1:
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        _group_reduce_kernel[(B * T, H)](dqg, dq, H=H, HV=HV, K=K, KP=KP, R=R)
        _group_reduce_kernel[(B * T, H)](dkg, dk, H=H, HV=HV, K=K, KP=KP, R=R)
    return dq, dk, dv, dg, dbeta


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        ctx.g_dtype = g.dtype
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        g, beta = g.contiguous().float(), beta.contiguous()
        o, gc, Ai, w, h, d = _fwd(q, k, v, g, beta)
        ctx.save_for_backward(q, k, v, g, beta, gc, Ai, w, h, d)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, gc, Ai, w, h, d = ctx.saved_tensors
        dq, dk, dv, dg, dbeta = _bwd(q, k, v, g, beta, gc, Ai, w, h, d, do.contiguous())
        return dq, dk, dv, dg.to(ctx.g_dtype), dbeta


def gated_delta_rule(
    q: torch.Tensor,      # [B, T, H, K]
    k: torch.Tensor,      # [B, T, H, K]
    v: torch.Tensor,      # [B, T, H, V]
    g: torch.Tensor,      # [B, T, H]
    beta: torch.Tensor,   # [B, T, H]
) -> torch.Tensor:        # [B, T, H, V]
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _GatedDeltaRule.apply(q, k, v, g, beta)
