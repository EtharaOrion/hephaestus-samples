"""Chunk-parallel gated delta rule, forward and backward, in Triton.

Per (batch, value-head) the sequence is cut into chunks of BT timesteps. Within a
chunk the log-gate is accumulated once, the mutual delta corrections collapse into
a unit lower-triangular system that is inverted exactly by block doubling, and the
[K, V] recurrent state is carried across chunk boundaries in float32.
"""

import torch
import triton
import triton.language as tl

BT = 64
MAX_EXP = tl.constexpr(30.0)


@triton.jit
def _ex(x):
    return tl.exp(tl.minimum(x, MAX_EXP))


@triton.jit
def gdr_cumsum(g, a, T, H, NT, BT: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    o = tl.arange(0, BT)
    t = i_t * BT + o
    m = t < T
    b_g = tl.load(g + (i_b.to(tl.int64) * T + t) * H + i_h, mask=m, other=0.0).to(tl.float32)
    tl.store(a + i_bh.to(tl.int64) * (NT * BT) + i_t * BT + o, tl.cumsum(b_g, axis=0))


@triton.jit
def _tri_inv(b_M, BT: tl.constexpr, NLEV: tl.constexpr):
    """Exact inverse of (I + b_M) for strictly lower-triangular b_M.

    Doubling on the block size: given the inverses of the two diagonal halves,
    X - X B X supplies the off-diagonal block of the inverse of the whole.
    """
    i = tl.arange(0, BT)
    b_X = (i[:, None] == i[None, :]).to(tl.float32)
    for lvl in tl.static_range(0, NLEV):
        s = 1 << lvl
        mm = ((i[:, None] // (2 * s)) == (i[None, :] // (2 * s))) & \
             ((i[:, None] // s) != (i[None, :] // s))
        b_B = tl.where(mm, b_M, 0.0)
        b_X = b_X - tl.dot(tl.dot(b_X, b_B, input_precision="tf32x3"), b_X,
                           input_precision="tf32x3")
    return b_X


@triton.jit
def gdr_wy(k, v, beta, a, Tinv, w, uu, T, K, V, H, HG, NT,
           BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NLEV: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_hv = i_bh % H
    HK = H // HG
    i_hk = i_hv // HG
    o = tl.arange(0, BT)
    t = i_t * BT + o
    m_t = t < T
    ok = tl.arange(0, BK)
    m_k = ok < K

    b_a = tl.load(a + i_bh.to(tl.int64) * (NT * BT) + i_t * BT + o)
    b_beta = tl.load(beta + (i_b.to(tl.int64) * T + t) * H + i_hv,
                     mask=m_t, other=0.0).to(tl.float32)
    b_k = tl.load(k + ((i_b.to(tl.int64) * T + t[:, None]) * HK + i_hk) * K + ok[None, :],
                  mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)

    b_C = tl.dot(b_k, tl.trans(b_k))
    lo = o[:, None] > o[None, :]
    b_M = tl.where(lo, b_beta[:, None] * _ex(b_a[:, None] - b_a[None, :]) * b_C, 0.0)
    b_X = _tri_inv(b_M, BT, NLEV)

    tl.store(Tinv + i_bh.to(tl.int64) * (NT * BT * BT) + i_t * BT * BT
             + o[:, None] * BT + o[None, :], b_X)

    b_kg = b_k * (b_beta * _ex(b_a))[:, None]
    base = (i_bh.to(tl.int64) * NT + i_t) * BT
    tl.store(w + base * K + o[:, None] * K + ok[None, :],
             tl.dot(b_X, b_kg, input_precision="tf32x3"), mask=m_k[None, :])

    for i_v in range(0, tl.cdiv(V, BV)):
        ov = i_v * BV + tl.arange(0, BV)
        m_v = ov < V
        b_v = tl.load(v + ((i_b.to(tl.int64) * T + t[:, None]) * H + i_hv) * V + ov[None, :],
                      mask=m_t[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        tl.store(uu + base * V + o[:, None] * V + ov[None, :],
                 tl.dot(b_X, b_v * b_beta[:, None], input_precision="tf32x3"),
                 mask=m_v[None, :])


@triton.jit
def gdr_state(k, a, w, uu, U, h, T, K, V, H, HG, NT,
              BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_hv = i_bh % H
    HK = H // HG
    i_hk = i_hv // HG
    o = tl.arange(0, BT)
    ok = tl.arange(0, BK)
    ov = i_v * BV + tl.arange(0, BV)
    m_k = ok < K
    m_v = ov < V
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    for i_t in range(0, NT):
        t = i_t * BT + o
        m_t = t < T
        base = (i_bh.to(tl.int64) * NT + i_t) * BT
        tl.store(h + ((i_bh.to(tl.int64) * NT + i_t) * K + ok[:, None]) * V + ov[None, :],
                 b_h, mask=m_k[:, None] & m_v[None, :])
        b_w = tl.load(w + base * K + o[:, None] * K + ok[None, :], mask=m_k[None, :], other=0.0)
        b_u = tl.load(uu + base * V + o[:, None] * V + ov[None, :], mask=m_v[None, :], other=0.0)
        b_U = b_u - tl.dot(b_w, b_h, input_precision="tf32x3")
        tl.store(U + base * V + o[:, None] * V + ov[None, :], b_U, mask=m_v[None, :])
        b_a = tl.load(a + i_bh.to(tl.int64) * (NT * BT) + i_t * BT + o)
        a_tot = tl.sum(tl.where(o == BT - 1, b_a, 0.0), axis=0)
        b_k = tl.load(k + ((i_b.to(tl.int64) * T + t[:, None]) * HK + i_hk) * K + ok[None, :],
                      mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_kd = b_k * _ex(a_tot - b_a)[:, None]
        b_h = b_h * _ex(a_tot) + tl.dot(tl.trans(b_kd), b_U, input_precision="tf32x3")


@triton.jit
def gdr_out(q, k, a, U, h, o_ptr, T, K, V, H, HG, NT, scale,
            BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_hv = i_bh % H
    HK = H // HG
    i_hk = i_hv // HG
    o = tl.arange(0, BT)
    t = i_t * BT + o
    m_t = t < T
    ok = tl.arange(0, BK)
    ov = i_v * BV + tl.arange(0, BV)
    m_k = ok < K
    m_v = ov < V

    b_a = tl.load(a + i_bh.to(tl.int64) * (NT * BT) + i_t * BT + o)
    b_q = tl.load(q + ((i_b.to(tl.int64) * T + t[:, None]) * HK + i_hk) * K + ok[None, :],
                  mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_k = tl.load(k + ((i_b.to(tl.int64) * T + t[:, None]) * HK + i_hk) * K + ok[None, :],
                  mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_P = tl.where(o[:, None] >= o[None, :],
                   _ex(b_a[:, None] - b_a[None, :]) * tl.dot(b_q, tl.trans(b_k)) * scale, 0.0)
    base = (i_bh.to(tl.int64) * NT + i_t) * BT
    b_U = tl.load(U + base * V + o[:, None] * V + ov[None, :], mask=m_v[None, :], other=0.0)
    b_h = tl.load(h + ((i_bh.to(tl.int64) * NT + i_t) * K + ok[:, None]) * V + ov[None, :],
                  mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_o = tl.dot(b_q * (_ex(b_a) * scale)[:, None], b_h, input_precision="tf32x3") \
        + tl.dot(b_P, b_U, input_precision="tf32x3")
    tl.store(o_ptr + ((i_b.to(tl.int64) * T + t[:, None]) * H + i_hv) * V + ov[None, :],
             b_o, mask=m_t[:, None] & m_v[None, :])


@triton.jit
def gdr_dul(q, k, a, do, dUl, T, K, V, H, HG, NT, scale,
            BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // H
    i_hv = i_bh % H
    HK = H // HG
    i_hk = i_hv // HG
    o = tl.arange(0, BT)
    t = i_t * BT + o
    m_t = t < T
    ok = tl.arange(0, BK)
    ov = i_v * BV + tl.arange(0, BV)
    m_k = ok < K
    m_v = ov < V
    b_a = tl.load(a + i_bh.to(tl.int64) * (NT * BT) + i_t * BT + o)
    b_q = tl.load(q + ((i_b.to(tl.int64) * T + t[:, None]) * HK + i_hk) * K + ok[None, :],
                  mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_k = tl.load(k + ((i_b.to(tl.int64) * T + t[:, None]) * HK + i_hk) * K + ok[None, :],
                  mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_P = tl.where(o[:, None] >= o[None, :],
                   _ex(b_a[:, None] - b_a[None, :]) * tl.dot(b_q, tl.trans(b_k)) * scale, 0.0)
    b_do = tl.load(do + ((i_b.to(tl.int64) * T + t[:, None]) * H + i_hv) * V + ov[None, :],
                   mask=m_t[:, None] & m_v[None, :], other=0.0).to(tl.float32)
    base = (i_bh.to(tl.int64) * NT + i_t) * BT
    tl.store(dUl + base * V + o[:, None] * V + ov[None, :],
             tl.dot(tl.trans(b_P), b_do, input_precision="tf32x3"), mask=m_v[None, :])


@triton.jit
def gdr_dstate(q, k, a, w, do, dUl, dU, dh, T, K, V, H, HG, NT, scale,
               BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_hv = i_bh % H
    HK = H // HG
    i_hk = i_hv // HG
    o = tl.arange(0, BT)
    ok = tl.arange(0, BK)
    ov = i_v * BV + tl.arange(0, BV)
    m_k = ok < K
    m_v = ov < V
    b_ds = tl.zeros([BK, BV], dtype=tl.float32)

    for i_r in range(0, NT):
        i_t = NT - 1 - i_r
        t = i_t * BT + o
        m_t = t < T
        base = (i_bh.to(tl.int64) * NT + i_t) * BT
        b_a = tl.load(a + i_bh.to(tl.int64) * (NT * BT) + i_t * BT + o)
        a_tot = tl.sum(tl.where(o == BT - 1, b_a, 0.0), axis=0)
        b_k = tl.load(k + ((i_b.to(tl.int64) * T + t[:, None]) * HK + i_hk) * K + ok[None, :],
                      mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_kd = b_k * _ex(a_tot - b_a)[:, None]
        b_dul = tl.load(dUl + base * V + o[:, None] * V + ov[None, :], mask=m_v[None, :], other=0.0)
        b_dU = b_dul + tl.dot(b_kd, b_ds, input_precision="tf32x3")
        tl.store(dU + base * V + o[:, None] * V + ov[None, :], b_dU, mask=m_v[None, :])
        tl.store(dh + ((i_bh.to(tl.int64) * NT + i_t) * K + ok[:, None]) * V + ov[None, :],
                 b_ds, mask=m_k[:, None] & m_v[None, :])
        b_q = tl.load(q + ((i_b.to(tl.int64) * T + t[:, None]) * HK + i_hk) * K + ok[None, :],
                      mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_qg = b_q * (_ex(b_a) * scale)[:, None]
        b_do = tl.load(do + ((i_b.to(tl.int64) * T + t[:, None]) * H + i_hv) * V + ov[None, :],
                       mask=m_t[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_w = tl.load(w + base * K + o[:, None] * K + ok[None, :], mask=m_k[None, :], other=0.0)
        b_ds = tl.dot(tl.trans(b_qg), b_do, input_precision="tf32x3") + b_ds * _ex(a_tot) \
            - tl.dot(tl.trans(b_w), b_dU, input_precision="tf32x3")


@triton.jit
def gdr_dqkv(q, k, v, beta, a, do, h, dh, U, dU, uu, w, Tinv,
             dq, dk, dv, dg, dbeta, T, K, V, H, HG, NT, scale,
             BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_hv = i_bh % H
    HK = H // HG
    i_hk = i_hv // HG
    o = tl.arange(0, BT)
    t = i_t * BT + o
    m_t = t < T
    ok = tl.arange(0, BK)
    m_k = ok < K
    base = (i_bh.to(tl.int64) * NT + i_t) * BT
    hbase = (i_bh.to(tl.int64) * NT + i_t) * K

    b_a = tl.load(a + i_bh.to(tl.int64) * (NT * BT) + i_t * BT + o)
    a_tot = tl.sum(tl.where(o == BT - 1, b_a, 0.0), axis=0)
    b_beta = tl.load(beta + (i_b.to(tl.int64) * T + t) * H + i_hv,
                     mask=m_t, other=0.0).to(tl.float32)
    b_q = tl.load(q + ((i_b.to(tl.int64) * T + t[:, None]) * HK + i_hk) * K + ok[None, :],
                  mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_k = tl.load(k + ((i_b.to(tl.int64) * T + t[:, None]) * HK + i_hk) * K + ok[None, :],
                  mask=m_t[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_w = tl.load(w + base * K + o[:, None] * K + ok[None, :], mask=m_k[None, :], other=0.0)
    b_X = tl.load(Tinv + i_bh.to(tl.int64) * (NT * BT * BT) + i_t * BT * BT
                  + o[:, None] * BT + o[None, :])

    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_dqg = tl.zeros([BT, BK], dtype=tl.float32)
    b_dkd = tl.zeros([BT, BK], dtype=tl.float32)
    b_dM = tl.zeros([BT, BT], dtype=tl.float32)
    b_dP = tl.zeros([BT, BT], dtype=tl.float32)
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    d_atot = tl.zeros([1], dtype=tl.float32)

    for i_v in range(0, tl.cdiv(V, BV)):
        ov = i_v * BV + tl.arange(0, BV)
        m_v = ov < V
        b_do = tl.load(do + ((i_b.to(tl.int64) * T + t[:, None]) * H + i_hv) * V + ov[None, :],
                       mask=m_t[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_U = tl.load(U + base * V + o[:, None] * V + ov[None, :], mask=m_v[None, :], other=0.0)
        b_dU = tl.load(dU + base * V + o[:, None] * V + ov[None, :], mask=m_v[None, :], other=0.0)
        b_uu = tl.load(uu + base * V + o[:, None] * V + ov[None, :], mask=m_v[None, :], other=0.0)
        b_h = tl.load(h + (hbase + ok[:, None]) * V + ov[None, :],
                      mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_dhv = tl.load(dh + (hbase + ok[:, None]) * V + ov[None, :],
                        mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_v = tl.load(v + ((i_b.to(tl.int64) * T + t[:, None]) * H + i_hv) * V + ov[None, :],
                      mask=m_t[:, None] & m_v[None, :], other=0.0).to(tl.float32)

        b_dw -= tl.dot(b_dU, tl.trans(b_h), input_precision="tf32x3")
        b_dYv = tl.dot(tl.trans(b_X), b_dU, input_precision="tf32x3")
        b_dM -= tl.dot(b_dYv, tl.trans(b_uu), input_precision="tf32x3")
        b_dP += tl.dot(b_do, tl.trans(b_U), input_precision="tf32x3")
        b_dbeta += tl.sum(b_dYv * b_v, axis=1)
        tl.store(dv + ((i_b.to(tl.int64) * T + t[:, None]) * H + i_hv) * V + ov[None, :],
                 b_dYv * b_beta[:, None], mask=m_t[:, None] & m_v[None, :])
        b_dqg += tl.dot(b_do, tl.trans(b_h), input_precision="tf32x3")
        b_dkd += tl.dot(b_U, tl.trans(b_dhv), input_precision="tf32x3")
        d_atot += tl.sum(b_h * b_dhv)

    d_atot = d_atot * _ex(a_tot)
    b_dYk = tl.dot(tl.trans(b_X), b_dw, input_precision="tf32x3")
    b_dM -= tl.dot(b_dYk, tl.trans(b_w), input_precision="tf32x3")
    lo = o[:, None] > o[None, :]
    le = o[:, None] >= o[None, :]
    b_dM = tl.where(lo, b_dM, 0.0)
    b_dP = tl.where(le, b_dP, 0.0)

    b_e = _ex(b_a[:, None] - b_a[None, :])
    b_Ckk = tl.dot(b_k, tl.trans(b_k))
    b_Cqk = tl.dot(b_q, tl.trans(b_k)) * scale
    b_M = tl.where(lo, b_beta[:, None] * b_e * b_Ckk, 0.0)
    b_P = tl.where(le, b_e * b_Cqk, 0.0)

    b_Z1 = b_dM * b_M
    b_Z2 = b_dP * b_P
    b_da = (tl.sum(b_Z1, axis=1) - tl.sum(b_Z1, axis=0)
            + tl.sum(b_Z2, axis=1) - tl.sum(b_Z2, axis=0))

    b_dbeta += tl.sum(b_dM * b_e * b_Ckk, axis=1)
    b_R = tl.where(lo, b_dM * b_beta[:, None] * b_e, 0.0)
    b_R2 = tl.where(le, b_dP * b_e, 0.0) * scale

    b_dk = tl.dot(b_R, b_k, input_precision="tf32x3") \
        + tl.dot(tl.trans(b_R), b_k, input_precision="tf32x3") \
        + tl.dot(tl.trans(b_R2), b_q, input_precision="tf32x3")
    b_dq = tl.dot(b_R2, b_k, input_precision="tf32x3")

    b_A = _ex(b_a)
    b_qg = b_q * (b_A * scale)[:, None]
    b_dq += b_dqg * (b_A * scale)[:, None]
    b_da += tl.sum(b_dqg * b_qg, axis=1)

    b_kg = b_k * b_A[:, None]
    b_dbeta += tl.sum(b_dYk * b_kg, axis=1)
    b_dkg = b_dYk * b_beta[:, None]
    b_dk += b_dkg * b_A[:, None]
    b_da += tl.sum(b_dkg * b_kg, axis=1)

    b_ed = _ex(a_tot - b_a)
    b_kd = b_k * b_ed[:, None]
    b_dk += b_dkd * b_ed[:, None]
    b_da -= tl.sum(b_dkd * b_kd, axis=1)
    d_atot += tl.sum(b_dkd * b_kd)

    b_da = b_da + tl.where(o == BT - 1, tl.sum(d_atot, axis=0), 0.0)
    b_dgv = tl.sum(b_da, axis=0) - tl.cumsum(b_da, axis=0) + b_da

    tl.store(dq + ((i_b.to(tl.int64) * T + t[:, None]) * H + i_hv) * K + ok[None, :],
             b_dq, mask=m_t[:, None] & m_k[None, :])
    tl.store(dk + ((i_b.to(tl.int64) * T + t[:, None]) * H + i_hv) * K + ok[None, :],
             b_dk, mask=m_t[:, None] & m_k[None, :])
    tl.store(dbeta + (i_b.to(tl.int64) * T + t) * H + i_hv, b_dbeta, mask=m_t)
    tl.store(dg + (i_b.to(tl.int64) * T + t) * H + i_hv, b_dgv, mask=m_t)


@triton.jit
def gdr_group_sum(src, dst, H, HK, HG, K, BK: tl.constexpr):
    i_p = tl.program_id(0)
    i_hk = tl.program_id(1)
    ok = tl.arange(0, BK)
    m_k = ok < K
    acc = tl.zeros([BK], dtype=tl.float32)
    for j in range(0, HG):
        acc += tl.load(src + (i_p.to(tl.int64) * H + i_hk * HG + j) * K + ok, mask=m_k, other=0.0)
    tl.store(dst + (i_p.to(tl.int64) * HK + i_hk) * K + ok, acc, mask=m_k)


def _pow2(x):
    return 1 << (x - 1).bit_length()


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        B, T, HK, K = q.shape
        H, V = v.shape[2], v.shape[3]
        HG = H // HK
        NT = triton.cdiv(T, BT)
        dev = q.device
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        g, beta = g.contiguous(), beta.contiguous()

        BK = _pow2(K)
        wide = q.element_size() > 2
        NS = 1 if wide else 2
        BV = min(32 if wide else 64, _pow2(V))
        NBH = B * H
        NLEV = BT.bit_length() - 1

        a = torch.empty(NBH, NT * BT, device=dev, dtype=torch.float32)
        gdr_cumsum[(NT, NBH)](g, a, T, H, NT, BT=BT)

        Tinv = torch.empty(NBH * NT, BT, BT, device=dev, dtype=torch.float32)
        w = torch.empty(NBH * NT, BT, K, device=dev, dtype=torch.float32)
        uu = torch.empty(NBH * NT, BT, V, device=dev, dtype=torch.float32)
        gdr_wy[(NT, NBH)](k, v, beta, a, Tinv, w, uu, T, K, V, H, HG, NT,
                          BT=BT, BK=BK, BV=BV, NLEV=NLEV, num_warps=4, num_stages=NS)

        U = torch.empty(NBH * NT, BT, V, device=dev, dtype=torch.float32)
        h = torch.empty(NBH * NT, K, V, device=dev, dtype=torch.float32)
        gdr_state[(triton.cdiv(V, BV), NBH)](k, a, w, uu, U, h, T, K, V, H, HG, NT,
                                             BT=BT, BK=BK, BV=BV, num_warps=4, num_stages=NS)

        o = v.new_empty(B, T, H, V)
        gdr_out[(NT, triton.cdiv(V, BV), NBH)](q, k, a, U, h, o, T, K, V, H, HG, NT,
                                               K ** -0.5, BT=BT, BK=BK, BV=BV,
                                               num_warps=4, num_stages=NS)

        ctx.save_for_backward(q, k, v, g, beta, a, Tinv, w, uu, U, h)
        ctx.dims = (B, T, HK, H, HG, K, V, NT, BK, BV, NS)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, a, Tinv, w, uu, U, h = ctx.saved_tensors
        B, T, HK, H, HG, K, V, NT, BK, BV, NS = ctx.dims
        dev = q.device
        do = do.contiguous()
        NBH = B * H
        scale = K ** -0.5
        NV = triton.cdiv(V, BV)

        dUl = torch.empty(NBH * NT, BT, V, device=dev, dtype=torch.float32)
        gdr_dul[(NT, NV, NBH)](q, k, a, do, dUl, T, K, V, H, HG, NT, scale,
                               BT=BT, BK=BK, BV=BV, num_warps=4, num_stages=NS)
        dU = torch.empty(NBH * NT, BT, V, device=dev, dtype=torch.float32)
        dh = torch.empty(NBH * NT, K, V, device=dev, dtype=torch.float32)
        gdr_dstate[(NV, NBH)](q, k, a, w, do, dUl, dU, dh, T, K, V, H, HG, NT, scale,
                              BT=BT, BK=BK, BV=BV, num_warps=4, num_stages=NS)

        dqf = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
        dkf = torch.empty(B, T, H, K, device=dev, dtype=torch.float32)
        dv = torch.empty(B, T, H, V, device=dev, dtype=torch.float32)
        dg = torch.empty(B, T, H, device=dev, dtype=torch.float32)
        dbeta = torch.empty(B, T, H, device=dev, dtype=torch.float32)
        gdr_dqkv[(NT, NBH)](q, k, v, beta, a, do, h, dh, U, dU, uu, w, Tinv,
                            dqf, dkf, dv, dg, dbeta, T, K, V, H, HG, NT, scale,
                            BT=BT, BK=BK, BV=min(32, BV), num_warps=4, num_stages=1)

        if HG > 1:
            dq = torch.empty(B, T, HK, K, device=dev, dtype=torch.float32)
            dk = torch.empty(B, T, HK, K, device=dev, dtype=torch.float32)
            gdr_group_sum[(B * T, HK)](dqf, dq, H, HK, HG, K, BK=BK)
            gdr_group_sum[(B * T, HK)](dkf, dk, H, HK, HG, K, BK=BK)
        else:
            dq, dk = dqf, dkf
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                dg.to(g.dtype), dbeta.to(beta.dtype))


def gated_delta_rule(
    q: torch.Tensor,      # [B, T, H, K]
    k: torch.Tensor,      # [B, T, H, K]
    v: torch.Tensor,      # [B, T, HV, V]
    g: torch.Tensor,      # [B, T, HV]
    beta: torch.Tensor,   # [B, T, HV]
) -> torch.Tensor:        # [B, T, HV, V]
    """Entry point. Signature and semantics are fixed."""
    return _GatedDeltaRule.apply(q, k, v, g, beta)
