"""Routed fused mixture-of-experts layer, forward and backward.

Routing produces a stable expert-major permutation of the T*A (token, slot)
pairs; every expert projection is then a tiled GEMM over one expert's
contiguous row segment, so each row is visited once and every accumulation
order is fixed by the tiling rather than by scheduling.

float32 operands reach the tensor cores as an exact bfloat16 hi/lo pair: for
a float32 value a, ah = bf16(a) and al = bf16(a - ah) together carry ~16
mantissa bits, and ah*bh + al*bh + ah*bl reconstructs the product with
float32-effective accuracy. Natively bfloat16 operands are already exact in
bf16 and are passed through unsplit.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dot3(a, b, acc, SA: tl.constexpr, SB: tl.constexpr):
    ah = a.to(tl.bfloat16)
    bh = b.to(tl.bfloat16)
    acc = tl.dot(ah, bh, acc)
    if SA:
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(al, bh, acc)
    if SB:
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(ah, bl, acc)
    return acc


@triton.jit
def _router_kernel(X, RW, RB, SEL, W, SSEL, SSUM, T, D, E, A,
                   sxm, sxk, srk, sre,
                   BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                   SP: tl.constexpr):
    """Router scores, top-A selection and combine weights for a row block.

    Each (score, expert) pair becomes one int64 key: the order-preserving
    integer image of the biased float32 score shifted left, OR'd with the
    inverted expert index in the low 31 bits. Keys are unique across the row,
    so repeatedly taking the largest key and masking it out yields exactly
    (biased score descending, expert index ascending).
    """
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    re = tl.arange(0, BE)
    tm = rt < T
    em = re < E
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    rk = tl.arange(0, BK)
    ap = X + rt[:, None].to(tl.int64) * sxm + rk[None, :] * sxk
    bp = RW + rk[:, None] * srk + re[None, :] * sre
    for k in range(0, tl.cdiv(D, BK)):
        km = rk < D - k * BK
        a = tl.load(ap, mask=tm[:, None] & km[None, :], other=0.0)
        b = tl.load(bp, mask=km[:, None] & em[None, :], other=0.0)
        acc = _dot3(a, b, acc, SP, SP)
        ap += BK * sxk
        bp += BK * srk
    s = tl.sigmoid(acc)
    rb = tl.load(RB + re, mask=em, other=0.0)
    bs = s + rb[None, :]
    bs = tl.where(bs == 0.0, 0.0, bs)
    ib = bs.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    key = (u << 31) | (E - 1 - re[None, :]).to(tl.int64)
    key = tl.where(em[None, :], key, -1)
    tot = tl.zeros((BT,), dtype=tl.float32)
    for a in range(0, A):
        mx = tl.max(key, axis=1)
        e = ((E - 1).to(tl.int64) - (mx & 0x7FFFFFFF)).to(tl.int32)
        hit = re[None, :] == e[:, None]
        sv = tl.sum(tl.where(hit, s, 0.0), axis=1)
        tl.store(SEL + rt * A + a, e, mask=tm)
        tl.store(SSEL + rt * A + a, sv, mask=tm)
        tot += sv
        key = tl.where(hit, -1, key)
    tl.store(SSUM + rt, tot, mask=tm)
    for a in range(0, A):
        sv = tl.load(SSEL + rt * A + a, mask=tm, other=0.0)
        tl.store(W + rt * A + a, sv / tot, mask=tm)


@triton.jit
def _hist_kernel(SEL, CNT, N, E, SUB: tl.constexpr, OUTER: tl.constexpr,
                 BE: tl.constexpr):
    b = tl.program_id(0)
    re = tl.arange(0, BE)
    acc = tl.zeros((BE,), dtype=tl.int32)
    base = b * (SUB * OUTER)
    for c in range(0, OUTER):
        idx = base + c * SUB + tl.arange(0, SUB)
        e = tl.load(SEL + idx, mask=idx < N, other=-1).to(tl.int32)
        acc += tl.sum((e[:, None] == re[None, :]).to(tl.int32), axis=0)
    tl.store(CNT + b * BE + re, acc, mask=re < E)


@triton.jit
def _scan_kernel(CNT, SEG, CNTS, TSTART, TEND, NB, E, BM,
                 CH: tl.constexpr, BE: tl.constexpr):
    re = tl.arange(0, BE)
    rc = tl.arange(0, CH)
    em = re < E
    run = tl.zeros((BE,), dtype=tl.int32)
    for c in range(0, tl.cdiv(NB, CH)):
        rb = c * CH + rc
        m = (rb[:, None] < NB) & em[None, :]
        v = tl.load(CNT + rb[:, None] * BE + re[None, :], mask=m, other=0)
        inc = tl.cumsum(v, axis=0)
        tl.store(CNT + rb[:, None] * BE + re[None, :],
                 inc - v + run[None, :], mask=m)
        run += tl.sum(v, axis=0)
    tot = tl.where(em, run, 0)
    tl.store(SEG + re, tl.cumsum(tot, axis=0) - tot, mask=em)
    tl.store(CNTS + re, tot, mask=em)
    tc = tl.where(em, (tot + BM - 1) // BM, 0)
    tend = tl.cumsum(tc, axis=0)
    tl.store(TSTART + re, tend - tc, mask=em)
    tl.store(TEND + re, tl.where(em, tend, 0x7FFFFFFF))


@triton.jit
def _scatter_kernel(SEL, WFLAT, CNT, SEG, ORDER, TOK, POS, WSORT, N, A, E,
                    SUB: tl.constexpr, OUTER: tl.constexpr, BE: tl.constexpr):
    b = tl.program_id(0)
    rs = tl.arange(0, SUB)
    re = tl.arange(0, BE)
    base = b * (SUB * OUTER)
    run = tl.zeros((BE,), dtype=tl.int32)
    for c in range(0, OUTER):
        idx = base + c * SUB + rs
        m = idx < N
        e = tl.load(SEL + idx, mask=m, other=-1).to(tl.int32)
        eq = (e[:, None] == re[None, :]).to(tl.int32)
        rank = tl.sum(((e[:, None] == e[None, :]) &
                       (rs[None, :] < rs[:, None])).to(tl.int32), axis=1)
        rank += tl.sum(eq * run[None, :], axis=1)
        run += tl.sum(eq, axis=0)
        ec = tl.where(e < 0, 0, e)
        pos = (tl.load(CNT + b * BE + ec, mask=m, other=0)
               + tl.load(SEG + ec, mask=m, other=0) + rank)
        tl.store(ORDER + pos, idx.to(tl.int32), mask=m)
        tl.store(TOK + pos, (idx // A).to(tl.int32), mask=m)
        tl.store(POS + idx, pos.to(tl.int32), mask=m)
        tl.store(WSORT + pos, tl.load(WFLAT + idx, mask=m, other=0.0), mask=m)


@triton.jit
def _tile_lookup(TEND, TSTART, SEG, CNTS, j, E, BM: tl.constexpr,
                 BE: tl.constexpr):
    re = tl.arange(0, BE)
    tend = tl.load(TEND + re)
    e = tl.sum((tend <= j).to(tl.int32), axis=0)
    ec = tl.where(e < E, e, 0)
    ts = tl.load(TSTART + ec)
    sg = tl.load(SEG + ec)
    cn = tl.load(CNTS + ec)
    return e, sg + (j - ts) * BM, sg + cn


@triton.jit
def _gemm1_kernel(X, W1, H, TOK, TEND, TSTART, SEG, CNTS, D, F2, E, NN,
                  sxm, sw1e, sw1d, sw1f,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  BE: tl.constexpr, SA: tl.constexpr, SB: tl.constexpr):
    pid = tl.program_id(0)
    j = pid // NN
    pn = pid % NN
    e, r0, rend = _tile_lookup(TEND, TSTART, SEG, CNTS, j, E, BM, BE)
    if e < E:
        rows = r0 + tl.arange(0, BM)
        mrow = rows < rend
        tok = tl.load(TOK + rows, mask=mrow, other=0).to(tl.int32)
        rn = pn * BN + tl.arange(0, BN)
        mn = rn < F2
        rk = tl.arange(0, BK)
        ap = X + tok[:, None].to(tl.int64) * sxm + rk[None, :]
        bp = W1 + e.to(tl.int64) * sw1e + rk[:, None] * sw1d + rn[None, :] * sw1f
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(D, BK)):
            km = rk < D - k * BK
            a = tl.load(ap, mask=mrow[:, None] & km[None, :], other=0.0)
            b = tl.load(bp, mask=km[:, None] & mn[None, :], other=0.0)
            acc = _dot3(a, b, acc, SA, SB)
            ap += BK
            bp += BK * sw1d
        tl.store(H + rows[:, None].to(tl.int64) * F2 + rn[None, :], acc,
                 mask=mrow[:, None] & mn[None, :])


@triton.jit
def _gemm2_kernel(H, W2, YP, ORDER, WSORT, TEND, TSTART, SEG, CNTS,
                  F, D, E, NN, sw2e, sw2f, sw2d,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  BE: tl.constexpr, SA: tl.constexpr, SB: tl.constexpr):
    pid = tl.program_id(0)
    j = pid // NN
    pn = pid % NN
    e, r0, rend = _tile_lookup(TEND, TSTART, SEG, CNTS, j, E, BM, BE)
    if e < E:
        rows = r0 + tl.arange(0, BM)
        mrow = rows < rend
        rn = pn * BN + tl.arange(0, BN)
        mn = rn < D
        rk = tl.arange(0, BK)
        F2 = 2 * F
        hp = H + rows[:, None].to(tl.int64) * F2 + rk[None, :]
        bp = W2 + e.to(tl.int64) * sw2e + rk[:, None] * sw2f + rn[None, :] * sw2d
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(F, BK)):
            km = rk < F - k * BK
            hm = mrow[:, None] & km[None, :]
            g = tl.load(hp, mask=hm, other=0.0)
            u = tl.load(hp + F, mask=hm, other=0.0)
            a = (g * tl.sigmoid(g)) * u
            b = tl.load(bp, mask=km[:, None] & mn[None, :], other=0.0)
            acc = _dot3(a, b, acc, SA, SB)
            hp += BK
            bp += BK * sw2f
        wv = tl.load(WSORT + rows, mask=mrow, other=0.0)
        orow = tl.load(ORDER + rows, mask=mrow, other=0).to(tl.int32)
        tl.store(YP + orow[:, None].to(tl.int64) * D + rn[None, :],
                 acc * wv[:, None], mask=mrow[:, None] & mn[None, :])


@triton.jit
def _combine_kernel(YP, Y, T, D, A, ND, BT: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    rt = (pid // ND) * BT + tl.arange(0, BT)
    rd = (pid % ND) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    m = mt[:, None] & md[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(0, A):
        p = (rt * A + a).to(tl.int64)
        acc += tl.load(YP + p[:, None] * D + rd[None, :], mask=m, other=0.0)
    tl.store(Y + rt[:, None].to(tl.int64) * D + rd[None, :],
             acc.to(Y.dtype.element_ty), mask=m)


@triton.jit
def _dact_kernel(DY, W2, H, RP, TOK, WSORT, TEND, TSTART, SEG, CNTS,
                 N, F, D, E, NN, sdm, sw2e, sw2f, sw2d,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 BE: tl.constexpr, SA: tl.constexpr, SB: tl.constexpr):
    pid = tl.program_id(0)
    j = pid // NN
    pn = pid % NN
    e, r0, rend = _tile_lookup(TEND, TSTART, SEG, CNTS, j, E, BM, BE)
    if e < E:
        rows = r0 + tl.arange(0, BM)
        mrow = rows < rend
        tok = tl.load(TOK + rows, mask=mrow, other=0).to(tl.int32)
        rn = pn * BN + tl.arange(0, BN)
        mn = rn < F
        rk = tl.arange(0, BK)
        ap = DY + tok[:, None].to(tl.int64) * sdm + rk[None, :]
        bp = W2 + e.to(tl.int64) * sw2e + rn[None, :] * sw2f + rk[:, None] * sw2d
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(D, BK)):
            km = rk < D - k * BK
            a = tl.load(ap, mask=mrow[:, None] & km[None, :], other=0.0)
            b = tl.load(bp, mask=km[:, None] & mn[None, :], other=0.0)
            acc = _dot3(a, b, acc, SA, SB)
            ap += BK
            bp += BK * sw2d
        F2 = 2 * F
        hp = H + rows[:, None].to(tl.int64) * F2 + rn[None, :]
        hm = mrow[:, None] & mn[None, :]
        g = tl.load(hp, mask=hm, other=0.0)
        u = tl.load(hp + F, mask=hm, other=0.0)
        sg = tl.sigmoid(g)
        silu = g * sg
        tl.store(RP + pn * N + rows, tl.sum(acc * (silu * u), axis=1),
                 mask=mrow)
        wv = tl.load(WSORT + rows, mask=mrow, other=0.0)
        da = acc * wv[:, None]
        tl.store(hp, da * u * (sg + silu * (1.0 - sg)), mask=hm)
        tl.store(hp + F, da * silu, mask=hm)


@triton.jit
def _rd_reduce_kernel(RP, RD, N, NFT, BLK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BLK + tl.arange(0, BLK)
    m = r < N
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for q in range(0, NFT):
        acc += tl.load(RP + q * N + r, mask=m, other=0.0)
    tl.store(RD + r, acc, mask=m)


@triton.jit
def _dweights_kernel(RD, POS, SEL, W, SSEL, SSUM, DLOGIT, T, E, A,
                     BT: tl.constexpr, BE: tl.constexpr):
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    re = tl.arange(0, BE)
    mt = rt < T
    inner = tl.zeros((BT,), dtype=tl.float32)
    for a in range(0, A):
        p = tl.load(POS + rt * A + a, mask=mt, other=0).to(tl.int32)
        rd = tl.load(RD + p, mask=mt, other=0.0)
        wv = tl.load(W + rt * A + a, mask=mt, other=0.0)
        inner += wv * rd
    ssum = tl.load(SSUM + rt, mask=mt, other=1.0)
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    for a in range(0, A):
        p = tl.load(POS + rt * A + a, mask=mt, other=0).to(tl.int32)
        rd = tl.load(RD + p, mask=mt, other=0.0)
        sv = tl.load(SSEL + rt * A + a, mask=mt, other=0.0)
        e = tl.load(SEL + rt * A + a, mask=mt, other=-1).to(tl.int32)
        dl = (rd - inner) / ssum * sv * (1.0 - sv)
        acc += tl.where(e[:, None] == re[None, :], dl[:, None], 0.0)
    tl.store(DLOGIT + rt[:, None].to(tl.int64) * E + re[None, :], acc,
             mask=mt[:, None] & (re[None, :] < E))


@triton.jit
def _dxg_kernel(DGU, W1, DXG, ORDER, TEND, TSTART, SEG, CNTS,
                F2, D, E, NN, sw1e, sw1d, sw1f,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                BE: tl.constexpr, SA: tl.constexpr, SB: tl.constexpr):
    pid = tl.program_id(0)
    j = pid // NN
    pn = pid % NN
    e, r0, rend = _tile_lookup(TEND, TSTART, SEG, CNTS, j, E, BM, BE)
    if e < E:
        rows = r0 + tl.arange(0, BM)
        mrow = rows < rend
        rn = pn * BN + tl.arange(0, BN)
        mn = rn < D
        rk = tl.arange(0, BK)
        ap = DGU + rows[:, None].to(tl.int64) * F2 + rk[None, :]
        bp = W1 + e.to(tl.int64) * sw1e + rn[None, :] * sw1d + rk[:, None] * sw1f
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(F2, BK)):
            km = rk < F2 - k * BK
            a = tl.load(ap, mask=mrow[:, None] & km[None, :], other=0.0)
            b = tl.load(bp, mask=km[:, None] & mn[None, :], other=0.0)
            acc = _dot3(a, b, acc, SA, SB)
            ap += BK
            bp += BK * sw1f
        orow = tl.load(ORDER + rows, mask=mrow, other=0).to(tl.int32)
        tl.store(DXG + orow[:, None].to(tl.int64) * D + rn[None, :], acc,
                 mask=mrow[:, None] & mn[None, :])


@triton.jit
def _dw1_kernel(X, DGU, DW1, TOK, SEG, CNTS, D, F2, NN,
                sxm, sw1e, sw1d, sw1f,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                SA: tl.constexpr, SB: tl.constexpr):
    e = tl.program_id(1)
    pid = tl.program_id(0)
    rm = (pid // NN) * BM + tl.arange(0, BM)
    rn = (pid % NN) * BN + tl.arange(0, BN)
    r0 = tl.load(SEG + e)
    cnt = tl.load(CNTS + e)
    mm = rm < D
    mn = rn < F2
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(cnt, BK)):
        kk = rk + k * BK
        km = kk < cnt
        tok = tl.load(TOK + r0 + kk, mask=km, other=0).to(tl.int32)
        a = tl.load(X + tok[None, :].to(tl.int64) * sxm + rm[:, None],
                    mask=mm[:, None] & km[None, :], other=0.0)
        b = tl.load(DGU + (r0 + kk)[:, None].to(tl.int64) * F2 + rn[None, :],
                    mask=km[:, None] & mn[None, :], other=0.0)
        acc = _dot3(a, b, acc, SA, SB)
    tl.store(DW1 + e.to(tl.int64) * sw1e + rm[:, None] * sw1d
             + rn[None, :] * sw1f, acc.to(DW1.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _dw2_kernel(H, DY, DW2, TOK, WSORT, SEG, CNTS, F, D, NN,
                sdm, sw2e, sw2f, sw2d,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                SA: tl.constexpr, SB: tl.constexpr):
    e = tl.program_id(1)
    pid = tl.program_id(0)
    rm = (pid // NN) * BM + tl.arange(0, BM)
    rn = (pid % NN) * BN + tl.arange(0, BN)
    r0 = tl.load(SEG + e)
    cnt = tl.load(CNTS + e)
    mm = rm < F
    mn = rn < D
    rk = tl.arange(0, BK)
    F2 = 2 * F
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(cnt, BK)):
        kk = rk + k * BK
        km = kk < cnt
        rows = (r0 + kk).to(tl.int64)
        tok = tl.load(TOK + r0 + kk, mask=km, other=0).to(tl.int32)
        wv = tl.load(WSORT + r0 + kk, mask=km, other=0.0)
        hp = H + rows[None, :] * F2 + rm[:, None]
        am = mm[:, None] & km[None, :]
        g = tl.load(hp, mask=am, other=0.0)
        u = tl.load(hp + F, mask=am, other=0.0)
        a = (g * tl.sigmoid(g)) * u * wv[None, :]
        b = tl.load(DY + tok[:, None].to(tl.int64) * sdm + rn[None, :],
                    mask=km[:, None] & mn[None, :], other=0.0)
        acc = _dot3(a, b, acc, SA, SB)
    tl.store(DW2 + e.to(tl.int64) * sw2e + rm[:, None] * sw2f
             + rn[None, :] * sw2d, acc.to(DW2.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _drw_kernel(X, DLOGIT, PART, T, D, E, SK, EPAD, sxm,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                SA: tl.constexpr, SB: tl.constexpr):
    pm = tl.program_id(0)
    ks = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = tl.program_id(2) * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < E
    chunk = tl.cdiv(tl.cdiv(T, SK), BK) * BK
    k0 = ks * chunk
    kend = min(k0 + chunk, T)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(k0, kend, BK):
        kk = k + rk
        km = kk < kend
        a = tl.load(X + kk[None, :].to(tl.int64) * sxm + rm[:, None],
                    mask=mm[:, None] & km[None, :], other=0.0)
        b = tl.load(DLOGIT + kk[:, None].to(tl.int64) * E + rn[None, :],
                    mask=km[:, None] & mn[None, :], other=0.0)
        acc = _dot3(a, b, acc, SA, SB)
    tl.store(PART + ks.to(tl.int64) * (D * EPAD) + rm[:, None].to(tl.int64)
             * EPAD + rn[None, :], acc, mask=mm[:, None] & mn[None, :])


@triton.jit
def _drw_reduce_kernel(PART, DRW, D, E, SK, EPAD, BLK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BLK + tl.arange(0, BLK)
    m = r < D * E
    p = (r // E).to(tl.int64) * EPAD + (r % E)
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for q in range(0, SK):
        acc += tl.load(PART + q.to(tl.int64) * (D * EPAD) + p, mask=m,
                       other=0.0)
    tl.store(DRW + r, acc.to(DRW.dtype.element_ty), mask=m)


@triton.jit
def _dx_kernel(DXG, DLOGIT, RW, DX, T, D, E, A, ND, srk, sre,
               BT: tl.constexpr, BD: tl.constexpr, BE: tl.constexpr,
               SA: tl.constexpr, SB: tl.constexpr):
    pid = tl.program_id(0)
    rt = (pid // ND) * BT + tl.arange(0, BT)
    rd = (pid % ND) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    m = mt[:, None] & md[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(0, A):
        p = (rt * A + a).to(tl.int64)
        acc += tl.load(DXG + p[:, None] * D + rd[None, :], mask=m, other=0.0)
    re = tl.arange(0, BE)
    for e0 in range(0, E, BE):
        ce = e0 + re
        me = ce < E
        dl = tl.load(DLOGIT + rt[:, None].to(tl.int64) * E + ce[None, :],
                     mask=mt[:, None] & me[None, :], other=0.0)
        rwt = tl.load(RW + ce[:, None] * srk + rd[None, :] * sre,
                      mask=me[:, None] & md[None, :], other=0.0)
        acc = _dot3(dl, rwt, acc, SA, SB)
    tl.store(DX + rt[:, None].to(tl.int64) * D + rd[None, :],
             acc.to(DX.dtype.element_ty), mask=m)


def _p2(n, lo=1):
    return 1 << (max(int(n), lo) - 1).bit_length()


def _routing(x, router_w, router_b, A, E, sp):
    T, D = x.shape
    dev = x.device
    be = _p2(E, 16)
    bt = max(16, min(128, 8192 // be))
    bk = _p2(min(D, 64), 16)
    sel = torch.empty((T, A), device=dev, dtype=torch.int32)
    w = torch.empty((T, A), device=dev, dtype=torch.float32)
    ssel = torch.empty((T, A), device=dev, dtype=torch.float32)
    ssum = torch.empty((T,), device=dev, dtype=torch.float32)
    _router_kernel[(triton.cdiv(T, bt),)](
        x, router_w, router_b, sel, w, ssel, ssum, T, D, E, A,
        x.stride(0), x.stride(1), router_w.stride(0), router_w.stride(1),
        BT=bt, BE=be, BK=bk, SP=sp, num_warps=4, num_stages=3)
    return sel, w, ssel, ssum


def _dispatch(sel, w, N, A, E, BM):
    dev = sel.device
    sub = 64
    outer = 1
    while triton.cdiv(N, sub * outer) > 1024 and outer < 16:
        outer *= 2
    nb = triton.cdiv(N, sub * outer)
    be = _p2(E, 16)
    cnt = torch.empty((nb, be), device=dev, dtype=torch.int32)
    meta = torch.empty((4, be), device=dev, dtype=torch.int32)
    seg, cnts, tstart, tend = meta[0], meta[1], meta[2], meta[3]
    selflat = sel.view(-1)
    _hist_kernel[(nb,)](selflat, cnt, N, E, SUB=sub, OUTER=outer, BE=be,
                        num_warps=4)
    ch = max(1, min(64, 8192 // be))
    _scan_kernel[(1,)](cnt, seg, cnts, tstart, tend, nb, E, BM,
                       CH=ch, BE=be, num_warps=4)
    idx = torch.empty((3, N), device=dev, dtype=torch.int32)
    order, tok, pos = idx[0], idx[1], idx[2]
    wsort = torch.empty((N,), device=dev, dtype=torch.float32)
    _scatter_kernel[(nb,)](selflat, w.view(-1), cnt, seg, order, tok, pos,
                           wsort, N, A, E, SUB=sub, OUTER=outer, BE=be,
                           num_warps=4)
    return seg, cnts, tstart, tend, order, tok, pos, wsort


def _bm(N, E):
    avg = N / max(E, 1)
    if avg >= 256:
        return 128
    if avg >= 64:
        return 64
    return 32


# Tile shapes (BN, BK, warps, stages) for the four expert GEMMs, keyed by
# operand dtype. The float32 rows split both operands into bf16 hi/lo tiles,
# so a stage holds twice the operand data and the tiles are correspondingly
# narrower to stay inside the per-block shared-memory limit.
GEMM1CFG = {False: (128, 64, 8, 3), True: (128, 32, 4, 4)}
GEMM2CFG = {False: (256, 32, 8, 3), True: (256, 32, 8, 3)}
DACTCFG = {False: (64, 128, 8, 3), True: (64, 64, 4, 3)}
DXGCFG = {False: (256, 64, 8, 3), True: (128, 32, 4, 4)}
# Tile shapes (BM, BN, BK, warps, stages) for the weight gradients, widest
# first. One program owns one output tile of one expert and walks that
# expert's whole row segment, so expert popularity skew turns directly into
# block-runtime skew; _wcfg steps down this list until the tile grid holds
# enough blocks to spread that skew over the SMs.
WCFG1 = {False: [(256, 64, 64, 8, 3), (128, 64, 64, 4, 3), (64, 32, 128, 4, 3)],
         True: [(128, 128, 32, 4, 4), (128, 64, 32, 4, 4), (64, 32, 128, 4, 2)]}
WCFG2 = {False: [(64, 256, 64, 4, 4), (64, 128, 128, 4, 2)],
         True: [(128, 128, 32, 4, 3), (64, 128, 64, 4, 3), (64, 32, 128, 4, 4)]}
WTILES = 800


def _gcfg(tbl, fp32, bm, n):
    bn, bk, w, st = tbl[fp32]
    bn = min(bn, _p2(n, 16))
    return dict(BN=bn, BK=bk, num_warps=w if bm * bn >= 8192 else 4,
                num_stages=st)


def _wcfg(tbl, fp32, m, n, e):
    cands = tbl[fp32]
    for i, (bm, bn, bk, w, st) in enumerate(cands):
        bm = min(bm, _p2(m, 16))
        bn = min(bn, _p2(n, 16))
        tiles = -(-m // bm) * (-(-n // bn)) * e
        if tiles >= WTILES or i == len(cands) - 1:
            return dict(BM=bm, BN=bn, BK=bk,
                        num_warps=w if bm * bn >= 8192 else 4, num_stages=st)


class _FusedMoe(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = router_w.shape[1]
        F = w1.shape[2] // 2
        F2 = 2 * F
        N = T * A
        dev = x.device
        fp32 = x.dtype == torch.float32
        sp = fp32
        sel, w, ssel, ssum = _routing(x, router_w, router_b, A, E, sp)
        BM = _bm(N, E)
        seg, cnts, tstart, tend, order, tok, pos, wsort = _dispatch(
            sel, w, N, A, E, BM)
        be = _p2(E, 16)
        ntm = triton.cdiv(N, BM) + E
        h = torch.empty((N, F2), device=dev, dtype=torch.float32)
        c1 = _gcfg(GEMM1CFG, fp32, BM, F2)
        nn1 = triton.cdiv(F2, c1["BN"])
        _gemm1_kernel[(ntm * nn1,)](
            x, w1, h, tok, tend, tstart, seg, cnts, D, F2, E, nn1,
            x.stride(0), w1.stride(0), w1.stride(1), w1.stride(2),
            BM=BM, BE=be, SA=sp, SB=sp, **c1)
        yp = torch.empty((N, D), device=dev, dtype=torch.float32)
        c2 = _gcfg(GEMM2CFG, fp32, BM, D)
        nn2 = triton.cdiv(D, c2["BN"])
        _gemm2_kernel[(ntm * nn2,)](
            h, w2, yp, order, wsort, tend, tstart, seg, cnts, F, D, E, nn2,
            w2.stride(0), w2.stride(1), w2.stride(2),
            BM=BM, BE=be, SA=sp, SB=sp, **c2)
        y = torch.empty((T, D), device=dev, dtype=x.dtype)
        bd = _p2(min(D, 128), 16)
        bt = max(1, 4096 // bd)
        nd = triton.cdiv(D, bd)
        _combine_kernel[(triton.cdiv(T, bt) * nd,)](
            yp, y, T, D, A, nd, BT=bt, BD=bd, num_warps=4)
        ctx.save_for_backward(x, router_w, w1, w2, h, sel, w, ssel, ssum,
                              seg, cnts, tstart, tend, order, tok, pos, wsort)
        ctx.meta = (T, D, E, F, A, N, BM, be, ntm, fp32)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, h, sel, w, ssel, ssum, seg, cnts, tstart, tend,
         order, tok, pos, wsort) = ctx.saved_tensors
        T, D, E, F, A, N, BM, be, ntm, fp32 = ctx.meta
        F2 = 2 * F
        dev = x.device
        sp = fp32
        dy = dy.contiguous()
        # dw2 consumes the forward gate/up values, which _dact_kernel then
        # overwrites in place with dgu.
        w2c = _wcfg(WCFG2, fp32, F, D, E)
        dw2 = torch.empty_like(w2)
        nn = triton.cdiv(D, w2c["BN"])
        _dw2_kernel[(triton.cdiv(F, w2c["BM"]) * nn, E)](
            h, dy, dw2, tok, wsort, seg, cnts, F, D, nn,
            dy.stride(0), w2.stride(0), w2.stride(1), w2.stride(2),
            SA=True, SB=sp, **w2c)
        ca = _gcfg(DACTCFG, fp32, BM, F)
        nft = triton.cdiv(F, ca["BN"])
        rp = torch.empty((nft, N), device=dev, dtype=torch.float32)
        _dact_kernel[(ntm * nft,)](
            dy, w2, h, rp, tok, wsort, tend, tstart, seg, cnts,
            N, F, D, E, nft, dy.stride(0),
            w2.stride(0), w2.stride(1), w2.stride(2),
            BM=BM, BE=be, SA=sp, SB=sp, **ca)
        rd = torch.empty((N,), device=dev, dtype=torch.float32)
        _rd_reduce_kernel[(triton.cdiv(N, 256),)](rp, rd, N, nft, BLK=256,
                                                  num_warps=4)
        dlogit = torch.empty((T, E), device=dev, dtype=torch.float32)
        bt = max(16, min(64, 8192 // be))
        _dweights_kernel[(triton.cdiv(T, bt),)](
            rd, pos, sel, w, ssel, ssum, dlogit, T, E, A, BT=bt, BE=be,
            num_warps=4)
        dxg = torch.empty((N, D), device=dev, dtype=torch.float32)
        cx = _gcfg(DXGCFG, fp32, BM, D)
        nn = triton.cdiv(D, cx["BN"])
        _dxg_kernel[(ntm * nn,)](
            h, w1, dxg, order, tend, tstart, seg, cnts, F2, D, E, nn,
            w1.stride(0), w1.stride(1), w1.stride(2),
            BM=BM, BE=be, SA=sp, SB=sp, **cx)
        w1c = _wcfg(WCFG1, fp32, D, F2, E)
        dw1 = torch.empty_like(w1)
        nn = triton.cdiv(F2, w1c["BN"])
        _dw1_kernel[(triton.cdiv(D, w1c["BM"]) * nn, E)](
            x, h, dw1, tok, seg, cnts, D, F2, nn,
            x.stride(0), w1.stride(0), w1.stride(1), w1.stride(2),
            SA=sp, SB=True, **w1c)
        bnr = min(be, 128)
        bmr = 64 if bnr >= 64 else 128
        nmr = triton.cdiv(D, bmr)
        ner = triton.cdiv(E, bnr)
        epad = ner * bnr
        sk = max(1, min(64, 1024 // max(nmr * ner, 1), triton.cdiv(T, 64)))
        part = torch.empty((sk, D * epad), device=dev, dtype=torch.float32)
        _drw_kernel[(nmr, sk, ner)](
            x, dlogit, part, T, D, E, sk, epad, x.stride(0),
            BM=bmr, BN=bnr, BK=32, SA=sp, SB=True, num_warps=4, num_stages=3)
        drw = torch.empty((D, E), device=dev, dtype=router_w.dtype)
        _drw_reduce_kernel[(triton.cdiv(D * E, 256),)](
            part, drw, D, E, sk, epad, BLK=256, num_warps=4)
        dx = torch.empty((T, D), device=dev, dtype=x.dtype)
        bd = _p2(min(D, 128), 16)
        ber = min(be, 128)
        bt = max(16, min(4096 // bd, 8192 // ber))
        nd = triton.cdiv(D, bd)
        _dx_kernel[(triton.cdiv(T, bt) * nd,)](
            dxg, dlogit, router_w, dx, T, D, E, A, nd,
            router_w.stride(1), router_w.stride(0),
            BT=bt, BD=bd, BE=ber, SA=sp, SB=sp, num_warps=4)
        return dx, drw, None, dw1, dw2, None


def fused_moe(
    x: torch.Tensor,         # [T, D] float32 or bfloat16
    router_w: torch.Tensor,  # [D, E] same dtype as x
    router_b: torch.Tensor,  # [E] float32, selection-only bias, no gradient
    w1: torch.Tensor,        # [E, D, 2F]  [:, :, :F] gate, [:, :, F:] up
    w2: torch.Tensor,        # [E, F, D]
    A: int,                  # experts per token
) -> torch.Tensor:           # [T, D] same dtype as x
    """Entry point. Signature and semantics are fixed; the body is yours."""
    return _FusedMoe.apply(x, router_w, router_b, w1, w2, A)
