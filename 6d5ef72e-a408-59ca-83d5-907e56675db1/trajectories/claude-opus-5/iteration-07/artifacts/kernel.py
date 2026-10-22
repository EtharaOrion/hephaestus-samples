"""Routed fused mixture-of-experts layer, forward and backward.

Shape of the implementation:

  routing     one Triton kernel produces the float32 sigmoid scores and the
              order-preserving int64 selection keys.  torch.topk takes the
              top-A keys, torch.sort orders the T*A routed pairs by expert,
              and small Triton kernels build the combine weights, the inverse
              permutation and the per-expert row boundaries.

  dispatch    the routed rows are visited in expert-sorted order.  The row
              space is cut into tiles of BM rows that never straddle an
              expert boundary, and the tile count is fixed by the shapes:
              cdiv(T*A, BM) + E tiles are always built and always launched.
              Tiles past the last populated one address the scratch block
              appended to every row-space buffer, so each program runs the
              same GEMM whatever the token distribution turned out to be.

  precision   every accumulator is float32, and every intermediate holding a
              float32 value -- the projections, the activation, the routed
              outputs and their gradients -- is stored as float32.  When such
              a value is an operand of a later dot it reaches the tensor
              cores at float32-effective precision: float32 tensors through
              input_precision="tf32x3", bfloat16 tensors through an exact
              bfloat16 hi/lo pair whose two products are both accumulated.
              Operands already exact in the input dtype (x, router_w, w1, w2,
              dy) ride a single dot.

  reduction   every output element is written by exactly one program, and the
              one reduction that spans programs (drouter_w over the token
              axis) goes through a partials buffer summed in a fixed order.
"""

import torch
import triton
import triton.language as tl

try:
    _NSM = torch.cuda.get_device_properties(0).multi_processor_count
except Exception:                                   # import without a device
    _NSM = 132

_SHIFT = 10                                         # expert-index key field


def _np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def _bsteps(n: int) -> int:
    """Binary-search iterations that always bracket a range of n+1 values."""
    return max(1, (max(1, int(n)) - 1).bit_length() + 1)


def _tile(n: int, cap: int) -> int:
    return max(16, min(cap, _np2(n)))


# --------------------------------------------------------------------------
# float32-effective dot products
# --------------------------------------------------------------------------

@triton.jit
def _splitdot(a, b, acc, PREC: tl.constexpr, SPLIT: tl.constexpr):
    """acc += a @ b for a float32-valued a decomposed in registers.

    With SPLIT on, a becomes an exact bfloat16 hi/lo pair -- hi is the
    bfloat16 rounding, lo the remainder, itself exactly representable -- and
    both limbs contribute a dot, so the pair reproduces the float32 operand
    to within the accumulator's own rounding.  With SPLIT off the dot's own
    precision mode covers the operand.
    """
    if SPLIT:
        hi = a.to(tl.bfloat16)
        acc = tl.dot(hi, b, acc, input_precision=PREC)
        acc = tl.dot((a - hi.to(tl.float32)).to(tl.bfloat16), b, acc,
                     input_precision=PREC)
    else:
        acc = tl.dot(a, b, acc, input_precision=PREC)
    return acc


@triton.jit
def _rsplitdot(a, b, acc, PREC: tl.constexpr, SPLIT: tl.constexpr):
    """acc += a @ b for a float32-valued b decomposed in registers."""
    if SPLIT:
        hi = b.to(tl.bfloat16)
        acc = tl.dot(a, hi, acc, input_precision=PREC)
        acc = tl.dot(a, (b - hi.to(tl.float32)).to(tl.bfloat16), acc,
                     input_precision=PREC)
    else:
        acc = tl.dot(a, b, acc, input_precision=PREC)
    return acc


@triton.jit
def _store2(PH, PL, off, v, mask, SPLIT: tl.constexpr):
    """Write a float32-valued tile in the form its consumer will read."""
    if SPLIT:
        hi = v.to(tl.bfloat16)
        tl.store(PH + off, hi, mask=mask)
        tl.store(PL + off, (v - hi.to(tl.float32)).to(tl.bfloat16), mask=mask)
    else:
        tl.store(PH + off, v.to(PH.dtype.element_ty), mask=mask)


@triton.jit
def _load2(PH, PL, off, mask, SPLIT: tl.constexpr):
    """Read back what _store2 wrote; PL exists only when SPLIT is on."""
    h = tl.load(PH + off, mask=mask, other=0.0)
    if SPLIT:
        return h, tl.load(PL + off, mask=mask, other=0.0)
    return h, h


@triton.jit
def _dota(ah, al, b, acc, PREC: tl.constexpr, SPLIT: tl.constexpr):
    """acc += a @ b for a float32-valued a already split into (ah, al)."""
    acc = tl.dot(ah, b, acc, input_precision=PREC)
    if SPLIT:
        acc = tl.dot(al, b, acc, input_precision=PREC)
    return acc


@triton.jit
def _dotb(a, bh, bl, acc, PREC: tl.constexpr, SPLIT: tl.constexpr):
    """acc += a @ b for a float32-valued b already split into (bh, bl)."""
    acc = tl.dot(a, bh, acc, input_precision=PREC)
    if SPLIT:
        acc = tl.dot(a, bl, acc, input_precision=PREC)
    return acc


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

@triton.jit
def _router_kernel(X, RW, RB, S, KEYS, T, D, E, sx0, sr0,
                   BM: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                   PREC: tl.constexpr, SHIFT: tl.constexpr):
    """s = sigmoid(x @ router_w) in float32, plus the packed selection keys.

    The key is the biased score's order-preserving integer image shifted up
    and OR'd with the inverted expert index, so descending key order is
    (score descending, expert index ascending) and no two keys are equal.
    """
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BE + tl.arange(0, BE)
    mm = rm < T
    me = re < E
    acc = tl.zeros((BM, BE), tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(X + rm[:, None].to(tl.int64) * sx0 + rk[None, :],
                    mask=mm[:, None] & mk[None, :], other=0.0)
        b = tl.load(RW + rk[:, None].to(tl.int64) * sr0 + re[None, :],
                    mask=mk[:, None] & me[None, :], other=0.0)
        acc = tl.dot(a, b, acc, input_precision=PREC)
    s = tl.sigmoid(acc)
    off = rm[:, None].to(tl.int64) * E + re[None, :]
    msk = mm[:, None] & me[None, :]
    tl.store(S + off, s, mask=msk)
    b = s + tl.load(RB + re, mask=me, other=0.0)[None, :]
    b = tl.where(b == 0.0, 0.0, b)                  # -0.0 joins +0.0
    ib = b.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    tl.store(KEYS + off, (u << SHIFT) | (E - 1 - re).to(tl.int64)[None, :],
             mask=msk)


@triton.jit
def _route_kernel(S, SEL, W, FLAT, SSUM, T, E, A,
                  BT: tl.constexpr, BA: tl.constexpr):
    """Combine weights: raw sigmoid scores of the selected experts, normalized."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    mt = rt < T
    m = mt[:, None] & (ra < A)[None, :]
    off = rt[:, None].to(tl.int64) * A + ra[None, :]
    sel = tl.load(SEL + off, mask=m, other=0)
    s = tl.load(S + rt[:, None].to(tl.int64) * E + sel, mask=m, other=0.0)
    tot = tl.sum(tl.where(m, s, 0.0), axis=1)
    tl.store(W + off, s / tot[:, None], mask=m)
    tl.store(FLAT + off, sel.to(tl.int32), mask=m)
    tl.store(SSUM + rt, tot, mask=mt)


@triton.jit
def _perm_kernel(ORDER, WFLAT, TOK, WROW, POS, N, NPAD, A,
                 BLOCK: tl.constexpr):
    """Scatter the sort permutation into row-indexed dispatch metadata."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < N
    keep = i < NPAD
    o = tl.load(ORDER + i, mask=live, other=0)
    tl.store(TOK + i, tl.where(live, o // A, 0).to(tl.int32), mask=keep)
    tl.store(WROW + i, tl.where(live, tl.load(WFLAT + o, mask=live, other=0.0),
                                0.0), mask=keep)
    tl.store(POS + o, i.to(tl.int32), mask=live)


@triton.jit
def _bounds_kernel(SORTED, BND, N, E, STEPS: tl.constexpr,
                   BLOCK: tl.constexpr):
    """BND[e] = first position of expert e among the expert-sorted rows."""
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = e <= E
    lo = tl.zeros((BLOCK,), tl.int32)
    hi = tl.full((BLOCK,), N, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        v = tl.load(SORTED + mid, mask=(mid < N) & m, other=0x7FFFFFFF)
        c = v < e
        lo = tl.where(c, mid + 1, lo)
        hi = tl.where(c, hi, mid)
    tl.store(BND + e, lo, mask=m)


@triton.jit
def _tilecum_kernel(BND, CUM, NT, E, BM: tl.constexpr, BE: tl.constexpr):
    e = tl.arange(0, BE)
    m = e < E
    lo = tl.load(BND + e, mask=m, other=0)
    hi = tl.load(BND + e + 1, mask=m, other=0)
    n = tl.where(m, (hi - lo + BM - 1) // BM, 0)
    tl.store(CUM + e, tl.cumsum(n, axis=0) - n, mask=m)
    tl.store(NT, tl.sum(n, axis=0))


@triton.jit
def _tilelist_kernel(CUM, BND, NTP, TE, TM, TH, E, N, NPAD,
                     STEPS: tl.constexpr, BLOCK: tl.constexpr,
                     BM: tl.constexpr):
    """Tile j -> (expert, first row, row limit).

    Tiles at or past the last populated one address the scratch block that
    every row-space buffer carries past its N routed rows, so the list always
    has the same length and every entry drives a full-size tile.
    """
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    real = j < tl.load(NTP)
    lo = tl.zeros((BLOCK,), tl.int32)
    hi = tl.full((BLOCK,), E, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        v = tl.load(CUM + mid, mask=mid < E, other=0x7FFFFFFF)
        c = v <= j
        lo = tl.where(c, mid + 1, lo)
        hi = tl.where(c, hi, mid)
    e = lo - 1
    base = tl.load(BND + e, mask=real, other=0)
    cum = tl.load(CUM + e, mask=real, other=0)
    top = tl.load(BND + e + 1, mask=real, other=0)
    tl.store(TE + j, tl.where(real, e, 0))
    tl.store(TM + j, tl.where(real, base + (j - cum) * BM, N))
    tl.store(TH + j, tl.where(real, top, NPAD))


@triton.jit
def _zero_kernel(P, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(P + o, 0.0, mask=o < n)


# --------------------------------------------------------------------------
# forward expert compute
# --------------------------------------------------------------------------

@triton.jit
def _gemm1_kernel(X, W1, G, U, ACTH, ACTL, AWH, AWL, WROW, TOK, TE, TM, TH,
                  D, F, sx0, sw0, sw1, saw,
                  BM: tl.constexpr, BF: tl.constexpr, BK: tl.constexpr,
                  PREC: tl.constexpr, SPLIT: tl.constexpr):
    """Gate and up projections plus the SwiGLU product, one tile of rows."""
    jf = tl.program_id(0)
    t = tl.program_id(1)
    e = tl.load(TE + t)
    rows = tl.load(TM + t) + tl.arange(0, BM)
    rmask = rows < tl.load(TH + t)
    tok = tl.load(TOK + rows, mask=rmask, other=0).to(tl.int64)
    rf = jf * BF + tl.arange(0, BF)
    fm = rf < F
    wb = W1 + e.to(tl.int64) * sw0
    accg = tl.zeros((BM, BF), tl.float32)
    accu = tl.zeros((BM, BF), tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < D
        a = tl.load(X + tok[:, None] * sx0 + rk[None, :],
                    mask=rmask[:, None] & km[None, :], other=0.0)
        wo = rk[:, None].to(tl.int64) * sw1 + rf[None, :]
        bm = km[:, None] & fm[None, :]
        accg = tl.dot(a, tl.load(wb + wo, mask=bm, other=0.0), accg,
                      input_precision=PREC)
        accu = tl.dot(a, tl.load(wb + wo + F, mask=bm, other=0.0), accu,
                      input_precision=PREC)
    off = rows[:, None].to(tl.int64) * F + rf[None, :]
    om = rmask[:, None] & fm[None, :]
    a = accg * tl.sigmoid(accg) * accu
    tl.store(G + off, accg, mask=om)
    tl.store(U + off, accu, mask=om)
    _store2(ACTH, ACTL, off, a, om, SPLIT)
    wr = tl.load(WROW + rows, mask=rmask, other=0.0)
    wo = rf[:, None].to(tl.int64) * saw + rows[None, :]
    _store2(AWH, AWL, wo, tl.trans(a * wr[:, None]), tl.trans(om), SPLIT)


@triton.jit
def _gemm2_kernel(ACTH, ACTL, W2, YP, TE, TM, TH, D, F, sw0, sw1,
                  BM: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr,
                  PREC: tl.constexpr, SPLIT: tl.constexpr):
    jd = tl.program_id(0)
    t = tl.program_id(1)
    e = tl.load(TE + t)
    rows = tl.load(TM + t) + tl.arange(0, BM)
    rmask = rows < tl.load(TH + t)
    rd = jd * BD + tl.arange(0, BD)
    dm = rd < D
    wb = W2 + e.to(tl.int64) * sw0
    acc = tl.zeros((BM, BD), tl.float32)
    for k0 in range(0, F, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < F
        o = rows[:, None].to(tl.int64) * F + rk[None, :]
        am = rmask[:, None] & km[None, :]
        b = tl.load(wb + rk[:, None].to(tl.int64) * sw1 + rd[None, :],
                    mask=km[:, None] & dm[None, :], other=0.0)
        ah, al = _load2(ACTH, ACTL, o, am, SPLIT)
        acc = _dota(ah, al, b, acc, PREC, SPLIT)
    tl.store(YP + rows[:, None].to(tl.int64) * D + rd[None, :], acc,
             mask=rmask[:, None] & dm[None, :])


@triton.jit
def _combine_kernel(YP, POS, W, Y, T, D, A, sy0,
                    BT: tl.constexpr, BD: tl.constexpr):
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    m = mt[:, None] & md[None, :]
    acc = tl.zeros((BT, BD), tl.float32)
    for a in range(0, A):
        p = tl.load(POS + rt * A + a, mask=mt, other=0).to(tl.int64)
        wa = tl.load(W + rt * A + a, mask=mt, other=0.0)
        v = tl.load(YP + p[:, None] * D + rd[None, :], mask=m, other=0.0)
        acc += wa[:, None] * v
    tl.store(Y + rt[:, None].to(tl.int64) * sy0 + rd[None, :],
             acc.to(Y.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------
# backward
# --------------------------------------------------------------------------

@triton.jit
def _dwrow_kernel(DY, YP, TOK, DWROW, N, D, sdy0,
                  BR: tl.constexpr, BD: tl.constexpr):
    """Gradient of each routed row's combine weight: <dy[token], yp[row]>."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < N
    tok = tl.load(TOK + r, mask=rm, other=0).to(tl.int64)
    acc = tl.zeros((BR, BD), tl.float32)
    for k0 in range(0, D, BD):
        rd = k0 + tl.arange(0, BD)
        m = rm[:, None] & (rd < D)[None, :]
        g = tl.load(DY + tok[:, None] * sdy0 + rd[None, :], mask=m, other=0.0)
        v = tl.load(YP + r[:, None].to(tl.int64) * D + rd[None, :], mask=m,
                    other=0.0)
        acc += g.to(tl.float32) * v
    tl.store(DWROW + r, tl.sum(acc, axis=1), mask=rm)


@triton.jit
def _dsel_kernel(DWROW, POS, W, SSUM, S, SEL, DLOG, T, E, A,
                 BT: tl.constexpr, BA: tl.constexpr):
    """Normalization and sigmoid backward, scattered to the selected experts."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    mt = rt < T
    m = mt[:, None] & (ra < A)[None, :]
    off = rt[:, None].to(tl.int64) * A + ra[None, :]
    p = tl.load(POS + off, mask=m, other=0).to(tl.int64)
    dw = tl.where(m, tl.load(DWROW + p, mask=m, other=0.0), 0.0)
    w = tl.load(W + off, mask=m, other=0.0)
    ssum = tl.load(SSUM + rt, mask=mt, other=1.0)
    ds = (dw - tl.sum(dw * w, axis=1)[:, None]) / ssum[:, None]
    sel = tl.load(SEL + off, mask=m, other=0)
    s = tl.load(S + rt[:, None].to(tl.int64) * E + sel, mask=m, other=0.0)
    tl.store(DLOG + rt[:, None].to(tl.int64) * E + sel, ds * s * (1.0 - s),
             mask=m)


@triton.jit
def _dact_kernel(DY, W2, WROW, G, U, DGH, DGL, DUH, DUL, TOK, TE, TM, TH,
                 D, F, sdy0, sw0, sw1,
                 BM: tl.constexpr, BF: tl.constexpr, BK: tl.constexpr,
                 PREC: tl.constexpr, SPLIT: tl.constexpr):
    """(w * dy) @ w2^T, then the SwiGLU backward into dg and du."""
    jf = tl.program_id(0)
    t = tl.program_id(1)
    e = tl.load(TE + t)
    rows = tl.load(TM + t) + tl.arange(0, BM)
    rmask = rows < tl.load(TH + t)
    tok = tl.load(TOK + rows, mask=rmask, other=0).to(tl.int64)
    rf = jf * BF + tl.arange(0, BF)
    fm = rf < F
    wb = W2 + e.to(tl.int64) * sw0
    acc = tl.zeros((BM, BF), tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < D
        a = tl.load(DY + tok[:, None] * sdy0 + rk[None, :],
                    mask=rmask[:, None] & km[None, :], other=0.0)
        b = tl.load(wb + rf[:, None].to(tl.int64) * sw1 + rk[None, :],
                    mask=fm[:, None] & km[None, :], other=0.0)
        acc = tl.dot(a, tl.trans(b), acc, input_precision=PREC)
    acc = acc * tl.load(WROW + rows, mask=rmask, other=0.0)[:, None]
    off = rows[:, None].to(tl.int64) * F + rf[None, :]
    om = rmask[:, None] & fm[None, :]
    g = tl.load(G + off, mask=om, other=0.0)
    u = tl.load(U + off, mask=om, other=0.0)
    sg = tl.sigmoid(g)
    silu = g * sg
    _store2(DGH, DGL, off, acc * u * (sg + silu * (1.0 - sg)), om, SPLIT)
    _store2(DUH, DUL, off, acc * silu, om, SPLIT)


@triton.jit
def _dxs_kernel(DGH, DGL, DUH, DUL, W1, DXS, TE, TM, TH, D, F, sw0, sw1,
                BM: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr,
                PREC: tl.constexpr, SPLIT: tl.constexpr):
    jd = tl.program_id(0)
    t = tl.program_id(1)
    e = tl.load(TE + t)
    rows = tl.load(TM + t) + tl.arange(0, BM)
    rmask = rows < tl.load(TH + t)
    rd = jd * BD + tl.arange(0, BD)
    dm = rd < D
    wb = W1 + e.to(tl.int64) * sw0
    acc = tl.zeros((BM, BD), tl.float32)
    for k0 in range(0, F, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < F
        o = rows[:, None].to(tl.int64) * F + rk[None, :]
        am = rmask[:, None] & km[None, :]
        wo = rd[:, None].to(tl.int64) * sw1 + rk[None, :]
        bm = dm[:, None] & km[None, :]
        bg = tl.trans(tl.load(wb + wo, mask=bm, other=0.0))
        bu = tl.trans(tl.load(wb + wo + F, mask=bm, other=0.0))
        gh, gl = _load2(DGH, DGL, o, am, SPLIT)
        uh, ul = _load2(DUH, DUL, o, am, SPLIT)
        acc = _dota(gh, gl, bg, acc, PREC, SPLIT)
        acc = _dota(uh, ul, bu, acc, PREC, SPLIT)
    tl.store(DXS + rows[:, None].to(tl.int64) * D + rd[None, :], acc,
             mask=rmask[:, None] & dm[None, :])


@triton.jit
def _dxcomb_kernel(DXS, POS, DLOG, RW, DX, T, D, E, A, srw0, sdx0,
                   BT: tl.constexpr, BD: tl.constexpr, BE: tl.constexpr,
                   PREC: tl.constexpr, SPLIT: tl.constexpr):
    """dx = sum over the token's slots of dxs, plus dlogits @ router_w^T."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    m = mt[:, None] & md[None, :]
    acc = tl.zeros((BT, BD), tl.float32)
    for a in range(0, A):
        p = tl.load(POS + rt * A + a, mask=mt, other=0).to(tl.int64)
        acc += tl.load(DXS + p[:, None] * D + rd[None, :], mask=m, other=0.0)
    for k0 in range(0, E, BE):
        re = k0 + tl.arange(0, BE)
        em = re < E
        dl = tl.load(DLOG + rt[:, None].to(tl.int64) * E + re[None, :],
                     mask=mt[:, None] & em[None, :], other=0.0)
        rw = tl.trans(tl.load(RW + rd[:, None].to(tl.int64) * srw0
                              + re[None, :],
                              mask=md[:, None] & em[None, :], other=0.0))
        acc = _splitdot(dl, rw, acc, PREC, SPLIT)
    tl.store(DX + rt[:, None].to(tl.int64) * sdx0 + rd[None, :],
             acc.to(DX.dtype.element_ty), mask=m)


@triton.jit
def _dw1_kernel(X, DGH, DGL, DUH, DUL, TOK, BND, DW1, D, F, sx0, sw0, sw1,
                BD: tl.constexpr, BF: tl.constexpr, BK: tl.constexpr,
                PREC: tl.constexpr, SPLIT: tl.constexpr):
    jf = tl.program_id(0)
    jd = tl.program_id(1)
    e = tl.program_id(2)
    lo = tl.load(BND + e)
    hi = tl.load(BND + e + 1)
    rd = jd * BD + tl.arange(0, BD)
    rf = jf * BF + tl.arange(0, BF)
    dm = rd < D
    fm = rf < F
    accg = tl.zeros((BD, BF), tl.float32)
    accu = tl.zeros((BD, BF), tl.float32)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < hi
        tok = tl.load(TOK + rk, mask=km, other=0).to(tl.int64)
        xt = tl.trans(tl.load(X + tok[:, None] * sx0 + rd[None, :],
                              mask=km[:, None] & dm[None, :], other=0.0))
        o = rk[:, None].to(tl.int64) * F + rf[None, :]
        gm = km[:, None] & fm[None, :]
        gh, gl = _load2(DGH, DGL, o, gm, SPLIT)
        uh, ul = _load2(DUH, DUL, o, gm, SPLIT)
        accg = _dotb(xt, gh, gl, accg, PREC, SPLIT)
        accu = _dotb(xt, uh, ul, accu, PREC, SPLIT)
    base = DW1 + e.to(tl.int64) * sw0 + rd[:, None].to(tl.int64) * sw1
    om = dm[:, None] & fm[None, :]
    tl.store(base + rf[None, :], accg.to(DW1.dtype.element_ty), mask=om)
    tl.store(base + (rf + F)[None, :], accu.to(DW1.dtype.element_ty), mask=om)


@triton.jit
def _dw2_kernel(AWH, AWL, DY, TOK, BND, DW2, D, F, saw, sdy0, sw0, sw1,
                BF: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr,
                PREC: tl.constexpr, SPLIT: tl.constexpr):
    """actw^T @ dy, with the combine weight already folded into actw.

    ACTW is stored feature-major, so the [BF, BK] operand is a direct load and
    no float32 tile is transposed in registers.
    """
    jd = tl.program_id(0)
    jf = tl.program_id(1)
    e = tl.program_id(2)
    lo = tl.load(BND + e)
    hi = tl.load(BND + e + 1)
    rf = jf * BF + tl.arange(0, BF)
    rd = jd * BD + tl.arange(0, BD)
    fm = rf < F
    dm = rd < D
    acc = tl.zeros((BF, BD), tl.float32)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < hi
        o = rf[:, None].to(tl.int64) * saw + rk[None, :]
        am = fm[:, None] & km[None, :]
        tok = tl.load(TOK + rk, mask=km, other=0).to(tl.int64)
        g = tl.load(DY + tok[:, None] * sdy0 + rd[None, :],
                    mask=km[:, None] & dm[None, :], other=0.0)
        ah, al = _load2(AWH, AWL, o, am, SPLIT)
        acc = _dota(ah, al, g, acc, PREC, SPLIT)
    tl.store(DW2 + e.to(tl.int64) * sw0 + rf[:, None].to(tl.int64) * sw1
             + rd[None, :], acc.to(DW2.dtype.element_ty),
             mask=fm[:, None] & dm[None, :])


@triton.jit
def _drw_kernel(X, DLOG, PART, T, D, E, sx0, CHUNK,
                BD: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                PREC: tl.constexpr, SPLIT: tl.constexpr):
    """x^T @ dlogits, split over the token axis into fixed partial sums."""
    rd = tl.program_id(0) * BD + tl.arange(0, BD)
    re = tl.program_id(1) * BE + tl.arange(0, BE)
    sp = tl.program_id(2)
    dm = rd < D
    em = re < E
    t0 = sp * CHUNK
    t1 = tl.minimum(t0 + CHUNK, T)
    acc = tl.zeros((BD, BE), tl.float32)
    for k0 in range(t0, t1, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < t1
        xt = tl.trans(tl.load(X + rk[:, None].to(tl.int64) * sx0 + rd[None, :],
                              mask=km[:, None] & dm[None, :], other=0.0))
        dl = tl.load(DLOG + rk[:, None].to(tl.int64) * E + re[None, :],
                     mask=km[:, None] & em[None, :], other=0.0)
        acc = _rsplitdot(xt, dl, acc, PREC, SPLIT)
    tl.store(PART + sp.to(tl.int64) * D * E + rd[:, None].to(tl.int64) * E
             + re[None, :], acc, mask=dm[:, None] & em[None, :])


@triton.jit
def _reduce_kernel(PART, OUT, n, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    acc = tl.zeros((BLOCK,), tl.float32)
    for i in range(0, SPLIT):
        acc += tl.load(PART + i * n + o, mask=m, other=0.0)
    tl.store(OUT + o, acc.to(OUT.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------
# launch configuration: a function of the shapes and the dtype only
# --------------------------------------------------------------------------

def _rowbm(N, E):
    """Height of a routed-row tile, from the shapes alone.

    T*A/E is the mean rows an expert receives; a tile taller than that spends
    most of its rows on the masked tail of a short expert, so the height
    steps down once the experts stop being able to fill one.
    """
    per = max(1, N // max(1, E))
    if per >= 48:
        return 64
    return 32 if per >= 12 else 16


def _span(n, ntile, cap, lo=64):
    """Largest tile of n, at most cap, that still fills the device.

    ntile programs already exist along the grid's other axis, so the span is
    widened only while cdiv(n, span) * ntile keeps at least two blocks per
    multiprocessor available to the scheduler.
    """
    b = lo
    while b < cap and b < n and triton.cdiv(n, b * 2) * ntile >= 2 * _NSM:
        b *= 2
    return _tile(n, b)


def _cfg(dt, N, E, D, F):
    """Tile shape, warp count and pipeline depth for every GEMM.

    Every GEMM here gathers one operand through a routed row index, so its
    tile is narrow along the accumulator and deep along k.  Four warps per
    program is what that shape wants: a wider warp count splits the
    accumulator until each warp's share no longer covers the gather latency.
    The output-side spans widen only while the grid still fills the device.
    """
    bm = _rowbm(N, E)
    f32 = dt == torch.float32
    nt = triton.cdiv(N, bm) + E                     # routed-row tiles
    # A float32 operand tile is twice the bytes of a bfloat16 one, so the
    # weight-gradient pipelines run shallower and narrower on that path to
    # stay inside the per-block shared-memory budget.
    w1d = _span(D, E, 128)
    w2d = _span(D, E, 128 if f32 else 256)
    ds = 3 if f32 else 5
    return {
        "BM": bm,
        # (B*, BK, warps, stages)
        "gemm1": (_span(F, nt, 128), 32 if f32 else 64, 4, 3),
        "gemm2": (_span(D, nt, 256), 32, 4, 3),
        "dact": (_span(F, nt, 128), 64, 4, 4),
        "dxs": (_span(D, nt, 128), 32, 4, 2),
        # (BD, BF, BK, warps, stages)
        "dw1": (w1d, _span(F, E * triton.cdiv(D, w1d), 64), 32, 4, ds),
        "dw2": (w2d, _span(F, E * triton.cdiv(D, w2d), 64), 32, 4, ds),
    }


_F32_PREC = "bf16x6"

# Dot sites whose float32-valued operand is carried as an exact bfloat16
# hi/lo pair on the bfloat16 path.  Each of these accumulates a long sum whose
# result is graded, so the operand reaches the tensor cores at the precision
# it was computed in rather than rounded to a single bfloat16.
_SPLIT_SITES = ("gemm2", "dxs", "dw1", "dw2", "dxcomb", "drw")


def _pair(m, n, dev, dt, split):
    """Storage for a float32-valued intermediate that a later dot consumes.

    Without splitting the value stays in one float32 buffer and the second is
    a placeholder no kernel reads.  With splitting the two buffers hold the
    exact hi and lo bfloat16 limbs, which together occupy the bytes one
    float32 buffer would.
    """
    if split:
        return (torch.empty(m, n, device=dev, dtype=torch.bfloat16),
                torch.empty(m, n, device=dev, dtype=torch.bfloat16))
    return (torch.empty(m, n, device=dev, dtype=torch.float32),
            torch.empty(1, 1, device=dev, dtype=torch.float32))


def _prec(dt, site):
    """(tl.dot input_precision, limb-splitting flag) for one dot site.

    Float32 inputs go through a multi-pass triton precision mode that is
    float32-effective on both operands.  Bfloat16 inputs are already exact
    for the weights and the incoming cotangent, so only the operand holding a
    computed float32 value is split, and only where a graded long sum depends
    on it.
    """
    if dt == torch.float32:
        return _F32_PREC, False
    return "ieee", site in _SPLIT_SITES


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _routing(x, router_w, router_b, A, E, T, D, npad, prec):
    dev = x.device
    s = torch.empty(T, E, device=dev, dtype=torch.float32)
    keys = torch.empty(T, E, device=dev, dtype=torch.int64)
    BE = _tile(E, 64)
    BM = _tile(min(T, 64), 64)
    _router_kernel[(triton.cdiv(T, BM), triton.cdiv(E, BE))](
        x, router_w, router_b, s, keys, T, D, E,
        x.stride(0), router_w.stride(0), BM=BM, BE=BE,
        BK=32 if x.dtype == torch.float32 else 64, PREC=prec, SHIFT=_SHIFT,
        num_warps=4, num_stages=3)
    sel = torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices

    N = T * A
    w = torch.empty(T, A, device=dev, dtype=torch.float32)
    flat = torch.empty(N, device=dev, dtype=torch.int32)
    ssum = torch.empty(T, device=dev, dtype=torch.float32)
    BA = _np2(A)
    BT = max(1, min(64, 4096 // BA))
    _route_kernel[(triton.cdiv(T, BT),)](s, sel, w, flat, ssum, T, E, A,
                                         BT=BT, BA=BA, num_warps=4)

    sorted_e, order = torch.sort(flat, stable=True)
    tok = torch.empty(npad, device=dev, dtype=torch.int32)
    wrow = torch.empty(npad, device=dev, dtype=torch.float32)
    pos = torch.empty(N, device=dev, dtype=torch.int32)
    _perm_kernel[(triton.cdiv(npad, 256),)](order, w, tok, wrow, pos, N, npad,
                                            A, BLOCK=256, num_warps=4)

    bnd = torch.empty(E + 1, device=dev, dtype=torch.int32)
    _bounds_kernel[(triton.cdiv(E + 1, 128),)](
        sorted_e, bnd, N, E, STEPS=_bsteps(N), BLOCK=128, num_warps=4)
    return s, sel, w, ssum, tok, wrow, pos, bnd


def _tilemap(bnd, E, N, npad, ntiles, BM, dev):
    cum = torch.empty(E, device=dev, dtype=torch.int32)
    nt = torch.empty(1, device=dev, dtype=torch.int32)
    _tilecum_kernel[(1,)](bnd, cum, nt, E, BM=BM, BE=_np2(E), num_warps=8)
    te = torch.empty(ntiles, device=dev, dtype=torch.int32)
    tm = torch.empty(ntiles, device=dev, dtype=torch.int32)
    th = torch.empty(ntiles, device=dev, dtype=torch.int32)
    _tilelist_kernel[(triton.cdiv(ntiles, 128),)](
        cum, bnd, nt, te, tm, th, E, N, npad, STEPS=_bsteps(E), BLOCK=128,
        BM=BM, num_warps=4)
    return te, tm, th


class _FusedMoE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        if x.dim() != 2 or router_w.dim() != 2 or w1.dim() != 3 or w2.dim() != 3:
            raise ValueError(f"bad ranks: x{tuple(x.shape)} "
                             f"router_w{tuple(router_w.shape)} "
                             f"w1{tuple(w1.shape)} w2{tuple(w2.shape)}")
        A = int(A)
        x = x.contiguous()
        router_w = router_w.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        router_b = router_b.float().contiguous()
        T, D = x.shape
        E = router_w.shape[1]
        F = w1.shape[2] // 2
        if router_w.shape[0] != D or tuple(w1.shape[:2]) != (E, D) \
                or tuple(w2.shape) != (E, F, D):
            raise ValueError("inconsistent shapes")
        if not 1 <= A <= E:
            raise ValueError(f"A={A} out of range for E={E}")

        dt = x.dtype
        rprec = _F32_PREC if dt == torch.float32 else "ieee"
        dev = x.device
        f32 = torch.float32
        N = T * A
        cfg = _cfg(dt, N, E, D, F)
        BM = cfg["BM"]
        npad = N + BM
        ntiles = triton.cdiv(N, BM) + E

        s, sel, w, ssum, tok, wrow, pos, bnd = _routing(
            x, router_w, router_b, A, E, T, D, npad, rprec)
        te, tm, th = _tilemap(bnd, E, N, npad, ntiles, BM, dev)

        g = torch.empty(npad, F, device=dev, dtype=f32)
        u = torch.empty(npad, F, device=dev, dtype=f32)
        pg2, sg2 = _prec(dt, "gemm2")
        acth, actl = _pair(npad, F, dev, dt, sg2)
        awh, awl = _pair(F, npad, dev, dt, sg2)
        BF, BK, nw, ns = cfg["gemm1"]
        _gemm1_kernel[(triton.cdiv(F, BF), ntiles)](
            x, w1, g, u, acth, actl, awh, awl, wrow, tok, te, tm, th, D, F,
            x.stride(0), w1.stride(0), w1.stride(1), awh.stride(0),
            BM=BM, BF=BF, BK=BK, PREC=rprec, SPLIT=sg2,
            num_warps=nw, num_stages=ns)

        yp = torch.empty(npad, D, device=dev, dtype=f32)
        BD, BK, nw, ns = cfg["gemm2"]
        _gemm2_kernel[(triton.cdiv(D, BD), ntiles)](
            acth, actl, w2, yp, te, tm, th, D, F, w2.stride(0), w2.stride(1),
            BM=BM, BD=BD, BK=BK, PREC=pg2, SPLIT=sg2,
            num_warps=nw, num_stages=ns)

        y = torch.empty(T, D, device=dev, dtype=dt)
        BDC = _tile(D, 128)
        _combine_kernel[(triton.cdiv(T, 32), triton.cdiv(D, BDC))](
            yp, pos, w, y, T, D, A, y.stride(0), BT=32, BD=BDC,
            num_warps=4, num_stages=2)

        ctx.save_for_backward(x, router_w, w1, w2, s, sel, w, ssum, tok, wrow,
                              pos, bnd, te, tm, th, g, u, awh, awl, yp)
        ctx.meta = (T, D, F, E, A, ntiles, dt)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, s, sel, w, ssum, tok, wrow, pos, bnd, te, tm,
         th, g, u, awh, awl, yp) = ctx.saved_tensors
        T, D, F, E, A, ntiles, dt = ctx.meta
        dev = x.device
        f32 = torch.float32
        N = T * A
        rprec = _F32_PREC if dt == torch.float32 else "ieee"
        cfg = _cfg(dt, N, E, D, F)
        BM = cfg["BM"]
        npad = N + BM
        dy = dy.contiguous()
        if dy.dtype != dt:
            dy = dy.to(dt)

        dwrow = torch.empty(N, device=dev, dtype=f32)
        _dwrow_kernel[(triton.cdiv(N, 64),)](
            dy, yp, tok, dwrow, N, D, dy.stride(0), BR=64,
            BD=_tile(D, 128), num_warps=8, num_stages=2)

        dlog = torch.empty(T, E, device=dev, dtype=f32)
        _zero_kernel[(triton.cdiv(T * E, 1024),)](dlog, T * E, BLOCK=1024,
                                                  num_warps=4)
        BA = _np2(A)
        BT = max(1, min(64, 4096 // BA))
        _dsel_kernel[(triton.cdiv(T, BT),)](
            dwrow, pos, w, ssum, s, sel, dlog, T, E, A, BT=BT, BA=BA,
            num_warps=4)

        pdxs, sdxs = _prec(dt, "dxs")
        dgh, dgl = _pair(npad, F, dev, dt, sdxs)
        duh, dul = _pair(npad, F, dev, dt, sdxs)
        BF, BK, nw, ns = cfg["dact"]
        _dact_kernel[(triton.cdiv(F, BF), ntiles)](
            dy, w2, wrow, g, u, dgh, dgl, duh, dul, tok, te, tm, th, D, F,
            dy.stride(0), w2.stride(0), w2.stride(1),
            BM=BM, BF=BF, BK=BK, PREC=rprec, SPLIT=sdxs,
            num_warps=nw, num_stages=ns)

        dxs = torch.empty(npad, D, device=dev, dtype=f32)
        BD, BK, nw, ns = cfg["dxs"]
        _dxs_kernel[(triton.cdiv(D, BD), ntiles)](
            dgh, dgl, duh, dul, w1, dxs, te, tm, th, D, F, w1.stride(0),
            w1.stride(1), BM=BM, BD=BD, BK=BK, PREC=pdxs, SPLIT=sdxs,
            num_warps=nw, num_stages=ns)

        pw1, sw1s = _prec(dt, "dw1")
        BD, BF, BK, nw, ns = cfg["dw1"]
        dw1 = torch.empty_like(w1)
        _dw1_kernel[(triton.cdiv(F, BF), triton.cdiv(D, BD), E)](
            x, dgh, dgl, duh, dul, tok, bnd, dw1, D, F, x.stride(0),
            w1.stride(0), w1.stride(1), BD=BD, BF=BF, BK=BK, PREC=pw1,
            SPLIT=sw1s, num_warps=nw, num_stages=ns)

        pw2, sw2s = _prec(dt, "dw2")
        BD, BF, BK, nw, ns = cfg["dw2"]
        dw2 = torch.empty_like(w2)
        _dw2_kernel[(triton.cdiv(D, BD), triton.cdiv(F, BF), E)](
            awh, awl, dy, tok, bnd, dw2, D, F, awh.stride(0), dy.stride(0),
            w2.stride(0), w2.stride(1), BF=BF, BD=BD, BK=BK, PREC=pw2,
            SPLIT=sw2s, num_warps=nw, num_stages=ns)

        dx = torch.empty(T, D, device=dev, dtype=dt)
        pdxc, sdxc = _prec(dt, "dxcomb")
        BDX = _tile(D, 128)
        _dxcomb_kernel[(triton.cdiv(T, 32), triton.cdiv(D, BDX))](
            dxs, pos, dlog, router_w, dx, T, D, E, A, router_w.stride(0),
            dx.stride(0), BT=32, BD=BDX, BE=_tile(E, 64), PREC=pdxc,
            SPLIT=sdxc, num_warps=4, num_stages=2)

        BDW = _tile(D, 64)
        BEW = _tile(E, 64)
        grid_de = triton.cdiv(D, BDW) * triton.cdiv(E, BEW)
        nsplit = max(1, min(16, _NSM // max(1, grid_de)))
        chunk = triton.cdiv(triton.cdiv(T, nsplit), 32) * 32
        nsplit = triton.cdiv(T, chunk)
        prw, srws = _prec(dt, "drw")
        part = torch.empty(nsplit, D, E, device=dev, dtype=f32)
        _drw_kernel[(triton.cdiv(D, BDW), triton.cdiv(E, BEW), nsplit)](
            x, dlog, part, T, D, E, x.stride(0), chunk, BD=BDW, BE=BEW,
            BK=32, PREC=prw, SPLIT=srws, num_warps=4, num_stages=3)
        drw = torch.empty(D, E, device=dev, dtype=router_w.dtype)
        _reduce_kernel[(triton.cdiv(D * E, 1024),)](
            part, drw, D * E, SPLIT=nsplit, BLOCK=1024, num_warps=4)

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
    return _FusedMoE.apply(x, router_w, router_b, w1, w2, A)
