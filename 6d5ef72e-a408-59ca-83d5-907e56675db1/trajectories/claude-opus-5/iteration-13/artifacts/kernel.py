"""Routed fused mixture-of-experts layer, forward and backward.

Pipeline per call:

  router GEMM -> [T, E] float32 sigmoid scores
  selection   -> per token, the A largest (score + bias, -index) keys
  dispatch    -> routes (token, expert) ordered by expert, padded to tiles
  expert GEMMs-> gate/up projection + SwiGLU, then the down projection
  combine     -> weighted sum of the A expert outputs per token

The backward re-uses the saved activations and the same expert-ordered
dispatch, and adds the router score path.

Numerics: router scores use exact products and float32 accumulation.  Every
float32-valued intermediate that a later GEMM contracts is stored as an
exact high/low bfloat16 limb pair, and each consumer contracts as many
limbs as its own error budget needs -- two or three for the weight
gradients, one where the value only reaches dx.
"""

import torch
import triton
import triton.language as tl

# Number of bfloat16 tensor-core passes used to contract one operand pair.
# A float32 value is carried as an exact high/low bfloat16 pair, so LIM=2
# reproduces roughly 16 mantissa bits and LIM=3 (dropping the low*low term)
# holds both operands to that width.  LIM=1 keeps only the high limb.
#
#   LIM 1: dot(ah, b)                                b exactly bfloat16
#   LIM 2: dot(ah, b) + dot(al, b)                   b exactly bfloat16
#   LIM 3: dot(ah, bh) + dot(ah, bl) + dot(al, bh)   b float32, split here
_L1 = tl.constexpr(1)
_L2 = tl.constexpr(2)


@triton.jit
def _split_hi(v):
    return v.to(tl.bfloat16)


@triton.jit
def _split_lo(v, hi):
    return (v - hi.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _dotl(ah, al, b, LIM: tl.constexpr):
    """Contract an already-split operand pair against b."""
    if LIM == _L1:
        return tl.dot(ah, b)
    elif LIM == _L2:
        return tl.dot(ah, b) + tl.dot(al, b)
    else:
        bh = _split_hi(b)
        bl = _split_lo(b, bh)
        return tl.dot(ah, bh) + tl.dot(ah, bl) + tl.dot(al, bh)


@triton.jit
def _dotr(a, b, LIM: tl.constexpr):
    """Contract two operands held in their input dtype."""
    if LIM == _L1:
        return tl.dot(a, b)
    else:
        ah = _split_hi(a)
        al = _split_lo(a, ah)
        bh = _split_hi(b)
        bl = _split_lo(b, bh)
        return tl.dot(ah, bh) + tl.dot(ah, bl) + tl.dot(al, bh)


@triton.jit
def _dotlb(a, bh, bl, LIM: tl.constexpr):
    """Contract a against an already-split right operand."""
    if LIM == _L2:
        return tl.dot(a, bh) + tl.dot(a, bl)
    else:
        ah = _split_hi(a)
        al = _split_lo(a, ah)
        return tl.dot(ah, bh) + tl.dot(al, bh) + tl.dot(ah, bl)


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

@triton.jit
def _router_kernel(X, RW, S, T, D, E, sx0, sr0,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   IEEE: tl.constexpr):
    """S = sigmoid(X @ RW) in float32."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mm = rm < T
    mn = rn < E
    acc = tl.zeros((BM, BN), tl.float32)
    xp = X + rm.to(tl.int64)[:, None] * sx0
    wp = RW + rn[None, :]
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(xp + rk[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        b = tl.load(wp + rk[:, None] * sr0, mask=mk[:, None] & mn[None, :],
                    other=0.0)
        if IEEE:
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc += tl.dot(a, b)
    s = tl.sigmoid(acc)
    tl.store(S + rm.to(tl.int64)[:, None] * E + rn[None, :], s,
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _select_kernel(S, B, SEL, WGT, T, E, A,
                   EP: tl.constexpr, AP: tl.constexpr, BT: tl.constexpr,
                   SHIFT: tl.constexpr):
    """Top-A of (score + bias) per token, ties to the lower expert index.

    Each (score, expert) pair becomes one int64 key: the order preserving
    integer image of the biased float32 score shifted left by SHIFT bits,
    or'd with the inverted expert index in the low SHIFT bits.  SHIFT holds
    every expert index, so keys are unique and taking the A largest keys in
    turn yields (score descending, index ascending).  -0.0 is mapped onto
    +0.0 before the transform.
    """
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    re = tl.arange(0, EP)
    mt = rt < T
    me = re < E
    m = mt[:, None] & me[None, :]
    s = tl.load(S + rt.to(tl.int64)[:, None] * E + re[None, :], mask=m,
                other=0.0)
    bias = tl.load(B + re, mask=me, other=0.0)
    bs = s + bias[None, :]
    bs = tl.where(bs == 0.0, 0.0, bs)
    ib = bs.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    key = (u << SHIFT) | (E - 1 - re).to(tl.int64)[None, :]
    key = tl.where(m, key, -1)
    low = (1 << SHIFT) - 1
    ra = tl.arange(0, AP)
    tot = tl.zeros((BT,), tl.float32)
    esel = tl.zeros((BT, AP), tl.int32)
    wsel = tl.zeros((BT, AP), tl.float32)
    for a in range(A):
        best = tl.max(key, axis=1)
        e = E - 1 - (best & low).to(tl.int32)
        hit = re[None, :] == e[:, None]
        sv = tl.sum(tl.where(hit, s, 0.0), axis=1)
        tot += sv
        slot = ra[None, :] == a
        esel = tl.where(slot, e[:, None], esel)
        wsel = tl.where(slot, sv[:, None], wsel)
        key = tl.where(hit, -1, key)
    ms = mt[:, None] & (ra[None, :] < A)
    p = rt.to(tl.int64)[:, None] * A + ra[None, :]
    tl.store(SEL + p, esel, mask=ms)
    tl.store(WGT + p, wsel / tot[:, None], mask=ms)


@triton.jit
def _offs_kernel(SORTED, OFFS, N, E, EP: tl.constexpr, STEPS: tl.constexpr):
    """OFFS[e] = index of the first route whose expert is >= e."""
    re = tl.arange(0, EP)
    lo = tl.zeros((EP,), tl.int32)
    hi = tl.full((EP,), N, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        v = tl.load(SORTED + mid, mask=mid < N, other=E)
        c = v < re
        lo = tl.where(c, mid + 1, lo)
        hi = tl.where(c, hi, mid)
    tl.store(OFFS + re, lo, mask=re <= E)


@triton.jit
def _meta_kernel(ORDER, WGTF, RTOK, RWGT, RDST, N, A, BLK: tl.constexpr):
    """Expert-ordered route table: source token, weight, destination slot."""
    pid = tl.program_id(0)
    j = pid * BLK + tl.arange(0, BLK)
    m = j < N
    o = tl.load(ORDER + j, mask=m, other=0)
    w = tl.load(WGTF + o, mask=m, other=0.0)
    tl.store(RTOK + j, (o // A).to(tl.int32), mask=m)
    tl.store(RDST + j, o.to(tl.int32), mask=m)
    tl.store(RWGT + j, w, mask=m)


@triton.jit
def _blktab_kernel(OFFS, BEXP, BROW, E, MAXB,
                   EP: tl.constexpr, BM: tl.constexpr, BB: tl.constexpr):
    """Map each padded tile index to (expert, first route row)."""
    pid = tl.program_id(0)
    b = pid * BB + tl.arange(0, BB)
    re = tl.arange(0, EP)
    me = re < E
    o0 = tl.load(OFFS + re, mask=me, other=0)
    o1 = tl.load(OFFS + re + 1, mask=me, other=0)
    nb = tl.where(me, (o1 - o0 + BM - 1) // BM, 0)
    cnb = tl.cumsum(nb, axis=0)
    lt = cnb[None, :] <= b[:, None]
    idx = tl.sum(lt.to(tl.int32), axis=1)
    valid = (idx < E) & (b < MAXB)
    e = tl.where(valid, idx, 0)
    hit = re[None, :] == e[:, None]
    base = tl.sum(tl.where(hit, cnb[None, :] - nb[None, :], 0), axis=1)
    seg = tl.sum(tl.where(hit, o0[None, :], 0), axis=1)
    tl.store(BEXP + b, tl.where(valid, e, -1), mask=b < MAXB)
    tl.store(BROW + b, seg + (b - base) * BM, mask=b < MAXB)


# --------------------------------------------------------------------------
# forward expert compute
# --------------------------------------------------------------------------

@triton.jit
def _gemm1_kernel(X, W1, ACTH, ACTL, GU, RTOK, RWGT, BEXP, BROW, OFFS, D, F,
                  sx0, sw0, sw1, sa0, sg0,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  LIM: tl.constexpr):
    """act = silu(Xg @ W1[:, :F]) * (Xg @ W1[:, F:]) for one tile of routes.

    The stored activation carries the combine weight already, so the down
    projection and dW2 contract it directly.  GU keeps the raw gate and up
    projections for the SwiGLU backward.
    """
    pid = tl.program_id(0)
    e = tl.load(BEXP + pid)
    if e < 0:
        return
    row0 = tl.load(BROW + pid)
    end = tl.load(OFFS + e + 1)
    pid_n = tl.program_id(1)
    rm = row0 + tl.arange(0, BM)
    mm = rm < end
    tok = tl.load(RTOK + rm, mask=mm, other=0)
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < F
    xp = X + tok.to(tl.int64)[:, None] * sx0
    wg = W1 + e.to(tl.int64) * sw0 + rn[None, :]
    wu = wg + F
    accg = tl.zeros((BM, BN), tl.float32)
    accu = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(xp + rk[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        bg = tl.load(wg + rk[:, None] * sw1, mask=mk[:, None] & mn[None, :],
                     other=0.0)
        bu = tl.load(wu + rk[:, None] * sw1, mask=mk[:, None] & mn[None, :],
                     other=0.0)
        accg += _dotr(a, bg, LIM)
        accu += _dotr(a, bu, LIM)
    w = tl.load(RWGT + rm, mask=mm, other=0.0)
    act = (accg * tl.sigmoid(accg)) * accu * w[:, None]
    st = mm[:, None] & mn[None, :]
    gp = GU + rm.to(tl.int64)[:, None] * sg0 + rn[None, :]
    tl.store(gp, accg, mask=st)
    tl.store(gp + F, accu, mask=st)
    ap = rm.to(tl.int64)[:, None] * sa0 + rn[None, :]
    hi = _split_hi(act)
    tl.store(ACTH + ap, hi, mask=st)
    tl.store(ACTL + ap, _split_lo(act, hi), mask=st)


@triton.jit
def _gemm2_kernel(ACTH, ACTL, W2, YP, RDST, BEXP, BROW, OFFS, F, D,
                  sa0, sw0, sw1, sy0,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  LIM: tl.constexpr):
    """yp[dst] = act @ W2[e]; rows land in (token, slot) order."""
    pid = tl.program_id(0)
    e = tl.load(BEXP + pid)
    if e < 0:
        return
    row0 = tl.load(BROW + pid)
    end = tl.load(OFFS + e + 1)
    pid_n = tl.program_id(1)
    rm = row0 + tl.arange(0, BM)
    mm = rm < end
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < D
    ao = rm.to(tl.int64)[:, None] * sa0
    wp = W2 + e.to(tl.int64) * sw0 + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, F, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < F
        ma = mm[:, None] & mk[None, :]
        b = tl.load(wp + rk[:, None] * sw1, mask=mk[:, None] & mn[None, :],
                    other=0.0)
        ah = tl.load(ACTH + ao + rk[None, :], mask=ma, other=0.0)
        al = tl.load(ACTL + ao + rk[None, :], mask=ma, other=0.0)
        acc += _dotl(ah, al, b, LIM)
    dst = tl.load(RDST + rm, mask=mm, other=0)
    tl.store(YP + dst.to(tl.int64)[:, None] * sy0 + rn[None, :],
             acc, mask=mm[:, None] & mn[None, :])


@triton.jit
def _combine_kernel(YP, Y, T, D, A, sy0, sy1,
                    BT: tl.constexpr, BN: tl.constexpr):
    """y[t] = sum_a yp[t, a]; the combine weight is already in yp."""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    rt = pid_t * BT + tl.arange(0, BT)
    rn = pid_n * BN + tl.arange(0, BN)
    mt = rt < T
    mn = rn < D
    acc = tl.zeros((BT, BN), tl.float32)
    base = YP + (rt.to(tl.int64) * A)[:, None] * sy0 + rn[None, :]
    for a in range(A):
        acc += tl.load(base + a * sy0, mask=mt[:, None] & mn[None, :],
                       other=0.0)
    tl.store(Y + rt.to(tl.int64)[:, None] * sy1 + rn[None, :],
             acc.to(Y.dtype.element_ty), mask=mt[:, None] & mn[None, :])


# --------------------------------------------------------------------------
# backward
# --------------------------------------------------------------------------

@triton.jit
def _dact_kernel(DY, W2, GU, DH1H, DH1L, DWP, RTOK, RDST, RWGT,
                 BEXP, BROW, OFFS, D, F,
                 sdy0, sw0, sw1, sg0, sh0, sp0,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 LIM: tl.constexpr):
    """dact = wgt * (dy @ W2[e]^T), then the SwiGLU backward into DH1.

    The unweighted tile also carries the combine-weight gradient
    <dy, f> = <dy @ W2^T, act>, summed here over this column block.
    """
    pid = tl.program_id(0)
    e = tl.load(BEXP + pid)
    if e < 0:
        return
    row0 = tl.load(BROW + pid)
    end = tl.load(OFFS + e + 1)
    pid_n = tl.program_id(1)
    rm = row0 + tl.arange(0, BM)
    mm = rm < end
    tok = tl.load(RTOK + rm, mask=mm, other=0)
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < F
    dp = DY + tok.to(tl.int64)[:, None] * sdy0
    wp = W2 + e.to(tl.int64) * sw0 + rn[:, None] * sw1
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        a = tl.load(dp + rk[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        b = tl.load(wp + rk[None, :], mask=mn[:, None] & mk[None, :], other=0.0)
        acc += _dotr(a, tl.trans(b), LIM)
    st = mm[:, None] & mn[None, :]
    gp = GU + rm.to(tl.int64)[:, None] * sg0 + rn[None, :]
    g = tl.load(gp, mask=st, other=0.0)
    u = tl.load(gp + F, mask=st, other=0.0)
    sg = tl.sigmoid(g)
    silu = g * sg
    dst = tl.load(RDST + rm, mask=mm, other=0)
    part = tl.sum(acc * (silu * u), axis=1)
    tl.store(DWP + dst.to(tl.int64) * sp0 + pid_n, part, mask=mm)
    w = tl.load(RWGT + rm, mask=mm, other=0.0)
    dact = acc * w[:, None]
    dg = dact * u * (sg + silu * (1.0 - sg))
    du = dact * silu
    ho = rm.to(tl.int64)[:, None] * sh0 + rn[None, :]
    gh = _split_hi(dg)
    uh = _split_hi(du)
    tl.store(DH1H + ho, gh, mask=st)
    tl.store(DH1H + ho + F, uh, mask=st)
    tl.store(DH1L + ho, _split_lo(dg, gh), mask=st)
    tl.store(DH1L + ho + F, _split_lo(du, uh), mask=st)


@triton.jit
def _dscore_kernel(DWP, WGT, SEL, S, DLOG, T, E, A, sp0,
                   NPART: tl.constexpr, NP: tl.constexpr, EP: tl.constexpr,
                   BT: tl.constexpr):
    """Combine-weight gradients -> router logit gradients [T, E]."""
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    re = tl.arange(0, EP)
    rp = tl.arange(0, NP)
    mt = rt < T
    me = re < E
    dot = tl.zeros((BT,), tl.float32)
    tot = tl.zeros((BT,), tl.float32)
    acc = tl.zeros((BT, EP), tl.float32)
    chosen = tl.zeros((BT, EP), tl.int1)
    sp = S + rt.to(tl.int64)[:, None] * E + re[None, :]
    s = tl.load(sp, mask=mt[:, None] & me[None, :], other=0.0)
    for a in range(A):
        p = tl.load(DWP + ((rt * A + a).to(tl.int64) * sp0)[:, None]
                    + rp[None, :], mask=mt[:, None] & (rp[None, :] < NPART),
                    other=0.0)
        dw = tl.sum(p, axis=1)
        w = tl.load(WGT + rt * A + a, mask=mt, other=0.0)
        dot += dw * w
        e = tl.load(SEL + rt * A + a, mask=mt, other=0)
        hit = re[None, :] == e[:, None]
        acc += tl.where(hit, dw[:, None], 0.0)
        chosen |= hit
        tot += tl.sum(tl.where(hit, s, 0.0), axis=1)
    ds = tl.where(chosen, (acc - dot[:, None]) / tot[:, None], 0.0)
    tl.store(DLOG + rt.to(tl.int64)[:, None] * E + re[None, :],
             ds * s * (1.0 - s), mask=mt[:, None] & me[None, :])


@triton.jit
def _dw1_kernel(X, DH1H, DH1L, DW1, RTOK, OFFS, D, F2,
                sx0, sh0, sw0, sw1,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LIM: tl.constexpr):
    """dW1[e] = Xg[e]^T @ DH1[e]."""
    e = tl.program_id(2)
    lo = tl.load(OFFS + e)
    hi = tl.load(OFFS + e + 1)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < F2
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < hi
        tok = tl.load(RTOK + rk, mask=mk, other=0)
        a = tl.load(X + tok[None, :].to(tl.int64) * sx0 + rm[:, None],
                    mask=mm[:, None] & mk[None, :], other=0.0)
        ho = rk[:, None].to(tl.int64) * sh0 + rn[None, :]
        mb = mk[:, None] & mn[None, :]
        bh = tl.load(DH1H + ho, mask=mb, other=0.0)
        bl = tl.load(DH1L + ho, mask=mb, other=0.0)
        acc += _dotlb(a, bh, bl, LIM)
    tl.store(DW1 + e.to(tl.int64) * sw0 + rm[:, None].to(tl.int64) * sw1
             + rn[None, :], acc.to(DW1.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _dw2_kernel(ACTH, ACTL, DY, DW2, RTOK, OFFS, F, D,
                sa0, sdy0, sw0, sw1,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LIM: tl.constexpr):
    """dW2[e] = act[e]^T @ DYg[e]; act already carries the combine weight."""
    e = tl.program_id(2)
    lo = tl.load(OFFS + e)
    hi = tl.load(OFFS + e + 1)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mm = rm < F
    mn = rn < D
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < hi
        tok = tl.load(RTOK + rk, mask=mk, other=0)
        b = tl.load(DY + tok[:, None].to(tl.int64) * sdy0 + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        ao = rk[None, :].to(tl.int64) * sa0 + rm[:, None]
        ma = mm[:, None] & mk[None, :]
        ah = tl.load(ACTH + ao, mask=ma, other=0.0)
        al = tl.load(ACTL + ao, mask=ma, other=0.0)
        acc += _dotl(ah, al, b, LIM)
    tl.store(DW2 + e.to(tl.int64) * sw0 + rm[:, None].to(tl.int64) * sw1
             + rn[None, :], acc.to(DW2.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _dxs_kernel(DH1H, DH1L, W1, DXS, RDST, BEXP, BROW, OFFS, F2, D,
                sh0, sw0, sw1, sd0,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                LIM: tl.constexpr):
    """dxs[dst] = dh1 @ W1[e]^T, one row per route."""
    pid = tl.program_id(0)
    e = tl.load(BEXP + pid)
    if e < 0:
        return
    row0 = tl.load(BROW + pid)
    end = tl.load(OFFS + e + 1)
    pid_n = tl.program_id(1)
    rm = row0 + tl.arange(0, BM)
    mm = rm < end
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < D
    hp = rm.to(tl.int64)[:, None] * sh0
    wp = W1 + e.to(tl.int64) * sw0 + rn[:, None] * sw1
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, F2, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < F2
        ma = mm[:, None] & mk[None, :]
        ah = tl.load(DH1H + hp + rk[None, :], mask=ma, other=0.0)
        b = tl.load(wp + rk[None, :], mask=mn[:, None] & mk[None, :], other=0.0)
        if LIM == _L1:
            acc += tl.dot(ah, tl.trans(b))
        else:
            al = tl.load(DH1L + hp + rk[None, :], mask=ma, other=0.0)
            acc += _dotl(ah, al, tl.trans(b), LIM)
    dst = tl.load(RDST + rm, mask=mm, other=0)
    tl.store(DXS + dst.to(tl.int64)[:, None] * sd0 + rn[None, :],
             acc.to(DXS.dtype.element_ty), mask=mm[:, None] & mn[None, :])


@triton.jit
def _dxcomb_kernel(DXS, DLOG, RW, DX, T, D, E, A, sd0, sr0, sx0,
                   BT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   LIM: tl.constexpr):
    """dx[t] = sum_a dxs[t, a] + dlogits[t] @ RW^T."""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    rt = pid_t * BT + tl.arange(0, BT)
    rn = pid_n * BN + tl.arange(0, BN)
    mt = rt < T
    mn = rn < D
    acc = tl.zeros((BT, BN), tl.float32)
    base = DXS + (rt.to(tl.int64) * A)[:, None] * sd0 + rn[None, :]
    for a in range(A):
        v = tl.load(base + a * sd0, mask=mt[:, None] & mn[None, :], other=0.0)
        acc += v.to(tl.float32)
    dp = DLOG + rt.to(tl.int64)[:, None] * E
    wp = RW + rn[None, :] * sr0
    for k0 in range(0, E, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < E
        dl = tl.load(dp + rk[None, :], mask=mt[:, None] & mk[None, :], other=0.0)
        b = tl.load(wp + rk[:, None], mask=mk[:, None] & mn[None, :], other=0.0)
        acc += _dotr(dl, b, LIM)
    tl.store(DX + rt.to(tl.int64)[:, None] * sx0 + rn[None, :],
             acc.to(DX.dtype.element_ty), mask=mt[:, None] & mn[None, :])


@triton.jit
def _drw_kernel(X, DLOG, OUT, T, D, E, sx0, so0, so1,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                SPLIT: tl.constexpr, LIM: tl.constexpr):
    """Partial dRW = X^T @ DLOG over a fixed slice of tokens."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    sp = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < E
    chunk = tl.cdiv(tl.cdiv(T, SPLIT), BK) * BK
    k_lo = sp * chunk
    k_hi = tl.minimum(k_lo + chunk, T)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(k_lo, k_hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < k_hi
        a = tl.load(X + rk[None, :].to(tl.int64) * sx0 + rm[:, None],
                    mask=mm[:, None] & mk[None, :], other=0.0)
        b = tl.load(DLOG + rk[:, None].to(tl.int64) * E + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        acc += _dotr(a, b, LIM)
    tl.store(OUT + sp.to(tl.int64) * so0 + rm[:, None].to(tl.int64) * so1
             + rn[None, :], acc, mask=mm[:, None] & mn[None, :])


@triton.jit
def _drw_reduce_kernel(PART, DRW, D, E, so0, so1, sr0, SPLIT: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    m = (rm[:, None] < D) & (rn[None, :] < E)
    acc = tl.zeros((BM, BN), tl.float32)
    p = PART + rm[:, None].to(tl.int64) * so1 + rn[None, :]
    for i in range(SPLIT):
        acc += tl.load(p + i.to(tl.int64) * so0, mask=m, other=0.0)
    tl.store(DRW + rm[:, None].to(tl.int64) * sr0 + rn[None, :],
             acc.to(DRW.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------

# (BM, BN, BK, num_warps, num_stages) per dtype, keyed by "input is bfloat16".
# The weight-gradient GEMMs contract over an expert's routes, so their BM/BN
# tile the weight itself and no dispatch padding is involved.
_CFG_DW1 = {True: (128, 64, 64, 4, 2), False: (64, 128, 32, 4, 3)}
_CFG_DW2 = {True: (64, 128, 64, 4, 2), False: (128, 64, 32, 4, 2)}
# Route-tiled kernels: BM here also sets how far each expert segment is padded.
_CFG_G1 = {True: (128, 128, 64, 8, 4), False: (128, 128, 32, 8, 3)}
_CFG_G2 = {True: (32, 256, 64, 4, 3), False: (128, 128, 64, 8, 3)}
_CFG_DA = {True: (64, 128, 64, 8, 4), False: (128, 128, 64, 8, 3)}
_CFG_DX = {True: (128, 256, 64, 8, 4), False: (128, 64, 64, 8, 2)}

# Limb passes per contraction.
#   G1, DA: both operands are the layer's own inputs, so for bfloat16 one
#           pass already multiplies exactly and accumulates in float32.
#   G2, W:  one side is a float32-valued intermediate or gradient, contracted
#           as two limbs (bfloat16 inputs) or three (float32 inputs).
#   DX:     reaches only dx, so the high limb alone stays inside its budget.
_LIM_G1 = {True: 1, False: 3}
_LIM_G2 = {True: 2, False: 3}
_LIM_DA = {True: 1, False: 3}
_LIM_DX = {True: 1, False: 3}
_LIM_W = {True: 2, False: 3}
# drw and dxcomb contract a float32 [T, E] gradient against an input-dtype
# matrix, so both sides are split whatever the input dtype is.
_LIM_R = 3


def _pow2_fit(budget, unit, cap):
    """Largest power of two <= cap whose product with unit fits budget."""
    n = 1
    while n * 2 <= cap and n * 2 * unit <= budget:
        n *= 2
    return n


def _route_bm(pref, n_avg):
    """Clamp a preferred route tile so padding cannot dominate a small expert."""
    cap = 32
    while cap < n_avg and cap < pref:
        cap *= 2
    return min(pref, cap)


def _routing(x, router_w, router_b, A):
    T, D = x.shape
    E = router_w.shape[1]
    dev = x.device
    bf16 = x.dtype == torch.bfloat16
    s = torch.empty(T, E, device=dev, dtype=torch.float32)
    BN = max(16, min(128, triton.next_power_of_2(E)))
    BM = 64 if BN >= 128 else 128
    _router_kernel[(triton.cdiv(T, BM), triton.cdiv(E, BN))](
        x, router_w, s, T, D, E, x.stride(0), router_w.stride(0),
        BM=BM, BN=BN, BK=64 if bf16 else 32, IEEE=not bf16,
        num_warps=8, num_stages=3)

    sel = torch.empty(T, A, device=dev, dtype=torch.int32)
    wgt = torch.empty(T, A, device=dev, dtype=torch.float32)
    EP = max(16, triton.next_power_of_2(E))
    # SHIFT spans every expert index, so the packed key never aliases the
    # score bits whatever E is; 10 matches the reference packing for E <= 1024.
    shift = max(10, max(1, (E - 1).bit_length()))
    AP = max(16, triton.next_power_of_2(A))
    # one [BT, EP] key tile plus an [BT, AP] slot tile must stay in registers
    BT = _pow2_fit(8192, EP + AP, 32)
    _select_kernel[(triton.cdiv(T, BT),)](
        s, router_b, sel, wgt, T, E, A, EP=EP, AP=AP, BT=BT, SHIFT=shift,
        num_warps=4, num_stages=1)

    N = T * A
    sorted_e, order = torch.sort(sel.view(-1), stable=True)
    offs = torch.empty(E + 1, device=dev, dtype=torch.int32)
    _offs_kernel[(1,)](sorted_e, offs, N, E,
                       EP=max(16, triton.next_power_of_2(E + 1)),
                       STEPS=int(N).bit_length() + 1, num_warps=1)
    rtok = torch.empty(N, device=dev, dtype=torch.int32)
    rdst = torch.empty(N, device=dev, dtype=torch.int32)
    rwgt = torch.empty(N, device=dev, dtype=torch.float32)
    _meta_kernel[(triton.cdiv(N, 256),)](order, wgt, rtok, rwgt, rdst, N, A,
                                         BLK=256, num_warps=4)
    return s, sel, wgt, offs, rtok, rwgt, rdst


def _blk_table(offs, E, N, BM):
    maxb = triton.cdiv(N, BM) + E
    bexp = torch.empty(maxb, device=offs.device, dtype=torch.int32)
    brow = torch.empty(maxb, device=offs.device, dtype=torch.int32)
    EP = max(16, triton.next_power_of_2(E))
    BB = _pow2_fit(8192, EP, 64)
    _blktab_kernel[(triton.cdiv(maxb, BB),)](
        offs, bexp, brow, E, maxb, EP=EP, BM=BM, BB=BB, num_warps=4)
    return bexp, brow, maxb


class _Tables:
    """Dispatch tile tables, one per distinct route tile size in this call."""

    def __init__(self, offs, E, N):
        self.offs, self.E, self.N, self.made = offs, E, N, {}

    def get(self, BM):
        if BM not in self.made:
            self.made[BM] = _blk_table(self.offs, self.E, self.N, BM)
        return self.made[BM]


class _FusedMoe(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        x = x.contiguous()
        router_w = router_w.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        T, D = x.shape
        E = router_w.shape[1]
        F = w1.shape[2] // 2
        N = T * A
        bf16 = x.dtype == torch.bfloat16
        dev = x.device

        s, sel, wgt, offs, rtok, rwgt, rdst = _routing(x, router_w,
                                                       router_b.float(), A)
        tabs = _Tables(offs, E, N)
        navg = N / E

        act = torch.empty(N, F, device=dev, dtype=torch.bfloat16)
        actl = torch.empty(N, F, device=dev, dtype=torch.bfloat16)
        gu = torch.empty(N, 2 * F, device=dev, dtype=torch.float32)
        pm, BN1, BK1, nw, ns = _CFG_G1[bf16]
        BM1 = _route_bm(pm, navg)
        bexp, brow, maxb = tabs.get(BM1)
        _gemm1_kernel[(maxb, triton.cdiv(F, BN1))](
            x, w1, act, actl, gu, rtok, rwgt, bexp, brow, offs, D, F,
            x.stride(0), w1.stride(0), w1.stride(1), act.stride(0), gu.stride(0),
            BM=BM1, BN=BN1, BK=BK1, LIM=_LIM_G1[bf16],
            num_warps=nw, num_stages=ns)

        yp = torch.empty(N, D, device=dev, dtype=torch.float32)
        pm, BN2, BK2, nw, ns = _CFG_G2[bf16]
        BM2 = _route_bm(pm, navg)
        bexp, brow, maxb = tabs.get(BM2)
        _gemm2_kernel[(maxb, triton.cdiv(D, BN2))](
            act, actl, w2, yp, rdst, bexp, brow, offs, F, D,
            act.stride(0), w2.stride(0), w2.stride(1), yp.stride(0),
            BM=BM2, BN=BN2, BK=BK2, LIM=_LIM_G2[bf16],
            num_warps=nw, num_stages=ns)

        y = torch.empty(T, D, device=dev, dtype=x.dtype)
        _combine_kernel[(triton.cdiv(T, 32), triton.cdiv(D, 128))](
            yp, y, T, D, A, yp.stride(0), y.stride(0),
            BT=32, BN=128, num_warps=4, num_stages=2)

        ctx.save_for_backward(x, router_w, w1, w2, s, sel, wgt, offs,
                              rtok, rwgt, rdst, act, actl, gu)
        ctx.meta = (T, D, F, E, A, N, bf16)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, s, sel, wgt, offs, rtok, rwgt, rdst,
         act, actl, gu) = ctx.saved_tensors
        T, D, F, E, A, N, bf16 = ctx.meta
        dev = x.device
        dy = dy.contiguous()
        tabs = _Tables(offs, E, N)
        navg = N / E

        dh1 = torch.empty(N, 2 * F, device=dev, dtype=torch.bfloat16)
        dh1l = torch.empty(N, 2 * F, device=dev, dtype=torch.bfloat16)
        pm, BNa, BKa, nw, ns = _CFG_DA[bf16]
        BMa = _route_bm(pm, navg)
        bexp, brow, maxb = tabs.get(BMa)
        npart = triton.cdiv(F, BNa)
        NP = triton.next_power_of_2(npart)
        dwp = torch.empty(N, NP, device=dev, dtype=torch.float32)
        _dact_kernel[(maxb, npart)](
            dy, w2, gu, dh1, dh1l, dwp, rtok, rdst, rwgt, bexp, brow, offs, D, F,
            dy.stride(0), w2.stride(0), w2.stride(1), gu.stride(0),
            dh1.stride(0), dwp.stride(0),
            BM=BMa, BN=BNa, BK=BKa, LIM=_LIM_DA[bf16],
            num_warps=nw, num_stages=ns)

        dlog = torch.empty(T, E, device=dev, dtype=torch.float32)
        EP = max(16, triton.next_power_of_2(E))
        BTs = _pow2_fit(8192, EP + NP, 32)
        _dscore_kernel[(triton.cdiv(T, BTs),)](
            dwp, wgt, sel, s, dlog, T, E, A, dwp.stride(0),
            NPART=npart, NP=NP, EP=EP, BT=BTs, num_warps=4, num_stages=1)

        dw1 = torch.empty_like(w1)
        m1, n1, k1, w1w, w1s = _CFG_DW1[bf16]
        _dw1_kernel[(triton.cdiv(D, m1), triton.cdiv(2 * F, n1), E)](
            x, dh1, dh1l, dw1, rtok, offs, D, 2 * F,
            x.stride(0), dh1.stride(0), dw1.stride(0), dw1.stride(1),
            BM=m1, BN=n1, BK=k1, LIM=_LIM_W[bf16], num_warps=w1w,
            num_stages=w1s)

        dw2 = torch.empty_like(w2)
        m2, n2, k2, w2w, w2s = _CFG_DW2[bf16]
        _dw2_kernel[(triton.cdiv(F, m2), triton.cdiv(D, n2), E)](
            act, actl, dy, dw2, rtok, offs, F, D,
            act.stride(0), dy.stride(0), dw2.stride(0), dw2.stride(1),
            BM=m2, BN=n2, BK=k2, LIM=_LIM_W[bf16], num_warps=w2w,
            num_stages=w2s)

        dxs = torch.empty(N, D, device=dev, dtype=torch.float32)
        pm, BNx, BKx, nw, ns = _CFG_DX[bf16]
        BMx = _route_bm(pm, navg)
        bexp, brow, maxb = tabs.get(BMx)
        _dxs_kernel[(maxb, triton.cdiv(D, BNx))](
            dh1, dh1l, w1, dxs, rdst, bexp, brow, offs, 2 * F, D,
            dh1.stride(0), w1.stride(0), w1.stride(1), dxs.stride(0),
            BM=BMx, BN=BNx, BK=BKx, LIM=_LIM_DX[bf16],
            num_warps=nw, num_stages=ns)

        dx = torch.empty(T, D, device=dev, dtype=x.dtype)
        _dxcomb_kernel[(triton.cdiv(T, 64), triton.cdiv(D, 128))](
            dxs, dlog, router_w, dx, T, D, E, A,
            dxs.stride(0), router_w.stride(0), dx.stride(0),
            BT=64, BN=128, BK=max(16, min(64, triton.next_power_of_2(E))),
            LIM=_LIM_R, num_warps=8, num_stages=2)

        split = max(1, min(32, triton.cdiv(T, 256)))
        BMr = 64
        BNr = max(16, min(64, triton.next_power_of_2(E)))
        part = torch.empty(split, D, E, device=dev, dtype=torch.float32)
        _drw_kernel[(triton.cdiv(D, BMr), triton.cdiv(E, BNr), split)](
            x, dlog, part, T, D, E, x.stride(0), part.stride(0), part.stride(1),
            BM=BMr, BN=BNr, BK=32, SPLIT=split,
            LIM=_LIM_R, num_warps=4, num_stages=3)
        drw = torch.empty_like(router_w)
        _drw_reduce_kernel[(triton.cdiv(D, 64), triton.cdiv(E, 64))](
            part, drw, D, E, part.stride(0), part.stride(1), router_w.stride(0),
            SPLIT=split, BM=64, BN=64, num_warps=4)

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
