"""Routed fused mixture-of-experts layer, forward and backward.

Shape of one call:

  A router GEMM produces the raw sigmoid scores S [T, E] and, in the same
  kernel, the order-preserving int64 selection keys.  torch.topk on those
  keys is the selection; a stable sort of the flattened [T*A] selection
  groups the routed pairs by expert.  A block table turns the ragged
  per-expert row ranges into one flat tile grid, and the expert projections,
  the SwiGLU, the combine and every backward contraction run as tiled Triton
  GEMMs over that table.

Buffers indexed by the routed-pair index r are in expert-major order.  INV
maps (token, slot) -> r, so the combine and the input-gradient reduction walk
each token's A slots in slot order, and every destination row is written by
exactly one program.  Reductions that are split across programs -- the
router-weight gradient and the combine-weight gradient -- partition on a
boundary computed from the shapes, and their partials are summed in index
order by a second kernel.

Precision.  Score arithmetic is float32.  A product whose operands are both
bfloat16 tensors is exact in one tensor-core pass; a float32-valued operand
is carried as an exact bfloat16 hi/lo pair so the dot keeps the full float32
significand.  On float32 inputs both operands are float32-valued and the
dots use a three-pass tf32 accumulation instead, which reaches the same
effective precision without the extra operand tiles two hi/lo expansions
would place in shared memory.  Per-route combine scalars are folded into
whichever operand is already float32-valued, so no product ever needs more
than one operand expanded.
"""

import torch
import triton
import triton.language as tl


def _np2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


# Operand handling for _dotacc.  A float32-valued operand is carried as an
# exact bfloat16 hi/lo pair and the corresponding tensor-core passes are
# summed; an operand that is already a bfloat16 tensor is exact in one pass
# and is not expanded.  On float32 inputs both operands are float32-valued,
# and a three-pass tf32 accumulation carries them without the extra operand
# tiles a hi/lo expansion of both sides would need.
_L_BF = 0      # a, b both exact in bfloat16          -> 1 pass
_L_A2 = 1      # a float32-valued, hi/lo pair         -> 2 passes
_L_B2 = 2      # b float32-valued, hi/lo pair         -> 2 passes
_L_TF3 = 3     # both float32-valued                  -> 3 tf32 passes
_L_IEEE = 4    # full float32 multiply, no tensor core


@triton.jit
def _hilo(v):
    """v (float32) as a bfloat16 hi/lo pair whose sum reproduces v."""
    h = v.to(tl.bfloat16)
    return h, (v - h.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _dotacc(a, b, acc, LIMB: tl.constexpr):
    if LIMB == 0:
        acc = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
    elif LIMB == 1:
        ah, al = _hilo(a.to(tl.float32))
        bh = b.to(tl.bfloat16)
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(al, bh, acc)
    elif LIMB == 2:
        ah = a.to(tl.bfloat16)
        bh, bl = _hilo(b.to(tl.float32))
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(ah, bl, acc)
    elif LIMB == 3:
        acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc,
                     input_precision="tf32x3")
    else:
        acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc,
                     input_precision="ieee")
    return acc


@triton.jit
def _swz(pid, nm, nn, GM: tl.constexpr):
    ing = GM * nn
    gid = pid // ing
    fm = gid * GM
    gs = tl.minimum(nm - fm, GM)
    pm = fm + ((pid % ing) % gs)
    pn = (pid % ing) // gs
    return pm, pn


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

@triton.jit
def _router_kernel(X, RW, RB, S, KEY, T, D, E,
                   BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                   LIMB: tl.constexpr):
    """S = sigmoid(x @ router_w) in float32, plus the selection keys.

    The key packs the biased score's order-preserving integer bits with the
    inverted expert index, so descending key order is (score descending,
    expert index ascending) and no two keys are ever equal.  -0.0 is folded
    onto +0.0 first so both zeros form one tie group.
    """
    pid_t = tl.program_id(0)
    pid_e = tl.program_id(1)
    rt = pid_t * BT + tl.arange(0, BT)
    re = pid_e * BE + tl.arange(0, BE)
    tm = rt < T
    em = re < E
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < D
        a = tl.load(X + rt[:, None] * D + rk[None, :],
                    mask=tm[:, None] & km[None, :], other=0.0)
        b = tl.load(RW + rk[:, None] * E + re[None, :],
                    mask=km[:, None] & em[None, :], other=0.0)
        acc = _dotacc(a, b, acc, LIMB)
    s = tl.sigmoid(acc)
    o = rt[:, None] * E + re[None, :]
    m = tm[:, None] & em[None, :]
    tl.store(S + o, s, mask=m)
    bs = s + tl.load(RB + re, mask=em, other=0.0)[None, :]
    bs = tl.where(bs == 0.0, 0.0, bs)
    ib = bs.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    key = (u << 10) | (E - 1 - re.to(tl.int64))[None, :]
    tl.store(KEY + o, key, mask=m)


@triton.jit
def _cw_kernel(S, SEL, W, T, E, A, BT: tl.constexpr, BA: tl.constexpr):
    """Combine weights: the RAW selected scores, normalized. No bias here."""
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    m = (rt[:, None] < T) & (ra[None, :] < A)
    o = rt[:, None] * A + ra[None, :]
    sel = tl.load(SEL + o, mask=m, other=0)
    ss = tl.load(S + rt[:, None] * E + sel, mask=m, other=0.0)
    tot = tl.sum(tl.where(m, ss, 0.0), axis=1)
    tl.store(W + o, ss / tot[:, None], mask=m)


@triton.jit
def _meta_kernel(ORDER, WF, TOKI, INV, WS, R, A, BLK: tl.constexpr):
    """Expert-major routing metadata from the sort permutation."""
    pid = tl.program_id(0)
    i = pid * BLK + tl.arange(0, BLK)
    m = i < R
    o = tl.load(ORDER + i, mask=m, other=0)
    tl.store(TOKI + i, (o // A).to(tl.int32), mask=m)
    tl.store(INV + o, i.to(tl.int32), mask=m)
    tl.store(WS + i, tl.load(WF + o, mask=m, other=0.0), mask=m)


@triton.jit
def _offs_kernel(VALS, OFFS, R, E):
    """OFFS[e] = first position in the expert-sorted keys with value >= e."""
    e = tl.program_id(0)
    lo = 0
    hi = R
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(VALS + mid) < e:
            lo = mid + 1
        else:
            hi = mid
    tl.store(OFFS + e, lo)


@triton.jit
def _btable_kernel(OFFS, TE, TS, TN, E, BM: tl.constexpr, BE: tl.constexpr):
    """Flatten the ragged per-expert row ranges into one tile list.

    Tile p covers rows [TS[p], TS[p]+TN[p]) of expert TE[p]; TE[p] < 0 marks a
    padding tile, so the grid can be sized from a bound computed on the host.
    """
    p = tl.program_id(0)
    re = tl.arange(0, BE)
    m = re < E
    o0 = tl.load(OFFS + re, mask=m, other=0)
    o1 = tl.load(OFFS + re + 1, mask=m, other=0)
    n = tl.where(m, o1 - o0, 0)
    tiles = (n + BM - 1) // BM
    cs = tl.cumsum(tiles, axis=0)
    st = cs - tiles
    hit = (st <= p) & (p < cs)
    if p < tl.sum(tiles):
        e = tl.sum(tl.where(hit, re, 0))
        j = p - tl.sum(tl.where(hit, st, 0))
        base = tl.sum(tl.where(hit, o0, 0))
        cnt = tl.sum(tl.where(hit, n, 0))
        tl.store(TE + p, e)
        tl.store(TS + p, base + j * BM)
        tl.store(TN + p, tl.minimum(BM, cnt - j * BM))
    else:
        tl.store(TE + p, -1)


# --------------------------------------------------------------------------
# expert forward
# --------------------------------------------------------------------------

@triton.jit
def _gemm1_kernel(X, W1, GU, ACT, ACW, WS, TOKI, TE, TS, TN, D, F, NT,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GM: tl.constexpr, LIMB: tl.constexpr, EK: tl.constexpr):
    """GU[r] = x[tok[r]] @ w1[e(r)], and the SwiGLU activation with it.

    One program owns column tile pn of the GATE half and the matching tile of
    the UP half, so the gathered x tile is loaded once and feeds both, and
    silu(gate)*up is finished in registers.  GU is kept for the backward,
    which needs the two halves separately; ACT is what the second projection
    and the w2 gradient consume, so neither has to re-read both halves.
    """
    nn = tl.cdiv(F, BN)
    pm, pn = _swz(tl.program_id(0), NT, nn, GM)
    e = tl.load(TE + pm)
    if e >= 0:
        rs = tl.load(TS + pm)
        nv = tl.load(TN + pm)
        ir = tl.arange(0, BM)
        rmask = ir < nv
        tok = tl.load(TOKI + rs + ir, mask=rmask, other=0)
        rn = pn * BN + tl.arange(0, BN)
        nmask = rn < F
        F2 = 2 * F
        wp = W1 + e.to(tl.int64) * (D * F2) + rn[None, :]
        xp = X + tok[:, None].to(tl.int64) * D
        ag = tl.zeros((BM, BN), dtype=tl.float32)
        au = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            if EK:
                a = tl.load(xp + rk[None, :], mask=rmask[:, None], other=0.0)
                bg = tl.load(wp + rk[:, None] * F2, mask=nmask[None, :],
                             other=0.0)
                bu = tl.load(wp + F + rk[:, None] * F2, mask=nmask[None, :],
                             other=0.0)
            else:
                km = rk < D
                a = tl.load(xp + rk[None, :],
                            mask=rmask[:, None] & km[None, :], other=0.0)
                bg = tl.load(wp + rk[:, None] * F2,
                             mask=km[:, None] & nmask[None, :], other=0.0)
                bu = tl.load(wp + F + rk[:, None] * F2,
                             mask=km[:, None] & nmask[None, :], other=0.0)
            ag = _dotacc(a, bg, ag, LIMB)
            au = _dotacc(a, bu, au, LIMB)
        om = rmask[:, None] & nmask[None, :]
        row = (rs + ir)[:, None].to(tl.int64)
        gp = GU + row * F2 + rn[None, :]
        tl.store(gp, ag, mask=om)
        tl.store(gp + F, au, mask=om)
        act = ag * tl.sigmoid(ag) * au
        ao = row * F + rn[None, :]
        tl.store(ACT + ao, act, mask=om)
        # dw2 contracts ws * act over the routed pairs.  Folding the scalar
        # in here, where act still has its full float32 significand, keeps
        # that kernel down to one expanded operand.
        tl.store(ACW + ao, act * tl.load(WS + rs + ir, mask=rmask,
                                         other=0.0)[:, None], mask=om)


@triton.jit
def _gemm2_kernel(ACT, W2, FS, TE, TS, TN, F, D, NT,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GM: tl.constexpr, LIMB: tl.constexpr, EK: tl.constexpr):
    """FS[r] = act[r] @ w2[e(r)]."""
    nn = tl.cdiv(D, BN)
    pm, pn = _swz(tl.program_id(0), NT, nn, GM)
    e = tl.load(TE + pm)
    if e >= 0:
        rs = tl.load(TS + pm)
        nv = tl.load(TN + pm)
        ir = tl.arange(0, BM)
        rmask = ir < nv
        rn = pn * BN + tl.arange(0, BN)
        nmask = rn < D
        ap = (rs + ir)[:, None].to(tl.int64) * F
        wp = W2 + e.to(tl.int64) * (F * D) + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, F, BK):
            rk = k0 + tl.arange(0, BK)
            if EK:
                am = rmask[:, None]
                bmk = nmask[None, :]
            else:
                km = rk < F
                am = rmask[:, None] & km[None, :]
                bmk = km[:, None] & nmask[None, :]
            b = tl.load(wp + rk[:, None] * D, mask=bmk, other=0.0)
            acc = _dotacc(tl.load(ACT + ap + rk[None, :], mask=am, other=0.0),
                          b, acc, LIMB)
        tl.store(FS + (rs + ir)[:, None].to(tl.int64) * D + rn[None, :], acc,
                 mask=rmask[:, None] & nmask[None, :])


@triton.jit
def _combine_kernel(FS, INV, W, Y, T, D, A,
                    BT: tl.constexpr, BN: tl.constexpr):
    """y[t] = sum_a w[t,a] * FS[INV[t,a]], slots visited in slot order."""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    rt = pid_t * BT + tl.arange(0, BT)
    rn = pid_n * BN + tl.arange(0, BN)
    tm = rt < T
    nm = rn < D
    acc = tl.zeros((BT, BN), dtype=tl.float32)
    for a in range(A):
        o = rt * A + a
        idx = tl.load(INV + o, mask=tm, other=0)
        w = tl.load(W + o, mask=tm, other=0.0)
        v = tl.load(FS + idx[:, None].to(tl.int64) * D + rn[None, :],
                    mask=tm[:, None] & nm[None, :], other=0.0)
        acc += w[:, None] * v
    tl.store(Y + rt[:, None].to(tl.int64) * D + rn[None, :],
             acc.to(Y.dtype.element_ty), mask=tm[:, None] & nm[None, :])


# --------------------------------------------------------------------------
# expert backward
# --------------------------------------------------------------------------

@triton.jit
def _dact_kernel(DY, W2, GU, DG1, DG2, DWP, TOKI, WS, TE, TS, TN, F, D, NT,
                 NP, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 GM: tl.constexpr, LIMB: tl.constexpr, EK: tl.constexpr,
                 PLANES: tl.constexpr, FUSEDW: tl.constexpr):
    """Gradient through the SwiGLU, and the combine-weight gradient with it.

    The unscaled product P = dy[tok] @ w2[e]^T is what both consumers need:
    the activation gradient is WS * P routed through the SwiGLU derivative,
    and the gradient of the combine scalar is sum_j act[r,j] * P[r,j].  The
    per-tile partial of that row sum lands in DWP and is reduced across the
    N tiles by _rbwd_kernel, so FS is never re-read.  Keeping WS out of the
    dot also leaves both dot operands exact when the inputs are bfloat16.
    """
    nn = tl.cdiv(F, BN)
    pm, pn = _swz(tl.program_id(0), NT, nn, GM)
    e = tl.load(TE + pm)
    if e >= 0:
        rs = tl.load(TS + pm)
        nv = tl.load(TN + pm)
        ir = tl.arange(0, BM)
        rmask = ir < nv
        tok = tl.load(TOKI + rs + ir, mask=rmask, other=0)
        ws = tl.load(WS + rs + ir, mask=rmask, other=0.0)
        rn = pn * BN + tl.arange(0, BN)
        nmask = rn < F
        dp = DY + tok[:, None].to(tl.int64) * D
        wp = W2 + e.to(tl.int64) * (F * D) + rn[None, :] * D
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, D, BK):
            rk = k0 + tl.arange(0, BK)
            if EK:
                dv = tl.load(dp + rk[None, :], mask=rmask[:, None], other=0.0)
                b = tl.load(wp + rk[:, None], mask=nmask[None, :], other=0.0)
            else:
                km = rk < D
                dv = tl.load(dp + rk[None, :],
                             mask=rmask[:, None] & km[None, :], other=0.0)
                b = tl.load(wp + rk[:, None],
                            mask=km[:, None] & nmask[None, :], other=0.0)
            acc = _dotacc(dv, b, acc, LIMB)
        F2 = 2 * F
        row = (rs + ir)[:, None].to(tl.int64)
        gp = GU + row * F2 + rn[None, :]
        sm = rmask[:, None] & nmask[None, :]
        g = tl.load(gp, mask=sm, other=0.0)
        u = tl.load(gp + F, mask=sm, other=0.0)
        sg = tl.sigmoid(g)
        si = g * sg
        if FUSEDW:
            tl.store(DWP + pn.to(tl.int64) * NP + rs + ir,
                     tl.sum(tl.where(sm, si * u * acc, 0.0), axis=1),
                     mask=rmask)
        acc = acc * ws[:, None]
        dg = acc * u * (sg + si * (1.0 - sg))
        du = acc * si
        o = row * F2 + rn[None, :]
        if PLANES:
            # Both consumers of this gradient (dw1, dxs) need it as a
            # bfloat16 limb pair.  Splitting once here costs the same four
            # bytes per element as storing float32 and removes the split
            # from every tile re-read those two kernels make.
            gh = dg.to(tl.bfloat16)
            uh = du.to(tl.bfloat16)
            tl.store(DG1 + o, gh, mask=sm)
            tl.store(DG1 + o + F, uh, mask=sm)
            tl.store(DG2 + o, (dg - gh.to(tl.float32)).to(tl.bfloat16),
                     mask=sm)
            tl.store(DG2 + o + F, (du - uh.to(tl.float32)).to(tl.bfloat16),
                     mask=sm)
        else:
            tl.store(DG1 + o, dg, mask=sm)
            tl.store(DG1 + o + F, du, mask=sm)


@triton.jit
def _dwr_kernel(FS, DY, TOKI, DWP, R, D,
                BR: tl.constexpr, BD: tl.constexpr):
    """Combine-weight gradient as the reference states it: the float32 dot
    of the expert output with the cotangent, over D, in scalar arithmetic.

    dact can produce the same quantity for free from the product it already
    forms, but only to the precision of that product's dots; on float32
    inputs those are a three-pass tf32 accumulation, and this sum cancels
    enough that the two missing significand bits reach the score gradient.
    """
    pid = tl.program_id(0)
    ri = pid * BR + tl.arange(0, BR)
    rm = ri < R
    tok = tl.load(TOKI + ri, mask=rm, other=0)
    fp = FS + ri[:, None].to(tl.int64) * D
    dp = DY + tok[:, None].to(tl.int64) * D
    acc = tl.zeros((BR,), dtype=tl.float32)
    for d0 in range(0, D, BD):
        rd = d0 + tl.arange(0, BD)
        dm = rm[:, None] & (rd < D)[None, :]
        f = tl.load(fp + rd[None, :], mask=dm, other=0.0)
        g = tl.load(dp + rd[None, :], mask=dm, other=0.0).to(tl.float32)
        acc += tl.sum(f * g, axis=1)
    tl.store(DWP + ri, acc, mask=rm)


@triton.jit
def _rbwd_kernel(DWP, W, S, SEL, INV, DLOG, T, E, A, NP, NN,
                 BT: tl.constexpr, BA: tl.constexpr):
    """Reduce the combine-weight gradient partials, then differentiate the
    normalization and the sigmoid into the selected score positions."""
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    m = (rt[:, None] < T) & (ra[None, :] < A)
    o = rt[:, None] * A + ra[None, :]
    sel = tl.load(SEL + o, mask=m, other=0)
    idx = tl.load(INV + o, mask=m, other=0).to(tl.int64)
    dw = tl.zeros((BT, BA), dtype=tl.float32)
    for j in range(NN):
        dw += tl.load(DWP + j * NP + idx, mask=m, other=0.0)
    w = tl.load(W + o, mask=m, other=0.0)
    ss = tl.load(S + rt[:, None] * E + sel, mask=m, other=0.0)
    tot = tl.sum(tl.where(m, ss, 0.0), axis=1)
    dot = tl.sum(tl.where(m, dw * w, 0.0), axis=1)
    ds = (dw - dot[:, None]) / tot[:, None]
    tl.store(DLOG + rt[:, None] * E + sel, ds * ss * (1.0 - ss), mask=m)


@triton.jit
def _dxs_kernel(DG1, DG2, W1, DXS, TE, TS, TN, D, F2, NT,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GM: tl.constexpr, LIMB: tl.constexpr, EK: tl.constexpr,
                PLANES: tl.constexpr):
    """Per-route input gradient DXS[r] = dgu[r] @ w1[e(r)]^T."""
    nn = tl.cdiv(D, BN)
    pm, pn = _swz(tl.program_id(0), NT, nn, GM)
    e = tl.load(TE + pm)
    if e >= 0:
        rs = tl.load(TS + pm)
        nv = tl.load(TN + pm)
        ir = tl.arange(0, BM)
        rmask = ir < nv
        rn = pn * BN + tl.arange(0, BN)
        nmask = rn < D
        row = (rs + ir)[:, None].to(tl.int64) * F2
        wp = W1 + e.to(tl.int64) * (D * F2) + rn[None, :] * F2
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, F2, BK):
            rk = k0 + tl.arange(0, BK)
            if EK:
                am = rmask[:, None]
                bmk = nmask[None, :]
            else:
                km = rk < F2
                am = rmask[:, None] & km[None, :]
                bmk = km[:, None] & nmask[None, :]
            b = tl.load(wp + rk[:, None], mask=bmk, other=0.0)
            if PLANES:
                bb = b.to(tl.bfloat16)
                acc = tl.dot(tl.load(DG1 + row + rk[None, :], mask=am,
                                     other=0.0), bb, acc)
                acc = tl.dot(tl.load(DG2 + row + rk[None, :], mask=am,
                                     other=0.0), bb, acc)
            else:
                acc = _dotacc(tl.load(DG1 + row + rk[None, :], mask=am,
                                      other=0.0), b, acc, LIMB)
        tl.store(DXS + (rs + ir)[:, None].to(tl.int64) * D + rn[None, :], acc,
                 mask=rmask[:, None] & nmask[None, :])


@triton.jit
def _dw2_kernel(ACW, DY, TOKI, OFFS, DW2, F, D,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GM: tl.constexpr, LIMB: tl.constexpr):
    """dw2[e] = (ws * act)[rows(e)]^T @ dy[tok(rows(e))].

    The route scalar is folded into the activation once, in the forward, so
    the only operand needing expansion here is that product and dy stays
    exact in its own dtype.  An expert with no rows runs zero k iterations
    and stores exact zeros.
    """
    e = tl.program_id(1)
    nm = tl.cdiv(F, BM)
    nn = tl.cdiv(D, BN)
    pm, pn = _swz(tl.program_id(0), nm, nn, GM)
    rs = tl.load(OFFS + e)
    cnt = tl.load(OFFS + e + 1) - rs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mmask = rm < F
    nmask = rn < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, cnt, BK):
        ik = k0 + tl.arange(0, BK)
        km = ik < cnt
        rk = rs + ik
        o = rk[None, :].to(tl.int64) * F + rm[:, None]
        am = mmask[:, None] & km[None, :]
        tok = tl.load(TOKI + rk, mask=km, other=0)
        b = tl.load(DY + tok[:, None].to(tl.int64) * D + rn[None, :],
                    mask=km[:, None] & nmask[None, :], other=0.0)
        acc = _dotacc(tl.load(ACW + o, mask=am, other=0.0), b, acc, LIMB)
    tl.store(DW2 + e.to(tl.int64) * (F * D) + rm[:, None] * D + rn[None, :],
             acc.to(DW2.dtype.element_ty), mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _dw1_kernel(X, DG1, DG2, TOKI, OFFS, DW1, D, F2,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GM: tl.constexpr, LIMB: tl.constexpr, PLANES: tl.constexpr):
    """dw1[e] = x[tok(rows(e))]^T @ dgu[rows(e)]."""
    e = tl.program_id(1)
    nm = tl.cdiv(D, BM)
    nn = tl.cdiv(F2, BN)
    pm, pn = _swz(tl.program_id(0), nm, nn, GM)
    rs = tl.load(OFFS + e)
    cnt = tl.load(OFFS + e + 1) - rs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mmask = rm < D
    nmask = rn < F2
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, cnt, BK):
        ik = k0 + tl.arange(0, BK)
        km = ik < cnt
        rk = rs + ik
        tok = tl.load(TOKI + rk, mask=km, other=0)
        a = tl.load(X + tok[None, :].to(tl.int64) * D + rm[:, None],
                    mask=mmask[:, None] & km[None, :], other=0.0)
        o = rk[:, None].to(tl.int64) * F2 + rn[None, :]
        bm_ = km[:, None] & nmask[None, :]
        if PLANES:
            ab = a.to(tl.bfloat16)
            acc = tl.dot(ab, tl.load(DG1 + o, mask=bm_, other=0.0), acc)
            acc = tl.dot(ab, tl.load(DG2 + o, mask=bm_, other=0.0), acc)
        else:
            acc = _dotacc(a, tl.load(DG1 + o, mask=bm_, other=0.0), acc, LIMB)
    tl.store(DW1 + e.to(tl.int64) * (D * F2) + rm[:, None] * F2 + rn[None, :],
             acc.to(DW1.dtype.element_ty), mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _dxcomb_kernel(DXS, INV, DLOG, RW, DX, T, D, E, A,
                   BT: tl.constexpr, BN: tl.constexpr, BE: tl.constexpr,
                   LIMB: tl.constexpr):
    """dx = sum over the token's slots of DXS, plus the router path."""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    rt = pid_t * BT + tl.arange(0, BT)
    rn = pid_n * BN + tl.arange(0, BN)
    tm = rt < T
    nm = rn < D
    acc = tl.zeros((BT, BN), dtype=tl.float32)
    for a in range(A):
        idx = tl.load(INV + rt * A + a, mask=tm, other=0)
        acc += tl.load(DXS + idx[:, None].to(tl.int64) * D + rn[None, :],
                       mask=tm[:, None] & nm[None, :], other=0.0)
    for e0 in range(0, E, BE):
        re = e0 + tl.arange(0, BE)
        em = re < E
        dl = tl.load(DLOG + rt[:, None] * E + re[None, :],
                     mask=tm[:, None] & em[None, :], other=0.0)
        rw = tl.load(RW + rn[None, :].to(tl.int64) * E + re[:, None],
                     mask=em[:, None] & nm[None, :], other=0.0)
        acc = _dotacc(dl, rw, acc, LIMB)
    tl.store(DX + rt[:, None].to(tl.int64) * D + rn[None, :],
             acc.to(DX.dtype.element_ty), mask=tm[:, None] & nm[None, :])


@triton.jit
def _drw_kernel(X, DLOG, PART, DRW, T, D, E, NS,
                BM: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                LIMB: tl.constexpr, SPLIT: tl.constexpr):
    """drouter_w = x^T @ DLOG, optionally over a fixed token partition.

    The partition boundaries come from T and NS only, so the set of terms in
    each partial -- and hence the summation order in _drw_red_kernel -- is
    the same on every run.
    """
    pm = tl.program_id(0)
    pe = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    re = pe * BE + tl.arange(0, BE)
    mmask = rm < D
    emask = re < E
    if SPLIT:
        sp = tl.program_id(2)
        chunk = tl.cdiv(tl.cdiv(T, BK), NS) * BK
        k0 = sp * chunk
        kend = tl.minimum(k0 + chunk, T)
    else:
        k0 = 0
        kend = T
    acc = tl.zeros((BM, BE), dtype=tl.float32)
    for kk in range(k0, kend, BK):
        rk = kk + tl.arange(0, BK)
        km = rk < T
        a = tl.load(X + rk[None, :].to(tl.int64) * D + rm[:, None],
                    mask=mmask[:, None] & km[None, :], other=0.0)
        b = tl.load(DLOG + rk[:, None] * E + re[None, :],
                    mask=km[:, None] & emask[None, :], other=0.0)
        acc = _dotacc(a, b, acc, LIMB)
    om = mmask[:, None] & emask[None, :]
    if SPLIT:
        tl.store(PART + sp.to(tl.int64) * (D * E) + rm[:, None] * E
                 + re[None, :], acc, mask=om)
    else:
        tl.store(DRW + rm[:, None] * E + re[None, :],
                 acc.to(DRW.dtype.element_ty), mask=om)


@triton.jit
def _drw_red_kernel(PART, DRW, N, NS, BLK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLK + tl.arange(0, BLK)
    m = o < N
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for i in range(NS):
        acc += tl.load(PART + i.to(tl.int64) * N + o, mask=m, other=0.0)
    tl.store(DRW + o, acc.to(DRW.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------
# launch policy
# --------------------------------------------------------------------------

# Row tile for the routed-pair dimension of the four expert GEMMs.  A
# narrower tile wastes fewer padded rows on lightly loaded experts, but the
# measured cost of that padding stays under about 12% even at two routed
# pairs per expert, while the wider tile re-reads each expert's weight slice
# proportionally fewer times; the second effect dominates at every shape
# measured, so the tile is fixed rather than fitted to a load estimate.
_ROW_TILE = 128


def _rowtile(R: int) -> int:
    return _ROW_TILE if R >= _ROW_TILE else max(16, _np2(R))


# Float32 inputs: both operands carry a full float32 significand everywhere.
_LIMB_F32 = {k: _L_TF3 for k in
             ("g1", "g2", "dact", "dxs", "dw1", "dw2")}
# The score GEMM is O(T*D*E) and decides selection against a 1e-4 margin;
# the two router-path contractions are the same size and consume score
# gradients whose terms cancel heavily once the sigmoid saturates.  All
# three take the exact float32 multiply: together they are a low percent of
# the runtime, and tf32x3's two missing significand bits show up there.
_LIMB_F32["router"] = _L_IEEE
_LIMB_F32["dxc"] = _L_IEEE
_LIMB_F32["drw"] = _L_IEEE
# Bfloat16 inputs: x, w1, w2, router_w and dy are already exact in one pass,
# so only the float32-valued operand is expanded.  dxs and dw1 read an
# operand that dact already stored as limbs, and use these codes only on the
# float32 path, where that operand stays float32.
_LIMB_BF16 = {"router": _L_BF, "g1": _L_BF, "g2": _L_A2, "dact": _L_BF,
              "dxs": _L_A2, "dw1": _L_B2, "dw2": _L_A2, "dxc": _L_A2,
              "drw": _L_B2}


# Output tile and launch shape for the weight gradients, as
# (BM, BN, BK, num_warps, num_stages), per input dtype.  Both contract over
# the routed-pair dimension, so a program re-reads each operand once per
# tile of the OPPOSITE output extent: dw1 re-reads the gathered x once per
# F2 tile and the activation gradient once per D tile, dw2 likewise.
_W1_TILE = {torch.bfloat16: (128, 128, 32, 4, 4),
            torch.float32: (128, 64, 32, 4, 3)}
_W2_TILE = {torch.bfloat16: (64, 128, 32, 4, 3),
            torch.float32: (128, 128, 64, 8, 3)}


def _wtile(base, dt, M, N):
    bm, bn, bk, w, s = base[dt]
    if M >= bm and N >= bn:
        return bm, bn, bk, w, s
    bm = min(bm, 64 if M >= 64 else 32)
    bn = min(bn, 64 if N >= 64 else 32)
    return bm, bn, bk, 4, s


# Column tile and launch shape for the four expert GEMMs, as
# (BN, num_warps, num_stages), per input dtype.  These kernels re-read their
# streamed operand once per column tile, so BN divides that stream, and it
# costs no padding, unlike the row tile shared with the ragged block table.
# The two dtypes want opposite extremes: a tf32x3 dot holds three times the
# operand state of a single bfloat16 pass, so the float32 path runs out of
# registers at the tile widths the bfloat16 path prefers.
_G1_TILE = {torch.bfloat16: (128, 8, 3), torch.float32: (64, 8, 3)}
_G2_TILE = {torch.bfloat16: (256, 8, 3), torch.float32: (64, 4, 3)}
_DACT_TILE = {torch.bfloat16: (64, 4, 3), torch.float32: (64, 4, 3)}
_DXS_TILE = {torch.bfloat16: (256, 8, 3), torch.float32: (64, 4, 3)}


def _ntile(base, dt, N):
    bn, w, s = base[dt]
    while bn > 64 and N < bn:
        bn //= 2
    if N < bn:
        bn = 64
    return bn, w, s


def _cfg(dt):
    if dt == torch.float32:
        return _LIMB_F32, 32
    return _LIMB_BF16, 64


class _MoE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        x = x.contiguous()
        router_w = router_w.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        T, D = x.shape
        E = router_w.shape[1]
        F2 = w1.shape[2]
        F = F2 // 2
        R = T * A
        dt = x.dtype
        dev = x.device
        LB, BKk = _cfg(dt)

        S = torch.empty(T, E, device=dev, dtype=torch.float32)
        KEY = torch.empty(T, E, device=dev, dtype=torch.int64)
        BE = max(16, min(128, _np2(E)))
        _router_kernel[(triton.cdiv(T, 64), triton.cdiv(E, BE))](
            x, router_w, router_b, S, KEY, T, D, E,
            BT=64, BE=BE, BK=BKk, LIMB=LB["router"],
            num_warps=4, num_stages=3)

        sel = torch.topk(KEY, A, dim=-1, largest=True,
                         sorted=True).indices.to(torch.int32).contiguous()

        BA = _np2(A)
        BTw = max(1, min(64, 4096 // BA))
        W = torch.empty(T, A, device=dev, dtype=torch.float32)
        _cw_kernel[(triton.cdiv(T, BTw),)](S, sel, W, T, E, A,
                                           BT=BTw, BA=BA, num_warps=4)

        vals, order = torch.sort(sel.view(-1), stable=True)

        TOKI = torch.empty(R, device=dev, dtype=torch.int32)
        INV = torch.empty(R, device=dev, dtype=torch.int32)
        WS = torch.empty(R, device=dev, dtype=torch.float32)
        _meta_kernel[(triton.cdiv(R, 256),)](order, W.view(-1), TOKI, INV, WS,
                                             R, A, BLK=256, num_warps=4)

        OFFS = torch.empty(E + 1, device=dev, dtype=torch.int32)
        _offs_kernel[(E + 1,)](vals, OFFS, R, E, num_warps=1)

        BM = _rowtile(R)
        NT = triton.cdiv(R, BM) + E
        TE_ = torch.empty(NT, device=dev, dtype=torch.int32)
        TS_ = torch.empty(NT, device=dev, dtype=torch.int32)
        TN_ = torch.empty(NT, device=dev, dtype=torch.int32)
        _btable_kernel[(NT,)](OFFS, TE_, TS_, TN_, E, BM=BM,
                              BE=max(16, _np2(E)), num_warps=4)

        GU = torch.empty(R, F2, device=dev, dtype=torch.float32)
        ACT = torch.empty(R, F, device=dev, dtype=torch.float32)
        ACW = torch.empty(R, F, device=dev, dtype=torch.float32)
        bn, nw, ns = _ntile(_G1_TILE, dt, F)
        _gemm1_kernel[(NT * triton.cdiv(F, bn),)](
            x, w1, GU, ACT, ACW, WS, TOKI, TE_, TS_, TN_, D, F, NT,
            BM=BM, BN=bn, BK=BKk, GM=8, LIMB=LB["g1"], EK=(D % BKk == 0),
            num_warps=nw, num_stages=ns)

        FS = torch.empty(R, D, device=dev, dtype=torch.float32)
        bn, nw, ns = _ntile(_G2_TILE, dt, D)
        _gemm2_kernel[(NT * triton.cdiv(D, bn),)](
            ACT, w2, FS, TE_, TS_, TN_, F, D, NT,
            BM=BM, BN=bn, BK=BKk, GM=8, LIMB=LB["g2"], EK=(F % BKk == 0),
            num_warps=nw, num_stages=ns)

        y = torch.empty(T, D, device=dev, dtype=dt)
        _combine_kernel[(triton.cdiv(T, 64), triton.cdiv(D, 128))](
            FS, INV, W, y, T, D, A, BT=64, BN=128, num_warps=4, num_stages=2)

        ctx.save_for_backward(x, router_w, w1, w2, S, sel, W, TOKI, INV, WS,
                              OFFS, TE_, TS_, TN_, GU, ACW,
                              FS if dt == torch.float32 else W)
        ctx.meta = (T, D, F, E, A, R, BM, NT, dt)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, router_w, w1, w2, S, sel, W, TOKI, INV, WS,
         OFFS, TE_, TS_, TN_, GU, ACW, FS) = ctx.saved_tensors
        T, D, F, E, A, R, BM, NT, dt = ctx.meta
        F2 = 2 * F
        dev = x.device
        LB, BKk = _cfg(dt)
        dy = dy.contiguous()

        # The activation gradient is stored the same way, for the same
        # reason: dw1 and dxs are its only consumers and both want limbs.
        planes = dt != torch.float32
        if planes:
            DG1 = torch.empty(R, F2, device=dev, dtype=torch.bfloat16)
            DG2 = torch.empty(R, F2, device=dev, dtype=torch.bfloat16)
        else:
            DG1 = torch.empty(R, F2, device=dev, dtype=torch.float32)
            DG2 = DG1
        bn, nw, ns = _ntile(_DACT_TILE, dt, F)
        NN = triton.cdiv(F, bn)
        fusedw = planes
        DWP = torch.empty(NN if fusedw else 1, R, device=dev,
                          dtype=torch.float32)
        _dact_kernel[(NT * NN,)](
            dy, w2, GU, DG1, DG2, DWP, TOKI, WS, TE_, TS_, TN_, F, D, NT, R,
            BM=BM, BN=bn, BK=BKk, GM=8, LIMB=LB["dact"], EK=(D % BKk == 0),
            PLANES=planes, FUSEDW=fusedw, num_warps=nw, num_stages=ns)
        if not fusedw:
            _dwr_kernel[(triton.cdiv(R, 64),)](FS, dy, TOKI, DWP, R, D,
                                               BR=64, BD=128, num_warps=4,
                                               num_stages=2)

        DLOG = torch.zeros(T, E, device=dev, dtype=torch.float32)
        BA = _np2(A)
        BTw = max(1, min(64, 4096 // BA))
        _rbwd_kernel[(triton.cdiv(T, BTw),)](DWP, W, S, sel, INV, DLOG,
                                             T, E, A, R, NN if fusedw else 1,
                                             BT=BTw, BA=BA, num_warps=4)

        bm, bn, bk, nw, ns = _wtile(_W2_TILE, dt, F, D)
        dw2 = torch.empty_like(w2)
        _dw2_kernel[(triton.cdiv(F, bm) * triton.cdiv(D, bn), E)](
            ACW, dy, TOKI, OFFS, dw2, F, D,
            BM=bm, BN=bn, BK=bk, GM=8, LIMB=LB["dw2"],
            num_warps=nw, num_stages=ns)

        bm, bn, bk, nw, ns = _wtile(_W1_TILE, dt, D, F2)
        dw1 = torch.empty_like(w1)
        _dw1_kernel[(triton.cdiv(D, bm) * triton.cdiv(F2, bn), E)](
            x, DG1, DG2, TOKI, OFFS, dw1, D, F2,
            BM=bm, BN=bn, BK=bk, GM=8, LIMB=LB["dw1"], PLANES=planes,
            num_warps=nw, num_stages=ns)

        DXS = torch.empty(R, D, device=dev, dtype=torch.float32)
        bn, nw, ns = _ntile(_DXS_TILE, dt, D)
        _dxs_kernel[(NT * triton.cdiv(D, bn),)](
            DG1, DG2, w1, DXS, TE_, TS_, TN_, D, F2, NT,
            BM=BM, BN=bn, BK=BKk, GM=8, LIMB=LB["dxs"], EK=(F2 % BKk == 0),
            PLANES=planes, num_warps=nw, num_stages=ns)

        dx = torch.empty(T, D, device=dev, dtype=dt)
        _dxcomb_kernel[(triton.cdiv(T, 64), triton.cdiv(D, 128))](
            DXS, INV, DLOG, router_w, dx, T, D, E, A,
            BT=64, BN=128, BE=max(16, min(64, _np2(E))), LIMB=LB["dxc"],
            num_warps=4, num_stages=2)

        BEr = max(16, min(64, _np2(E)))
        drw = torch.empty(D, E, device=dev, dtype=dt)
        NS = max(1, min(16, T // 1024))
        ge = (triton.cdiv(D, 64), triton.cdiv(E, BEr))
        if NS > 1:
            PART = torch.empty(NS, D, E, device=dev, dtype=torch.float32)
            _drw_kernel[ge + (NS,)](
                x, DLOG, PART, drw, T, D, E, NS, BM=64, BE=BEr, BK=64,
                LIMB=LB["drw"], SPLIT=True, num_warps=4, num_stages=3)
            _drw_red_kernel[(triton.cdiv(D * E, 512),)](
                PART, drw, D * E, NS, BLK=512, num_warps=4)
        else:
            _drw_kernel[ge + (1,)](
                x, DLOG, drw, drw, T, D, E, 1, BM=64, BE=BEr, BK=64,
                LIMB=LB["drw"], SPLIT=False, num_warps=4, num_stages=3)

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
    return _MoE.apply(x, router_w, router_b, w1, w2, A)
