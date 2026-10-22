"""Routed fused mixture-of-experts layer, forward and backward.

One autograd.Function drives a pipeline of Triton kernels:

    router GEMM + sigmoid + selection keys -> top-A -> combine weights
      -> slots sorted by expert -> per-expert slot ranges -> tile map
      -> grouped GEMM 1 (gate|up, SwiGLU and the combine weight fused into
         the epilogue)
      -> grouped GEMM 2 -> ordered combine

and the mirror image on the way back: the grouped dact GEMM (which also
forms the per-slot combine-weight cotangents), the sigmoid/router chain
rule, then the dw2, dx-slot and dw1 grouped GEMMs and an ordered per-token
combine carrying the router term.

Tokens never move: the slot permutation is an index gather inside the GEMMs,
so the [T, D] activations are read in place.

Dot precision. Every dot runs on bfloat16 tensor-core operands with float32
accumulation. A value that is not exact in bfloat16 is carried as a high/low
pair of bfloat16 tensors -- hi = bf16(v), lo = bf16(v - hi), which together
hold the float32 mantissa -- and the dot expands to ah*bh + ah*bl + al*bh.
That is what the operands feeding the weight gradients use, and it is what
the whole float32 input path uses: with bfloat16 inputs the tokens and the
expert weights are already exact, so only the float32 intermediates (the
weighted activation and the gate/up cotangent) carry a low part.
"""

import torch
import triton
import triton.language as tl


def _pow2(n, lo=1):
    n = max(int(n), lo)
    return 1 << (n - 1).bit_length()


@triton.jit
def _dot3(ah, al, bh, bl, acc, SPLIT_A: tl.constexpr, SPLIT_B: tl.constexpr,
          TRANS_A: tl.constexpr, TRANS_B: tl.constexpr):
    """acc += op(a) @ op(b) from bfloat16 high/low operand pairs."""
    a0 = tl.trans(ah) if TRANS_A else ah
    b0 = tl.trans(bh) if TRANS_B else bh
    acc += tl.dot(a0, b0, input_precision="ieee")
    if SPLIT_B:
        acc += tl.dot(a0, tl.trans(bl) if TRANS_B else bl,
                      input_precision="ieee")
    if SPLIT_A:
        acc += tl.dot(tl.trans(al) if TRANS_A else al, b0,
                      input_precision="ieee")
    return acc


@triton.jit
def _split_kernel(SRC, HI, LO, n, BLOCK: tl.constexpr):
    """v -> (bf16(v), bf16(v - bf16(v))): an exact bfloat16 pair for v."""
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    v = tl.load(SRC + o, mask=m, other=0.0)
    hi = v.to(tl.bfloat16)
    tl.store(HI + o, hi, mask=m)
    tl.store(LO + o, (v - hi.to(tl.float32)).to(tl.bfloat16), mask=m)


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


@triton.jit
def _router_kernel(XH, XL, RH, RL, RB, S, KEYS,
                   T, D, E,
                   sx_t, sx_d, sr_d, sr_e,
                   BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
                   SPLIT: tl.constexpr):
    """s = sigmoid(x @ router_w) in float32, plus the packed selection keys.

    key = (order-preserving bits of the biased float32 score) << 10
          | (E - 1 - expert), so descending key order is exactly
          (biased score descending, expert index ascending) and equal scores
          never compete. -0.0 is folded onto +0.0 first.

    On the float32 path both operands are split and all four products are
    accumulated, so the logit keeps the float32 mantissa going into the
    selection.
    """
    pid = tl.program_id(0)
    rt = pid * BT + tl.arange(0, BT)
    re = tl.arange(0, BE)
    mt = rt < T
    me = re < E
    rt64 = rt.to(tl.int64)
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < D
        ma = mt[:, None] & mk[None, :]
        mb = mk[:, None] & me[None, :]
        ah = tl.load(XH + rt64[:, None] * sx_t + rk[None, :] * sx_d,
                     mask=ma, other=0.0)
        bh = tl.load(RH + rk[:, None] * sr_d + re[None, :] * sr_e,
                     mask=mb, other=0.0)
        acc += tl.dot(ah, bh, input_precision="ieee")
        if SPLIT:
            al = tl.load(XL + rt64[:, None] * sx_t + rk[None, :] * sx_d,
                         mask=ma, other=0.0)
            bl = tl.load(RL + rk[:, None] * sr_d + re[None, :] * sr_e,
                         mask=mb, other=0.0)
            acc += tl.dot(ah, bl, input_precision="ieee")
            acc += tl.dot(al, bh, input_precision="ieee")
            acc += tl.dot(al, bl, input_precision="ieee")
    s = tl.sigmoid(acc)
    rb = tl.load(RB + re, mask=me, other=0.0)
    biased = s + rb[None, :]
    biased = tl.where(biased == 0.0, 0.0, biased)
    ib = biased.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)
    key = (u << 10) | (E - 1 - re[None, :]).to(tl.int64)
    tl.store(S + rt64[:, None] * E + re[None, :], s,
             mask=mt[:, None] & me[None, :])
    tl.store(KEYS + rt64[:, None] * BE + re[None, :],
             tl.where(me[None, :], key, -1), mask=mt[:, None])


@triton.jit
def _emit_tiles(lo, re, E, TILE_E, TILE_M, NT_MAX,
                BM: tl.constexpr, BT: tl.constexpr):
    """tile id -> (expert, first slot) for one row-tile height."""
    ok = re <= E
    prev = tl.sum(tl.where(re[:, None] == (re[None, :] + 1), lo[None, :], 0), 1)
    nt = tl.where((re >= 1) & ok, (lo - prev + BM - 1) // BM, 0)
    cum = tl.cumsum(nt, 0)
    nt_all = tl.sum(nt, 0)
    start = tl.where(re < E, cum, 0x7FFFFFFF)
    for t0 in range(0, NT_MAX, BT):
        rt = t0 + tl.arange(0, BT)
        m = rt < NT_MAX
        e = tl.maximum(tl.sum(tl.where(start[None, :] <= rt[:, None], 1, 0),
                              1) - 1, 0)
        base = tl.sum(tl.where(re[None, :] == e[:, None], cum[None, :], 0), 1)
        off = tl.sum(tl.where(re[None, :] == e[:, None], lo[None, :], 0), 1)
        tl.store(TILE_E + rt, tl.where(rt < nt_all, e, -1), mask=m)
        tl.store(TILE_M + rt, off + (rt - base) * BM, mask=m)


@triton.jit
def _meta_kernel(SORTED_E, ORDER, S, SEL, TOK, WS, INV, OFFS, TILE_E, TILE_M,
                 TILE_E2, TILE_M2,
                 T, N, E, A, NT_MAX, NT_MAX2, ROUNDS,
                 BM: tl.constexpr, BM2: tl.constexpr, BT: tl.constexpr,
                 BA: tl.constexpr,
                 BE: tl.constexpr, BS: tl.constexpr, BLOCK: tl.constexpr):
    """All routing bookkeeping downstream of the sort.

    Program 0 turns the sorted expert ids into per-expert slot ranges and a
    row-tile map; the rest walk the slots, carrying each slot's token, its
    combine weight (the raw selected scores normalized per token) and the
    inverse permutation.

    OFFS[e], the first slot of expert e, is a lower bound found by a BS-way
    search: each round probes BS evenly spaced positions per expert at once
    and keeps the bracketing interval, so the whole search is a handful of
    dependent gathers. tile id -> (expert, first slot) then follows from the
    running tile count, and tiles past the end are marked -1.
    """
    pid = tl.program_id(0)
    if pid == 0:
        re = tl.arange(0, BE)
        ok = re <= E
        js = tl.arange(0, BS)
        lo = tl.zeros((BE,), dtype=tl.int32)
        width = tl.full((BE,), N, dtype=tl.int32)
        for _ in range(ROUNDS):
            live = width > 0
            step = tl.cdiv(width, BS - 1)
            pos = lo[:, None] + js[None, :] * step[:, None]
            v = tl.load(SORTED_E + pos, mask=pos < N, other=0x7FFFFFFF)
            cnt = tl.minimum(tl.sum(tl.where(v < re[:, None], 1, 0), 1),
                             BS - 1)
            adv = tl.where(cnt > 0, (cnt - 1) * step + 1, 0)
            lo = tl.where(live, lo + adv, lo)
            width = tl.where(live & (cnt > 0), step - 1, 0)
        lo = tl.where(ok, tl.minimum(lo, N), 0)
        tl.store(OFFS + re, lo, mask=ok)
        _emit_tiles(lo, re, E, TILE_E, TILE_M, NT_MAX, BM, BT)
        if BM2 != BM:
            _emit_tiles(lo, re, E, TILE_E2, TILE_M2, NT_MAX2, BM2, BT)
    else:
        p = (pid - 1) * BLOCK + tl.arange(0, BLOCK)
        m = p < N
        o = tl.load(ORDER + p, mask=m, other=0)
        t = o // A
        ra = tl.arange(0, BA)
        ma = m[:, None] & (ra < A)[None, :]
        sel = tl.load(SEL + t[:, None] * A + ra[None, :], mask=ma, other=0)
        z = tl.load(S + t[:, None] * E + sel, mask=ma, other=0.0)
        ssum = tl.sum(z, 1)
        mine = tl.sum(tl.where(ra[None, :] == (o - t * A)[:, None], z, 0.0), 1)
        tl.store(TOK + p, t.to(tl.int32), mask=m)
        tl.store(WS + p, mine / ssum, mask=m)
        tl.store(INV + o, p.to(tl.int32), mask=m)


@triton.jit
def _swizzle(pid, NT_M, NT_N: tl.constexpr, GM: tl.constexpr):
    """1D program id -> (row tile, column tile), grouped so that a block of
    GM row tiles sweeps every column tile before the next block starts."""
    gsz = GM * NT_N
    g = pid // gsz
    r = pid % gsz
    first = g * GM
    gm = min(NT_M - first, GM)
    return first + r % gm, r // gm


# ---------------------------------------------------------------------------
# forward expert compute
# ---------------------------------------------------------------------------


@triton.jit
def _gemm1_kernel(XH, XL, WH, WL, GU, AH, AL, WS,
                  TILE_E, TILE_M, OFFS, TOK, D, F,
                  sx_t, sx_d, sw_e, sw_d, sw_f,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  NT_M, NT_N: tl.constexpr, GM: tl.constexpr,
                  SPLIT: tl.constexpr, EVEN_K: tl.constexpr):
    """gate|up projections of the routed tokens.

    The epilogue applies SwiGLU and the slot's combine weight, and stores the
    weighted activation as a bfloat16 high/low pair: the forward GEMM then
    reads only the high part, and the weight gradient reads both.
    """
    tid, pn = _swizzle(tl.program_id(0), NT_M, NT_N, GM)
    e = tl.load(TILE_E + tid)
    if e < 0:
        return
    m0 = tl.load(TILE_M + tid)
    row_end = tl.load(OFFS + e + 1)
    rows = m0 + tl.arange(0, BM)
    mrow = rows < row_end
    tok = tl.load(TOK + rows, mask=mrow, other=0).to(tl.int64)
    rn = pn * BN + tl.arange(0, BN)
    mn = rn < F
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    xh = XH + tok[:, None] * sx_t
    xl = XL + tok[:, None] * sx_t
    wg = WH + e.to(tl.int64) * sw_e + rn[None, :] * sw_f
    wu = wg + F * sw_f
    wgl = WL + e.to(tl.int64) * sw_e + rn[None, :] * sw_f
    wul = wgl + F * sw_f
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        if EVEN_K:
            ma = mrow[:, None]
            mb = mn[None, :]
        else:
            mk = rk < D
            ma = mrow[:, None] & mk[None, :]
            mb = mk[:, None] & mn[None, :]
        ah = tl.load(xh + rk[None, :] * sx_d, mask=ma, other=0.0)
        bg = tl.load(wg + rk[:, None] * sw_d, mask=mb, other=0.0)
        bu = tl.load(wu + rk[:, None] * sw_d, mask=mb, other=0.0)
        if SPLIT:
            al = tl.load(xl + rk[None, :] * sx_d, mask=ma, other=0.0)
            bgl = tl.load(wgl + rk[:, None] * sw_d, mask=mb, other=0.0)
            bul = tl.load(wul + rk[:, None] * sw_d, mask=mb, other=0.0)
        else:
            al = ah
            bgl = bg
            bul = bu
        accg = _dot3(ah, al, bg, bgl, accg, SPLIT, SPLIT, False, False)
        accu = _dot3(ah, al, bu, bul, accu, SPLIT, SPLIT, False, False)
    sig = tl.sigmoid(accg)
    wv = tl.load(WS + rows, mask=mrow, other=0.0)
    a = (accg * sig) * accu * wv[:, None]
    hi = a.to(tl.bfloat16)
    ms = mrow[:, None] & mn[None, :]
    r64 = rows.to(tl.int64)
    gp = GU + r64[:, None] * (2 * F) + rn[None, :]
    tl.store(gp, accg, mask=ms)
    tl.store(gp + F, accu, mask=ms)
    ap = r64[:, None] * F + rn[None, :]
    tl.store(AH + ap, hi, mask=ms)
    tl.store(AL + ap, (a - hi.to(tl.float32)).to(tl.bfloat16), mask=ms)


@triton.jit
def _gemm2_kernel(AH, AL, WH, WL, O, TILE_E, TILE_M, OFFS, F, D,
                  sw_e, sw_f, sw_d,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  NT_M, NT_N: tl.constexpr, GM: tl.constexpr,
                  SPLIT: tl.constexpr, EVEN_K: tl.constexpr):
    """expert output projection of the weighted activation."""
    tid, pn = _swizzle(tl.program_id(0), NT_M, NT_N, GM)
    e = tl.load(TILE_E + tid)
    if e < 0:
        return
    m0 = tl.load(TILE_M + tid)
    row_end = tl.load(OFFS + e + 1)
    rows = m0 + tl.arange(0, BM)
    mrow = rows < row_end
    r64 = rows.to(tl.int64)
    rn = pn * BN + tl.arange(0, BN)
    mn = rn < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    ap = r64[:, None] * F
    wp = WH + e.to(tl.int64) * sw_e + rn[None, :] * sw_d
    wl = WL + e.to(tl.int64) * sw_e + rn[None, :] * sw_d
    for k0 in range(0, F, BK):
        rk = k0 + tl.arange(0, BK)
        if EVEN_K:
            ma = mrow[:, None]
            mb = mn[None, :]
        else:
            mk = rk < F
            ma = mrow[:, None] & mk[None, :]
            mb = mk[:, None] & mn[None, :]
        ah = tl.load(AH + ap + rk[None, :], mask=ma, other=0.0)
        bh = tl.load(wp + rk[:, None] * sw_f, mask=mb, other=0.0)
        if SPLIT:
            al = tl.load(AL + ap + rk[None, :], mask=ma, other=0.0)
            bl = tl.load(wl + rk[:, None] * sw_f, mask=mb, other=0.0)
        else:
            al = ah
            bl = bh
        acc = _dot3(ah, al, bh, bl, acc, SPLIT, SPLIT, False, False)
    tl.store(O + r64[:, None] * D + rn[None, :], acc.to(O.dtype.element_ty),
             mask=mrow[:, None] & mn[None, :])


@triton.jit
def _combine_kernel(O, INV, Y, T, D, A, BT: tl.constexpr, BD: tl.constexpr):
    """y[t] = sum_a f_a[t]; the combine weight already rides on the slots."""
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    m2 = mt[:, None] & (rd < D)[None, :]
    rt64 = rt.to(tl.int64)
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(A):
        p = tl.load(INV + rt64 * A + a, mask=mt, other=0).to(tl.int64)
        acc += tl.load(O + p[:, None] * D + rd[None, :], mask=m2,
                       other=0.0).to(tl.float32)
    tl.store(Y + rt64[:, None] * D + rd[None, :], acc.to(Y.dtype.element_ty),
             mask=m2)


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------


@triton.jit
def _dgu_kernel(YH, YL, WH, WL, GU, DH, DL, PART, WS, TOK,
                TILE_E, TILE_M, OFFS, D, F, N,
                sy_t, sy_d, sw_e, sw_f, sw_d,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                NT_M, NT_N: tl.constexpr, GM: tl.constexpr,
                SPLIT: tl.constexpr, EVEN_K: tl.constexpr):
    """dact = dy @ w2[e]^T, the SwiGLU chain rule into dgu, and the per-slot
    combine-weight cotangent <act, dact>.

    <act_p, dy[t] @ w2[e]^T> equals <dy[t], f_p> but is formed from operands
    that are exact in the input dtype, so the routing-score gradient does not
    inherit the rounding of the expert output. act is rebuilt from the stored
    preactivations rather than reloaded.
    """
    tid, pn = _swizzle(tl.program_id(0), NT_M, NT_N, GM)
    e = tl.load(TILE_E + tid)
    if e < 0:
        return
    m0 = tl.load(TILE_M + tid)
    row_end = tl.load(OFFS + e + 1)
    rows = m0 + tl.arange(0, BM)
    mrow = rows < row_end
    r64 = rows.to(tl.int64)
    tok = tl.load(TOK + rows, mask=mrow, other=0).to(tl.int64)
    n0 = pn * (2 * BN)
    rn = n0 + tl.arange(0, BN)
    rn2 = rn + BN
    mn = rn < F
    mn2 = rn2 < F
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    acc2 = tl.zeros((BM, BN), dtype=tl.float32)
    ep = e.to(tl.int64) * sw_e
    yh = YH + tok[:, None] * sy_t
    yl = YL + tok[:, None] * sy_t
    wp = WH + ep + rn[:, None] * sw_f
    wp2 = WH + ep + rn2[:, None] * sw_f
    wl = WL + ep + rn[:, None] * sw_f
    wl2 = WL + ep + rn2[:, None] * sw_f
    for k0 in range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        if EVEN_K:
            ma = mrow[:, None]
            mb = mn[:, None]
            mb2 = mn2[:, None]
        else:
            mk = rk < D
            ma = mrow[:, None] & mk[None, :]
            mb = mn[:, None] & mk[None, :]
            mb2 = mn2[:, None] & mk[None, :]
        ah = tl.load(yh + rk[None, :] * sy_d, mask=ma, other=0.0)
        bh = tl.load(wp + rk[None, :] * sw_d, mask=mb, other=0.0)
        bh2 = tl.load(wp2 + rk[None, :] * sw_d, mask=mb2, other=0.0)
        if SPLIT:
            al = tl.load(yl + rk[None, :] * sy_d, mask=ma, other=0.0)
            bl = tl.load(wl + rk[None, :] * sw_d, mask=mb, other=0.0)
            bl2 = tl.load(wl2 + rk[None, :] * sw_d, mask=mb2, other=0.0)
        else:
            al = ah
            bl = bh
            bl2 = bh2
        acc = _dot3(ah, al, bh, bl, acc, SPLIT, SPLIT, False, True)
        acc2 = _dot3(ah, al, bh2, bl2, acc2, SPLIT, SPLIT, False, True)
    wv = tl.load(WS + rows, mask=mrow, other=0.0)
    p1 = _dgu_epilogue(acc, GU, DH, DL, r64, rn, mrow, mn, wv, F)
    p2 = _dgu_epilogue(acc2, GU, DH, DL, r64, rn2, mrow, mn2, wv, F)
    tl.store(PART + pn.to(tl.int64) * N + rows, p1 + p2, mask=mrow)


@triton.jit
def _dgu_epilogue(acc, GU, DH, DL, r64, rn, mrow, mn, wv, F):
    """SwiGLU chain rule for one column tile, plus its <act, dact> partial."""
    ms = mrow[:, None] & mn[None, :]
    gp = GU + r64[:, None] * (2 * F) + rn[None, :]
    g = tl.load(gp, mask=ms, other=0.0)
    u = tl.load(gp + F, mask=ms, other=0.0)
    sig = tl.sigmoid(g)
    silu = g * sig
    part = tl.sum(tl.where(ms, (silu * u) * acc, 0.0), 1)
    accw = acc * wv[:, None]
    dgate = accw * u * (sig + silu * (1.0 - sig))
    dup = accw * silu
    gh = dgate.to(tl.bfloat16)
    uh = dup.to(tl.bfloat16)
    op = r64[:, None] * (2 * F) + rn[None, :]
    tl.store(DH + op, gh, mask=ms)
    tl.store(DH + op + F, uh, mask=ms)
    tl.store(DL + op, (dgate - gh.to(tl.float32)).to(tl.bfloat16), mask=ms)
    tl.store(DL + op + F, (dup - uh.to(tl.float32)).to(tl.bfloat16), mask=ms)
    return part


@triton.jit
def _dweights_kernel(S, SEL, INV, PART, DLOG, T, E, A, N,
                     NPART, BT: tl.constexpr, BA: tl.constexpr,
                     BE: tl.constexpr):
    """normalization + sigmoid chain rule, scattered into a dense [T, E].

    The per-slot weight cotangent is finished here: the column-tile partials
    left by the dact GEMM are summed in tile order.
    """
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    ra = tl.arange(0, BA)
    mt = rt < T
    m2 = mt[:, None] & (ra < A)[None, :]
    rt64 = rt.to(tl.int64)
    sel = tl.load(SEL + rt64[:, None] * A + ra[None, :], mask=m2, other=0)
    z = tl.load(S + rt64[:, None] * E + sel, mask=m2, other=0.0)
    ssum = tl.sum(z, 1)
    w = z / ssum[:, None]
    p = tl.load(INV + rt64[:, None] * A + ra[None, :], mask=m2, other=0)
    p64 = p.to(tl.int64)
    dwa = tl.zeros((BT, BA), dtype=tl.float32)
    for j in range(NPART):
        dwa += tl.load(PART + j * N + p64, mask=m2, other=0.0)
    inner = tl.sum(tl.where(m2, dwa * w, 0.0), 1)
    dl = tl.where(m2, (dwa - inner[:, None]) / ssum[:, None] * z * (1.0 - z),
                  0.0)
    re = tl.arange(0, BE)
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    for a in range(A):
        hot = ra == a
        sel_a = tl.sum(tl.where(hot[None, :], sel, 0), 1)
        dl_a = tl.sum(tl.where(hot[None, :], dl, 0.0), 1)
        acc += tl.where(re[None, :] == sel_a[:, None], dl_a[:, None], 0.0)
    tl.store(DLOG + rt64[:, None] * E + re[None, :], acc,
             mask=mt[:, None] & (re < E)[None, :])


@triton.jit
def _dw2_kernel(AH, AL, YH, YL, TOK, OFFS, DW2, F, D,
                sy_t, sy_d, sw_e, sw_f, sw_d,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                NT_M: tl.constexpr, SPLIT_Y: tl.constexpr):
    """dw2[e] = (w * act)_e^T @ dy_e over the expert's slot range."""
    pid = tl.program_id(0)
    e = tl.program_id(1)
    pm = pid % NT_M
    pn = pid // NT_M
    k0 = tl.load(OFFS + e)
    k1 = tl.load(OFFS + e + 1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < F
    mn = rn < D
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(k0, k1, BK):
        rk = k + tl.arange(0, BK)
        mk = rk < k1
        ma = mk[:, None] & mm[None, :]
        mb = mk[:, None] & mn[None, :]
        ap = rk.to(tl.int64)[:, None] * F + rm[None, :]
        ah = tl.load(AH + ap, mask=ma, other=0.0)
        al = tl.load(AL + ap, mask=ma, other=0.0)
        tok = tl.load(TOK + rk, mask=mk, other=0).to(tl.int64)
        yp = tok[:, None] * sy_t + rn[None, :] * sy_d
        bh = tl.load(YH + yp, mask=mb, other=0.0)
        bl = tl.load(YL + yp, mask=mb, other=0.0) if SPLIT_Y else bh
        acc = _dot3(ah, al, bh, bl, acc, True, SPLIT_Y, True, False)
    tl.store(DW2 + e.to(tl.int64) * sw_e + rm[:, None] * sw_f
             + rn[None, :] * sw_d, acc.to(DW2.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _dxg_kernel(DH, DL, WH, WL, DXG, TILE_E, TILE_M, OFFS, D, F,
                sw_e, sw_d, sw_f,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                NT_M, NT_N: tl.constexpr, GM: tl.constexpr,
                SPLIT: tl.constexpr, EVEN_K: tl.constexpr):
    """per-slot input gradient: dgu @ w1[e]^T."""
    tid, pn = _swizzle(tl.program_id(0), NT_M, NT_N, GM)
    e = tl.load(TILE_E + tid)
    if e < 0:
        return
    m0 = tl.load(TILE_M + tid)
    row_end = tl.load(OFFS + e + 1)
    rows = m0 + tl.arange(0, BM)
    mrow = rows < row_end
    r64 = rows.to(tl.int64)
    rn = pn * BN + tl.arange(0, BN)
    mn = rn < D
    K = 2 * F
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    ap = r64[:, None] * K
    wp = WH + e.to(tl.int64) * sw_e + rn[:, None] * sw_d
    wl = WL + e.to(tl.int64) * sw_e + rn[:, None] * sw_d
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        if EVEN_K:
            ma = mrow[:, None]
            mb = mn[:, None]
        else:
            mk = rk < K
            ma = mrow[:, None] & mk[None, :]
            mb = mn[:, None] & mk[None, :]
        ah = tl.load(DH + ap + rk[None, :], mask=ma, other=0.0)
        bh = tl.load(wp + rk[None, :] * sw_f, mask=mb, other=0.0)
        if SPLIT:
            al = tl.load(DL + ap + rk[None, :], mask=ma, other=0.0)
            bl = tl.load(wl + rk[None, :] * sw_f, mask=mb, other=0.0)
        else:
            al = ah
            bl = bh
        acc = _dot3(ah, al, bh, bl, acc, SPLIT, SPLIT, False, True)
    tl.store(DXG + r64[:, None] * D + rn[None, :],
             acc.to(DXG.dtype.element_ty), mask=mrow[:, None] & mn[None, :])


@triton.jit
def _dw1_kernel(XH, XL, DH, DL, TOK, OFFS, DW1, D, F,
                sx_t, sx_d, sw_e, sw_d, sw_f,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                NT_M: tl.constexpr, SPLIT_X: tl.constexpr):
    """dw1[e] = x_e^T @ dgu_e over the expert's slot range."""
    pid = tl.program_id(0)
    e = tl.program_id(1)
    pm = pid % NT_M
    pn = pid // NT_M
    k0 = tl.load(OFFS + e)
    k1 = tl.load(OFFS + e + 1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < D
    mn = rn < 2 * F
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(k0, k1, BK):
        rk = k + tl.arange(0, BK)
        mk = rk < k1
        tok = tl.load(TOK + rk, mask=mk, other=0).to(tl.int64)
        xp = tok[:, None] * sx_t + rm[None, :] * sx_d
        ma = mk[:, None] & mm[None, :]
        mb = mk[:, None] & mn[None, :]
        ah = tl.load(XH + xp, mask=ma, other=0.0)
        al = tl.load(XL + xp, mask=ma, other=0.0) if SPLIT_X else ah
        dp = rk.to(tl.int64)[:, None] * (2 * F) + rn[None, :]
        bh = tl.load(DH + dp, mask=mb, other=0.0)
        bl = tl.load(DL + dp, mask=mb, other=0.0)
        acc = _dot3(ah, al, bh, bl, acc, SPLIT_X, True, True, False)
    tl.store(DW1 + e.to(tl.int64) * sw_e + rm[:, None] * sw_d
             + rn[None, :] * sw_f, acc.to(DW1.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _dx_kernel(DXG, INV, DLOG, RH, RL, DX, PART, DRW, T, D, E, A, NSPLIT,
               sr_d, sr_e, sd_t, sd_d, sw_d, sw_e,
               BT: tl.constexpr, BD: tl.constexpr, BE: tl.constexpr,
               SPLIT_W: tl.constexpr):
    """dx[t] = sum_a dxg[slot(t, a)] + dlogit[t] @ router_w^T.

    The first row of programs also finishes drouter_w by summing its token
    slice partials in slice order.
    """
    if tl.program_id(0) == 0:
        rm = tl.program_id(1) * BD + tl.arange(0, BD)
        rn = tl.arange(0, BE)
        mm = rm < D
        acc = tl.zeros((BD, BE), dtype=tl.float32)
        for j in range(NSPLIT):
            acc += tl.load(PART + j.to(tl.int64) * (D * BE)
                           + rm[:, None] * BE + rn[None, :],
                           mask=mm[:, None], other=0.0)
        tl.store(DRW + rm[:, None] * sw_d + rn[None, :] * sw_e,
                 acc.to(DRW.dtype.element_ty),
                 mask=mm[:, None] & (rn < E)[None, :])
    rt = tl.program_id(0) * BT + tl.arange(0, BT)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    mt = rt < T
    md = rd < D
    m2 = mt[:, None] & md[None, :]
    rt64 = rt.to(tl.int64)
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    for a in range(A):
        p = tl.load(INV + rt64 * A + a, mask=mt, other=0).to(tl.int64)
        acc += tl.load(DXG + p[:, None] * D + rd[None, :],
                       mask=m2, other=0.0).to(tl.float32)
    re = tl.arange(0, BE)
    me = re < E
    dl = tl.load(DLOG + rt64[:, None] * E + re[None, :],
                 mask=mt[:, None] & me[None, :], other=0.0)
    dh = dl.to(tl.bfloat16)
    dlo = (dl - dh.to(tl.float32)).to(tl.bfloat16)
    rp = rd[:, None] * sr_d + re[None, :] * sr_e
    mb = md[:, None] & me[None, :]
    bh = tl.load(RH + rp, mask=mb, other=0.0)
    bl = tl.load(RL + rp, mask=mb, other=0.0) if SPLIT_W else bh
    acc = _dot3(dh, dlo, bh, bl, acc, True, SPLIT_W, False, True)
    tl.store(DX + rt64[:, None] * sd_t + rd[None, :] * sd_d,
             acc.to(DX.dtype.element_ty), mask=m2)


@triton.jit
def _drw_kernel(XH, XL, DLOG, PART, T, D, E,
                sx_t, sx_d,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                NSPLIT: tl.constexpr, SPLIT_X: tl.constexpr):
    """drouter_w = x^T @ dlogit, over a fixed token slice per program."""
    pm = tl.program_id(0)
    ks = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    mm = rm < D
    mn = rn < E
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    chunk = tl.cdiv(tl.cdiv(T, BK), NSPLIT) * BK
    lo = ks * chunk
    hi = min(lo + chunk, T)
    for k0 in range(lo, hi, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < hi
        xp = rk.to(tl.int64)[:, None] * sx_t + rm[None, :] * sx_d
        ma = mk[:, None] & mm[None, :]
        ah = tl.load(XH + xp, mask=ma, other=0.0)
        al = tl.load(XL + xp, mask=ma, other=0.0) if SPLIT_X else ah
        b = tl.load(DLOG + rk.to(tl.int64)[:, None] * E + rn[None, :],
                    mask=mk[:, None] & mn[None, :], other=0.0)
        bh = b.to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        acc = _dot3(ah, al, bh, bl, acc, SPLIT_X, True, True, False)
    tl.store(PART + ks.to(tl.int64) * (D * BN) + rm[:, None] * BN
             + rn[None, :], acc, mask=mm[:, None])


# ---------------------------------------------------------------------------
# launch policy: tiles follow the GEMM extents and the number of dots each
# operand pair expands to
# ---------------------------------------------------------------------------


_SMEM = 227 * 1024
_NSM = 132


# Base tiles per GEMM, measured on this device; the extents clamp them below.
_BASE = {
    "gemm1": (128, 128, 64, 8, 4),
    "gemm2": (128, 256, 64, 8, 4),
    "dgu": (64, 32, 64, 4, 4),
    "dxg": (128, 256, 64, 8, 4),
    "dw1": (128, 128, 32, 8, 5),
    "dw2": (128, 128, 32, 8, 5),
}


def _cfg(name, M, N, K, na, nb, nacc=1):
    """Tile shape, warps and stages for one grouped GEMM.

    na/nb are the number of bfloat16 operand tiles per side (2 when that
    operand rides as a high/low pair), nacc the number of accumulators.
    """
    bm, bn, bk, nw, st = _BASE[name]
    bn = min(bn, max(_pow2(N), 16))
    bm = min(bm, max(_pow2(M), 16))
    bk = min(bk, max(_pow2(K), 16))
    if bm * bn * nacc < 128 * 64:
        nw = 4
    while st > 2 and (bm * na + bn * nb * nacc) * bk * 2 * st > _SMEM:
        st -= 1
    while (bm * na + bn * nb * nacc) * bk * 2 * st > _SMEM:
        bk //= 2
    return bm, bn, bk, nw, st


def _split_pair(t, block=8192):
    """Exact bfloat16 high/low pair of a float32 tensor."""
    n = t.numel()
    hi = torch.empty_like(t, dtype=torch.bfloat16)
    lo = torch.empty_like(t, dtype=torch.bfloat16)
    _split_kernel[(triton.cdiv(n, block),)](t, hi, lo, n, BLOCK=block,
                                            num_warps=8)
    return hi, lo


def _meta(sorted_e, order, s, sel, tok, ws, inv, offs, T, N, E, A,
          bm, bm2, dev):
    """Slot bookkeeping plus the tile -> (expert, first slot) maps for the two
    row-tile heights the pipeline uses."""
    ntm = triton.cdiv(N, bm) + E
    ntm2 = triton.cdiv(N, bm2) + E
    tiles = torch.empty((2 * (ntm + ntm2),), device=dev, dtype=torch.int32)
    tile_e, tile_m = tiles[:ntm], tiles[ntm:2 * ntm]
    tile_e2 = tiles[2 * ntm:2 * ntm + ntm2]
    tile_m2 = tiles[2 * ntm + ntm2:]
    rounds = 1
    reach = 1
    while reach < N + 1:
        reach *= 15
        rounds += 1
    block = 256
    _meta_kernel[(triton.cdiv(N, block) + 1,)](
        sorted_e, order, s, sel, tok, ws, inv, offs, tile_e, tile_m,
        tile_e2, tile_m2, T, N, E, A, ntm, ntm2, rounds, BM=bm, BM2=bm2,
        BT=128, BA=_pow2(A), BE=_pow2(E + 1), BS=16, BLOCK=block,
        num_warps=4)
    if bm2 == bm:
        return (tile_e, tile_m, ntm), (tile_e, tile_m, ntm)
    return (tile_e, tile_m, ntm), (tile_e2, tile_m2, ntm2)


class _FusedMoe(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, router_w, router_b, w1, w2, A):
        A = int(A)
        T, D = x.shape
        E = router_w.shape[1]
        F = w1.shape[2] // 2
        N = T * A
        dev = x.device
        f32 = x.dtype == torch.float32
        nsp = 2 if f32 else 1

        if f32:
            xh, xl = _split_pair(x.contiguous())
            rh, rl = _split_pair(router_w.contiguous())
            w1h, w1l = _split_pair(w1)
            w2h, w2l = _split_pair(w2)
        else:
            xh = xl = x
            rh = rl = router_w
            w1h = w1l = w1
            w2h = w2l = w2

        s = torch.empty((T, E), device=dev, dtype=torch.float32)
        bep = _pow2(E, 16)
        keys = torch.empty((T, bep), device=dev, dtype=torch.int64)
        bt = max(1, min(64, 8192 // bep))
        _router_kernel[(triton.cdiv(T, bt),)](
            xh, xl, rh, rl, router_b, s, keys, T, D, E,
            xh.stride(0), xh.stride(1), rh.stride(0), rh.stride(1),
            BT=bt, BE=bep, BK=64, SPLIT=f32, num_warps=4, num_stages=3)

        sel = torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices
        sorted_e, order = torch.sort(sel.view(-1).to(torch.int32),
                                     stable=True)

        idx = torch.empty((2 * N + E + 1,), device=dev, dtype=torch.int32)
        tok, inv, offs = idx[:N], idx[N:2 * N], idx[2 * N:]
        ws = torch.empty((N,), device=dev, dtype=torch.float32)
        bm, bn1, bk1, nw1, st1 = _cfg("gemm1", N, F, D, nsp, nsp, nacc=2)
        bmg = _cfg("dgu", N, F, D, nsp, nsp, nacc=2)[0]
        map1, mapg = _meta(sorted_e, order, s, sel, tok, ws, inv, offs,
                           T, N, E, A, bm, bmg, dev)
        tile_e, tile_m, ntm = map1

        gu = torch.empty((N, 2 * F), device=dev, dtype=torch.float32)
        ah = torch.empty((N, F), device=dev, dtype=torch.bfloat16)
        al = torch.empty((N, F), device=dev, dtype=torch.bfloat16)
        ntn = triton.cdiv(F, bn1)
        _gemm1_kernel[(ntm * ntn,)](
            xh, xl, w1h, w1l, gu, ah, al, ws, tile_e, tile_m, offs, tok, D, F,
            xh.stride(0), xh.stride(1),
            w1h.stride(0), w1h.stride(1), w1h.stride(2),
            BM=bm, BN=bn1, BK=bk1, NT_M=ntm, NT_N=ntn, GM=8, SPLIT=f32,
            EVEN_K=(D % bk1 == 0), num_warps=nw1, num_stages=st1)

        _, bn2, bk2, nw2, st2 = _cfg("gemm2", N, D, F, nsp, nsp)
        o = torch.empty((N, D), device=dev, dtype=x.dtype)
        ntn2 = triton.cdiv(D, bn2)
        _gemm2_kernel[(ntm * ntn2,)](
            ah, al, w2h, w2l, o, tile_e, tile_m, offs, F, D,
            w2h.stride(0), w2h.stride(1), w2h.stride(2),
            BM=bm, BN=bn2, BK=bk2, NT_M=ntm, NT_N=ntn2, GM=8, SPLIT=f32,
            EVEN_K=(F % bk2 == 0), num_warps=nw2, num_stages=st2)

        y = torch.empty((T, D), device=dev, dtype=x.dtype)
        bd = min(128, _pow2(D))
        _combine_kernel[(triton.cdiv(T, 32), triton.cdiv(D, bd))](
            o, inv, y, T, D, A, BT=32, BD=bd, num_warps=4)

        ctx.save_for_backward(xh, xl, rh, rl, w1h, w1l, w2h, w2l, s, sel,
                              tok, ws, inv, offs, sorted_e, gu, ah, al)
        ctx.maps = (map1, mapg)
        ctx.meta = (T, D, E, F, A, N, f32, nsp, bm, x.dtype,
                    router_w.dtype, w1.dtype, w2.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        (xh, xl, rh, rl, w1h, w1l, w2h, w2l, s, sel, tok, ws, inv, offs,
         sorted_e, gu, ah, al) = ctx.saved_tensors
        (T, D, E, F, A, N, f32, nsp, bm, xdt, rdt, w1dt, w2dt) = ctx.meta
        dev = xh.device
        map1, mapg = ctx.maps
        npere = max(N // max(E, 1), 1)
        bd = min(128, _pow2(D))
        dy = dy.contiguous()
        if f32:
            yh, yl = _split_pair(dy)
        else:
            yh = yl = dy

        bmg, bng, bkg, nwg, stg = _cfg("dgu", N, F, D, nsp, nsp, nacc=2)
        teg, tmg, ntmg = mapg
        ntn = triton.cdiv(F, 2 * bng)
        dh = torch.empty((N, 2 * F), device=dev, dtype=torch.bfloat16)
        dl = torch.empty((N, 2 * F), device=dev, dtype=torch.bfloat16)
        part = torch.empty((ntn, N), device=dev, dtype=torch.float32)
        _dgu_kernel[(ntmg * ntn,)](
            yh, yl, w2h, w2l, gu, dh, dl, part, ws, tok, teg, tmg, offs,
            D, F, N, yh.stride(0), yh.stride(1),
            w2h.stride(0), w2h.stride(1), w2h.stride(2),
            BM=bmg, BN=bng, BK=bkg, NT_M=ntmg, NT_N=ntn, GM=8, SPLIT=f32,
            EVEN_K=(D % bkg == 0), num_warps=nwg, num_stages=stg)

        dlog = torch.empty((T, E), device=dev, dtype=torch.float32)
        ba = _pow2(A)
        be = _pow2(E)
        bt = max(1, min(32, 4096 // max(ba, be)))
        _dweights_kernel[(triton.cdiv(T, bt),)](
            s, sel, inv, part, dlog, T, E, A, N, ntn, BT=bt, BA=ba, BE=be,
            num_warps=4)

        bmw, bnw, bkw, nww, stw = _cfg("dw2", F, D, npere, 2, nsp)
        dw2 = torch.empty((E, F, D), device=dev, dtype=w2dt)
        ntmw = triton.cdiv(F, bmw)
        _dw2_kernel[(ntmw * triton.cdiv(D, bnw), E)](
            ah, al, yh, yl, tok, offs, dw2, F, D,
            yh.stride(0), yh.stride(1),
            dw2.stride(0), dw2.stride(1), dw2.stride(2),
            BM=bmw, BN=bnw, BK=bkw, NT_M=ntmw, SPLIT_Y=f32,
            num_warps=nww, num_stages=stw)

        _, bnx, bkx, nwx, stx = _cfg("dxg", N, D, 2 * F, nsp, nsp)
        tex, tmx, ntmx = map1
        dxg = torch.empty((N, D), device=dev,
                          dtype=torch.float32 if f32 else torch.bfloat16)
        ntnx = triton.cdiv(D, bnx)
        _dxg_kernel[(ntmx * ntnx,)](
            dh, dl, w1h, w1l, dxg, tex, tmx, offs, D, F,
            w1h.stride(0), w1h.stride(1), w1h.stride(2),
            BM=bm, BN=bnx, BK=bkx, NT_M=ntmx, NT_N=ntnx, GM=8, SPLIT=f32,
            EVEN_K=((2 * F) % bkx == 0), num_warps=nwx, num_stages=stx)

        bm1, bn1, bk1, nw1, st1 = _cfg("dw1", D, 2 * F, npere, nsp, 2)
        dw1 = torch.empty((E, D, 2 * F), device=dev, dtype=w1dt)
        ntm1 = triton.cdiv(D, bm1)
        _dw1_kernel[(ntm1 * triton.cdiv(2 * F, bn1), E)](
            xh, xl, dh, dl, tok, offs, dw1, D, F,
            xh.stride(0), xh.stride(1),
            dw1.stride(0), dw1.stride(1), dw1.stride(2),
            BM=bm1, BN=bn1, BK=bk1, NT_M=ntm1, SPLIT_X=f32,
            num_warps=nw1, num_stages=st1)

        bnr = _pow2(E, 16)
        bmr, bkr, str_ = 64, 64, 3
        while (bmr * nsp + bnr * 2) * bkr * 2 * str_ > _SMEM:
            bkr //= 2
        nks = max(1, min(triton.cdiv(T, bkr),
                         _NSM // max(1, triton.cdiv(D, bmr))))
        rpart = torch.empty((nks, D, bnr), device=dev, dtype=torch.float32)
        _drw_kernel[(triton.cdiv(D, bmr), nks)](
            xh, xl, dlog, rpart, T, D, E, xh.stride(0), xh.stride(1),
            BM=bmr, BN=bnr, BK=bkr, NSPLIT=nks, SPLIT_X=f32,
            num_warps=4, num_stages=str_)

        dx = torch.empty((T, D), device=dev, dtype=xdt)
        drw = torch.empty((D, E), device=dev, dtype=rdt)
        _dx_kernel[(triton.cdiv(T, 32), triton.cdiv(D, bd))](
            dxg, inv, dlog, rh, rl, dx, rpart, drw, T, D, E, A, nks,
            rh.stride(0), rh.stride(1), dx.stride(0), dx.stride(1),
            drw.stride(0), drw.stride(1),
            BT=32, BD=bd, BE=bnr, SPLIT_W=f32, num_warps=4)

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
