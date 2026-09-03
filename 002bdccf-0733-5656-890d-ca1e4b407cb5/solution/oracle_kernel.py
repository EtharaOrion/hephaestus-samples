"""Private oracle -- from-scratch Triton RAGGED top-k with variable-length rows.

UNVERIFIED IN THIS FOLDER: the radix-select core is the calibrated
`categories/topk` design (GPU-proven there), adapted here for a ragged flat
input x[NNZ] + seg_offsets[R+1]. Each program reads only its row's slice
[seg_offsets[r], seg_offsets[r+1]) -- no pad-to-max sentinel, no wasted
bandwidth on the length skew that the anchor pays. This copy has NOT been
re-run through the forge golden/controls path against this folder's bundle;
its headroom / full-reward attainment claim is PENDING GPU CALIBRATION at the
golden stage. No GPU was available at authoring time (per client brief).

Never shipped to an agent. Establishes that full reward is reachable through
every live gate (G1 symbol scan, exact G2 semantics, G3 timing, G4 adoption)
with no library selection/sorting call anywhere and no pad-to-max preprocess.

Algorithm (deterministic, no value-dependent branching):

  For each row r, key(i) = (ordered_bits32(float32(x[lo+i]))<<24)|(16777215-i)
  is unique over local i in [0, L_r), so (value desc, local index asc)
  collapses to descending key order. Global index = local + lo. IDX_BITS=24
  supports per-row length up to 2**24 = 16,777,216 (max Lmax = 2,100,000).

  1. Radix descent (7 passes of 8 bits): per-row 256-bin histograms + a
     digit-select kernel narrow a prefix. Programs are launched over
     (R, ceil(Lmax/CHUNK)); chunks past L_r are dropped by the row-length
     mask, so short rows pay only for the chunks they occupy.
  2. Count + compact: per-(row, chunk) counts of keys >= T, exclusive
     offsets, ordered compaction of k candidate keys per row.
  3. Counting-rank emission: rank within the k candidates, then values and
     GLOBAL indices are scatter-stored at their final rank. Values are
     moved as raw bits (int16/int32 views), a bit-exact gather of the input.

  64-bit offsets throughout: c6 has R=1024 * Lmax=2100000, flat NNZ can
  exceed 2**31.
"""

import torch
import triton
import triton.language as tl

CHUNK = 4096
IDX_BITS = 24
IDX_MASK = (1 << IDX_BITS) - 1  # 16777215
RANK_TI = 64
RANK_TJ = 64


@triton.jit
def _key_of(xf, col):
    """Order-preserving unique int64 key for (value desc, LOCAL index asc)."""
    xf = tl.where(xf == 0.0, 0.0, xf)  # -0.0 -> +0.0
    b = xf.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
    return (u << 24) | (16777215 - col)


@triton.jit
def _hist_kernel(X, SEG, LEN, PREFIX, H, shift: tl.constexpr, CHUNK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    L = tl.load(LEN + r)
    lo = tl.load(SEG + r)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < L
    xf = tl.load(X + lo + offs, mask=m, other=0.0).to(tl.float32)
    key = _key_of(xf, offs)
    prefix = tl.load(PREFIX + r)
    active = m & ((key >> (shift + 8)) == prefix)
    digit = tl.where(active, ((key >> shift) & 255).to(tl.int32), 0)
    h = tl.histogram(digit, 256)
    n_inactive = CHUNK - tl.sum(active.to(tl.int32))
    d = tl.arange(0, 256)
    h = h - tl.where(d == 0, n_inactive, 0)
    tl.atomic_add(H + r * 256 + d, h)


@triton.jit
def _select_digit_kernel(H, PREFIX, KREM):
    r = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 256)
    cnt = tl.load(H + r * 256 + d)
    s = tl.flip(tl.cumsum(tl.flip(cnt, 0), 0), 0)
    kr = tl.load(KREM + r)
    cond = (s >= kr) & ((s - cnt) < kr)
    b = tl.sum(tl.where(cond, d, 0))
    gt = tl.sum(tl.where(cond, s - cnt, 0))
    p = tl.load(PREFIX + r)
    tl.store(PREFIX + r, (p << 8) | b.to(tl.int64))
    tl.store(KREM + r, kr - gt)
    tl.store(H + r * 256 + d, tl.zeros_like(cnt))


@triton.jit
def _count_ge_kernel(X, SEG, LEN, T, CNT, nch, CHUNK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    L = tl.load(LEN + r)
    lo = tl.load(SEG + r)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < L
    xf = tl.load(X + lo + offs, mask=m, other=0.0).to(tl.float32)
    key = _key_of(xf, offs)
    t = tl.load(T + r)
    sel = m & (key >= t)
    tl.store(CNT + r * nch + c, tl.sum(sel.to(tl.int32)))


@triton.jit
def _compact_kernel(X, SEG, LEN, T, OFFS, CAND, k, nch, CHUNK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    L = tl.load(LEN + r)
    lo = tl.load(SEG + r)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < L
    xf = tl.load(X + lo + offs, mask=m, other=0.0).to(tl.float32)
    key = _key_of(xf, offs)
    t = tl.load(T + r)
    sel = m & (key >= t)
    base = tl.load(OFFS + r * nch + c)
    pos = base + tl.cumsum(sel.to(tl.int32), 0) - 1
    tl.store(CAND + r * k + pos.to(tl.int64), key, mask=sel)


@triton.jit
def _rank_emit_kernel(CAND, XB, SEG, VB, IDX, k, TI: tl.constexpr, TJ: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    it = tl.program_id(1)
    lo = tl.load(SEG + r)
    i = it * TI + tl.arange(0, TI)
    mi = i < k
    ki = tl.load(CAND + r * k + i.to(tl.int64), mask=mi, other=-1)
    rank = tl.zeros((TI,), dtype=tl.int32)
    for j0 in range(0, k, TJ):
        j = j0 + tl.arange(0, TJ)
        kj = tl.load(CAND + r * k + j.to(tl.int64), mask=j < k, other=-1)
        rank += tl.sum((kj[None, :] > ki[:, None]).to(tl.int32), axis=1)
    local = 16777215 - (ki & 16777215)
    glob = local + lo
    vb = tl.load(XB + glob, mask=mi, other=0)
    tl.store(IDX + r * k + rank.to(tl.int64), glob, mask=mi)
    tl.store(VB + r * k + rank.to(tl.int64), vb, mask=mi)


def _bits_view(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def _ragged_impl(x: torch.Tensor, seg_offsets: torch.Tensor, k: int):
    R = int(seg_offsets.shape[0]) - 1
    lens = (seg_offsets[1:] - seg_offsets[:-1]).contiguous()
    Lmax = int(lens.max().item())
    Lmin = int(lens.min().item())
    if not 1 <= k <= Lmin:
        raise ValueError(f"k={k} out of range for min row length {Lmin}")
    if Lmax > IDX_MASK + 1:
        raise ValueError(
            f"oracle supports per-row length <= {IDX_MASK + 1}, got {Lmax}"
        )
    x = x.contiguous()
    dev = x.device
    nch = triton.cdiv(Lmax, CHUNK)

    seg_lo = seg_offsets[:-1].contiguous()
    prefix = torch.zeros(R, dtype=torch.int64, device=dev)
    krem = torch.full((R,), k, dtype=torch.int32, device=dev)
    hist = torch.zeros(R, 256, dtype=torch.int32, device=dev)
    for p in range(6, -1, -1):
        _hist_kernel[(R, nch)](x, seg_lo, lens, prefix, hist, shift=8 * p, CHUNK=CHUNK)
        _select_digit_kernel[(R,)](hist, prefix, krem)

    cnt = torch.empty(R, nch, dtype=torch.int32, device=dev)
    _count_ge_kernel[(R, nch)](x, seg_lo, lens, prefix, cnt, nch, CHUNK=CHUNK)
    offs = (torch.cumsum(cnt, dim=1) - cnt).to(torch.int32).contiguous()
    cand = torch.empty(R, k, dtype=torch.int64, device=dev)
    _compact_kernel[(R, nch)](x, seg_lo, lens, prefix, offs, cand, k, nch, CHUNK=CHUNK)

    values = torch.empty(R, k, dtype=x.dtype, device=dev)
    out_idx = torch.empty(R, k, dtype=torch.int64, device=dev)
    _rank_emit_kernel[(R, triton.cdiv(k, RANK_TI))](
        cand,
        _bits_view(x),
        seg_lo,
        _bits_view(values),
        out_idx,
        k,
        TI=RANK_TI,
        TJ=RANK_TJ,
    )
    return values, out_idx


def topk_ragged(
    x: torch.Tensor,  # [NNZ] float32 or bfloat16, NaN-free
    seg_offsets: torch.Tensor,  # [R+1] int64 monotone, seg[0]=0, seg[-1]=NNZ
    k: int,
) -> tuple:  # (values [R, k] in x.dtype, indices [R, k] int64 GLOBAL)
    return _ragged_impl(x, seg_offsets, int(k))
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 6c929f40021012d3fc5347a991e0c6fb274ffbfc9f7f24122fc4a34b5534855c
# FORGE-CANARY-SLOT-1 4ff9f70f7f0cf0489a84a05379a2b13c29097216dfa01de2674a3043e0417b6d
# FORGE-CANARY-SLOT-2 bc32f0dbf9a92764528e598342a9ecd2c09c651984b8c9737ba8c4a4e0aba4fc
# FORGE-CANARY-SLOT-3 ed11e1f65bf47fbebf39521f6b503c8cf022e8fb15ff3fe355377d4750f1f106
# FORGE-CANARY-END
