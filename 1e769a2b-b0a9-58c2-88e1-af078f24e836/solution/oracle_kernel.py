"""Private oracle -- from-scratch Triton row-wise top-k over PRIME N.

UNVERIFIED IN THIS FOLDER: the radix-select core mirrors the calibrated
`categories/topk` private oracle (GPU-proven there). This copy has NOT been
re-run through the forge golden/controls path against this folder's bundle;
its headroom / full-reward attainment claim is PENDING GPU CALIBRATION at the
golden stage.

Never shipped to an agent. Establishes that full reward is reachable through
every live gate (G1 symbol scan, exact G2 semantics, G3 timing, G4 adoption)
with no library selection or sorting call anywhere. Handles PRIME N up to
2**21 = 2097152 (covers every graded and correctness shape); a prime N never
divides a common SM tile, which is the whole point of this variant.

Algorithm (deterministic, no value-dependent branching):

  key(r, i) = (ordered_bits32(float32(x[r, i])) << 21) | (2097151 - i)

  Every key is unique (index embedded), so (value desc, index asc) collapses
  to plain descending key order and ties never exist at any stage. -0.0 is
  canonicalised to +0.0 before the bit transform so both zeros form one tie
  group ordered by index, exactly as the reference defines.

  1. Radix descent, 7 passes of 8 bits over the 56-bit key space: per-row
     256-bin histograms (tl.histogram + int32 atomic adds, order-free sums,
     deterministic), then a per-row digit-select kernel walks the suffix
     counts and narrows the prefix. After the last pass the prefix IS the
     k-th largest key per row, exactly.
  2. Count + compact: per-(row, chunk) counts of keys >= T (exactly k per
     row, uniqueness again), exclusive chunk offsets, then an ordered
     compaction writes the k candidate keys per row.
  3. Counting-rank emission: rank(i) = #{j: key_j > key_i} within the k
     candidates (tiled pairwise comparison), then values and indices are
     scatter-stored at their final rank. Values are moved as raw bits
     (int16/int32 views), so the output is a bit-exact gather of the input.

  64-bit offsets throughout: one hidden input (c6: R=2500, N=1000003)
  exceeds 2**31 elements.
"""

import torch
import triton
import triton.language as tl

CHUNK = 4096
IDX_BITS = 21
IDX_MASK = (1 << IDX_BITS) - 1  # 2097151
RANK_TI = 64
RANK_TJ = 64


@triton.jit
def _key_of(xf, col):
    """Order-preserving unique int64 key for (value desc, index asc)."""
    xf = tl.where(xf == 0.0, 0.0, xf)  # -0.0 -> +0.0
    b = xf.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
    return (u << 21) | (2097151 - col)


@triton.jit
def _hist_kernel(X, PREFIX, H, N, shift: tl.constexpr, CHUNK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < N
    xf = tl.load(X + r * N + offs, mask=m, other=0.0).to(tl.float32)
    key = _key_of(xf, offs)
    prefix = tl.load(PREFIX + r)
    active = m & ((key >> (shift + 8)) == prefix)
    # tl.histogram's out-of-range policy varies with num_bins on this triton
    # build; keep every input in range by mapping inactive lanes to bin 0,
    # then subtract that count out. Exact under any policy.
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
    s = tl.flip(tl.cumsum(tl.flip(cnt, 0), 0), 0)  # count(digit >= d)
    kr = tl.load(KREM + r)
    # s is non-increasing in d, so exactly one bin satisfies both sides.
    cond = (s >= kr) & ((s - cnt) < kr)
    b = tl.sum(tl.where(cond, d, 0))
    gt = tl.sum(tl.where(cond, s - cnt, 0))
    p = tl.load(PREFIX + r)
    tl.store(PREFIX + r, (p << 8) | b.to(tl.int64))
    tl.store(KREM + r, kr - gt)
    tl.store(H + r * 256 + d, tl.zeros_like(cnt))  # zeroed for next pass


@triton.jit
def _count_ge_kernel(X, T, CNT, N, nch, CHUNK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < N
    xf = tl.load(X + r * N + offs, mask=m, other=0.0).to(tl.float32)
    key = _key_of(xf, offs)
    t = tl.load(T + r)
    sel = m & (key >= t)
    tl.store(CNT + r * nch + c, tl.sum(sel.to(tl.int32)))


@triton.jit
def _compact_kernel(X, T, OFFS, CAND, N, k, nch, CHUNK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < N
    xf = tl.load(X + r * N + offs, mask=m, other=0.0).to(tl.float32)
    key = _key_of(xf, offs)
    t = tl.load(T + r)
    sel = m & (key >= t)
    base = tl.load(OFFS + r * nch + c)
    pos = base + tl.cumsum(sel.to(tl.int32), 0) - 1
    tl.store(CAND + r * k + pos.to(tl.int64), key, mask=sel)


@triton.jit
def _rank_emit_kernel(CAND, XB, VB, IDX, N, k, TI: tl.constexpr, TJ: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    it = tl.program_id(1)
    i = it * TI + tl.arange(0, TI)
    mi = i < k
    ki = tl.load(CAND + r * k + i.to(tl.int64), mask=mi, other=-1)
    rank = tl.zeros((TI,), dtype=tl.int32)
    for j0 in range(0, k, TJ):
        j = j0 + tl.arange(0, TJ)
        kj = tl.load(CAND + r * k + j.to(tl.int64), mask=j < k, other=-1)
        rank += tl.sum((kj[None, :] > ki[:, None]).to(tl.int32), axis=1)
    idx = 2097151 - (ki & 2097151)
    vb = tl.load(XB + r * N + idx, mask=mi, other=0)
    tl.store(IDX + r * k + rank.to(tl.int64), idx, mask=mi)
    tl.store(VB + r * k + rank.to(tl.int64), vb, mask=mi)


def _bits_view(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def _topk_impl(x: torch.Tensor, k: int):
    R, N = x.shape
    if not 1 <= k <= N:
        raise ValueError(f"k={k} out of range for N={N}")
    if N > IDX_MASK + 1:
        raise ValueError(f"oracle supports N <= {IDX_MASK + 1}, got {N}")
    x = x.contiguous()
    dev = x.device
    nch = triton.cdiv(N, CHUNK)

    prefix = torch.zeros(R, dtype=torch.int64, device=dev)
    krem = torch.full((R,), k, dtype=torch.int32, device=dev)
    hist = torch.zeros(R, 256, dtype=torch.int32, device=dev)
    for p in range(6, -1, -1):
        _hist_kernel[(R, nch)](x, prefix, hist, N, shift=8 * p, CHUNK=CHUNK)
        _select_digit_kernel[(R,)](hist, prefix, krem)

    cnt = torch.empty(R, nch, dtype=torch.int32, device=dev)
    _count_ge_kernel[(R, nch)](x, prefix, cnt, N, nch, CHUNK=CHUNK)
    offs = (torch.cumsum(cnt, dim=1) - cnt).to(torch.int32).contiguous()
    cand = torch.empty(R, k, dtype=torch.int64, device=dev)
    _compact_kernel[(R, nch)](x, prefix, offs, cand, N, k, nch, CHUNK=CHUNK)

    values = torch.empty(R, k, dtype=x.dtype, device=dev)
    out_idx = torch.empty(R, k, dtype=torch.int64, device=dev)
    _rank_emit_kernel[(R, triton.cdiv(k, RANK_TI))](
        cand, _bits_view(x), _bits_view(values), out_idx, N, k, TI=RANK_TI, TJ=RANK_TJ
    )
    return values, out_idx


def topk_prime_row(
    x: torch.Tensor,  # [R, N] float32 or bfloat16, NaN-free, N prime
    k: int,
) -> tuple:  # (values [R, k] in x.dtype, indices [R, k] int64)
    return _topk_impl(x, int(k))
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 50b60f1e1b17fe30bb86ac7fb228cf7a7de098cb78a5cb1e72b144c1f74837c5
# FORGE-CANARY-SLOT-1 f3b6b52316a6518c57cfc9315823fc4a1c8a4ff8745408ea559892f394cf082c
# FORGE-CANARY-SLOT-2 fd2b75436b6b0b26fc14d2399d3961191945b62bf6fd4f37ff72f83f2bff7ea5
# FORGE-CANARY-SLOT-3 5f0f41d1ebd670dff3088a7c195b8a4eb6596de3385c4a1722d33e4587b15af1
# FORGE-CANARY-END
