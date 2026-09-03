"""Private oracle -- from-scratch Triton segmented descending top-k.

UNVERIFIED IN THIS FOLDER: the radix-select core is the calibrated
`categories/topk` private oracle (GPU-proven there), applied to the flattened
[R*G, S] segment view with LOCAL-column keys; the global index is recovered as
s*S + local afterwards. This copy has NOT been re-run through the forge
golden/controls path against this folder's bundle; its headroom / full-reward
claim is pending GPU calibration.

Never shipped to an agent. Establishes that full reward is reachable through
every live gate with no library selection/sorting call anywhere.

Algorithm (deterministic, no value-dependent branching):

  Flatten x[R, N] -> xv[M, S] with M = R*G, G = N // S (contiguous: flat
  segment m = r*G + g is exactly segment g of row r). Per flat segment,

    key(m, i) = (ordered_bits32(float32(xv[m, i])) << 21) | (2097151 - i)

  is unique over the LOCAL column i in [0, S), so (value desc, local index asc)
  collapses to descending key order and ties never exist.

  1. Radix descent (7 passes of 8 bits): per-segment 256-bin histograms then a
     digit-select kernel -> prefix = k-th largest key per flat segment.
  2. Count + compact: per-(segment, chunk) counts of keys >= T (exactly k),
     exclusive offsets, ordered compaction of the k candidate keys.
  3. Counting-rank emission: rank within the k candidates, then bit-exact value
     + int64 LOCAL index scatter-stored at the final rank.

  Then globalise: idx = local + s*S, reshaped to [R, G, k]. Supports segment S
  up to 2**21, any 1 <= k <= S, 64-bit offsets (R*N may exceed 2**31).
"""

import torch
import triton
import triton.language as tl

CHUNK = 4096
IDX_BITS = 21
IDX_MASK = (1 << IDX_BITS) - 1   # 2097151
RANK_TI = 64
RANK_TJ = 64


@triton.jit
def _key_of(xf, col):
    """Unique key for (value desc, LOCAL index asc)."""
    xf = tl.where(xf == 0.0, 0.0, xf)                 # -0.0 -> +0.0
    b = xf.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
    return (u << 21) | (2097151 - col)


@triton.jit
def _hist_kernel(X, PREFIX, H, S, shift: tl.constexpr, CHUNK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)                 # flat segment
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < S
    xf = tl.load(X + r * S + offs, mask=m, other=0.0).to(tl.float32)
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
def _count_ge_kernel(X, T, CNT, S, nch, CHUNK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < S
    xf = tl.load(X + r * S + offs, mask=m, other=0.0).to(tl.float32)
    key = _key_of(xf, offs)
    t = tl.load(T + r)
    sel = m & (key >= t)
    tl.store(CNT + r * nch + c, tl.sum(sel.to(tl.int32)))


@triton.jit
def _compact_kernel(X, T, OFFS, CAND, S, k, nch, CHUNK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < S
    xf = tl.load(X + r * S + offs, mask=m, other=0.0).to(tl.float32)
    key = _key_of(xf, offs)
    t = tl.load(T + r)
    sel = m & (key >= t)
    base = tl.load(OFFS + r * nch + c)
    pos = base + tl.cumsum(sel.to(tl.int32), 0) - 1
    tl.store(CAND + r * k + pos.to(tl.int64), key, mask=sel)


@triton.jit
def _rank_emit_kernel(CAND, XB, VB, IDX, S, k, TI: tl.constexpr, TJ: tl.constexpr):
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
    local = 2097151 - (ki & 2097151)                  # local column in [0, S)
    vb = tl.load(XB + r * S + local, mask=mi, other=0)
    tl.store(IDX + r * k + rank.to(tl.int64), local, mask=mi)
    tl.store(VB + r * k + rank.to(tl.int64), vb, mask=mi)


def _bits_view(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def _seg_impl(x: torch.Tensor, k: int, S: int):
    R, N = x.shape
    if S <= 0 or N % S != 0:
        raise ValueError(f"segment={S} must be positive and divide N={N}")
    if not 1 <= k <= S:
        raise ValueError(f"k={k} out of range for segment length S={S}")
    if S > IDX_MASK + 1:
        raise ValueError(f"oracle supports segment <= {IDX_MASK + 1}, got {S}")
    G = N // S
    M = R * G
    xv = x.contiguous().view(M, S)
    dev = x.device
    nch = triton.cdiv(S, CHUNK)

    prefix = torch.zeros(M, dtype=torch.int64, device=dev)
    krem = torch.full((M,), k, dtype=torch.int32, device=dev)
    hist = torch.zeros(M, 256, dtype=torch.int32, device=dev)
    for p in range(6, -1, -1):
        _hist_kernel[(M, nch)](xv, prefix, hist, S, shift=8 * p, CHUNK=CHUNK)
        _select_digit_kernel[(M,)](hist, prefix, krem)

    cnt = torch.empty(M, nch, dtype=torch.int32, device=dev)
    _count_ge_kernel[(M, nch)](xv, prefix, cnt, S, nch, CHUNK=CHUNK)
    offs = (torch.cumsum(cnt, dim=1) - cnt).to(torch.int32).contiguous()
    cand = torch.empty(M, k, dtype=torch.int64, device=dev)
    _compact_kernel[(M, nch)](xv, prefix, offs, cand, S, k, nch, CHUNK=CHUNK)

    values = torch.empty(M, k, dtype=x.dtype, device=dev)
    out_local = torch.empty(M, k, dtype=torch.int64, device=dev)
    _rank_emit_kernel[(M, triton.cdiv(k, RANK_TI))](
        cand, _bits_view(xv), _bits_view(values), out_local, S, k,
        TI=RANK_TI, TJ=RANK_TJ)

    # Globalise the local columns and reshape to [R, G, k].
    local = out_local.view(R, G, k)
    seg_off = (torch.arange(G, device=dev, dtype=torch.int64) * S).view(1, G, 1)
    idx = local + seg_off
    vals = values.view(R, G, k)
    return vals, idx


def topk_segmented(
    x: torch.Tensor,     # [R, N] float32 or bfloat16, NaN-free
    k: int,
    segment: int,        # S; divides N; 1 <= k <= S
) -> tuple:              # (values [R, G, k] in x.dtype, indices [R, G, k] int64)
    return _seg_impl(x, int(k), int(segment))
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 4f246aa291e433a8e030563862493822331fb1392d77d71f86ad3e7c8f1cfe27
# FORGE-CANARY-SLOT-1 996489b2cafa7a5b8fe7e241faa945ee167ffbc6692ef43396b04ef87eb6d764
# FORGE-CANARY-SLOT-2 869d801445f7bc24159c0b502b1cd254bd2da38ae8b4574cf1751281f4ba7413
# FORGE-CANARY-SLOT-3 7803ba2d4ce892e0da20b10fa5754e1928e0caad133f596c2147ea29f1f338bd
# FORGE-CANARY-END
