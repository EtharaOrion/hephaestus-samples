"""Private oracle -- from-scratch Triton fused row-wise softmax + top-k.

UNVERIFIED IN THIS FOLDER: the radix-select core is the calibrated
`categories/topk` design (GPU-proven there), preceded here by a canonical
fp32 online softmax reduction pass and followed by an in-kernel probs gather.
This copy has NOT been re-run through the forge golden/controls path against
this folder's bundle; its headroom / full-reward attainment claim is PENDING
GPU CALIBRATION at the golden stage. No GPU was available at authoring time
(per client brief).

Never shipped to an agent. Establishes that full reward is reachable through
every live gate (G1 symbol scan naming the softmax spellings, exact G2 index
semantics, dtype-tolerance probs, G3 timing, G4 adoption) with no library
selection/sorting call and no library softmax call.

Algorithm (deterministic, no value-dependent branching):

  key(r, i) = (ordered_bits32(fp32(logits[r, i])) << 24) | (16777215 - i)

  Softmax is strictly increasing so top-k of logits == top-k of probs; the
  index winners never depend on the reduction order.

  1. Online row max + expsum in fp32 in a single streaming pass (log-sum-exp
     style, but returned as (m, s) rather than log(s)+m). This is the
     canonical fp32 reduction the reference defines.
  2. Radix descent, 7 passes of 8 bits on the logits composite key. Produces
     the row's k-th largest key.
  3. Count + compact of keys >= T (exactly k per row).
  4. Counting-rank + probs emission: for each of the k candidates, rank
     within them, then load the original logit, compute p = exp(logit - m)/s
     in fp32, and scatter (probs, global-index) at the final rank.

  64-bit offsets throughout: c7 has R=1024 * N=2100000, > 2**31 elements.
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
    """Order-preserving unique int64 key for (value desc, index asc)."""
    xf = tl.where(xf == 0.0, 0.0, xf)  # -0.0 -> +0.0
    b = xf.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
    return (u << 24) | (16777215 - col)


@triton.jit
def _row_max_kernel(X, M, N, CHUNK: tl.constexpr):
    """Online per-row max, fp32. One program per (row, chunk); rows are
    reduced across chunks in a follow-up python-side .max(dim=1)."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < N
    xf = tl.load(X + r * N + offs, mask=m, other=float("-inf")).to(tl.float32)
    tl.store(M + r * tl.num_programs(1) + c, tl.max(xf, axis=0))


@triton.jit
def _row_expsum_kernel(X, M, S, N, CHUNK: tl.constexpr):
    """Per-row exp((x - m).float()).sum in fp32. Follow-up sum across chunks."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < N
    xf = tl.load(X + r * N + offs, mask=m, other=float("-inf")).to(tl.float32)
    mv = tl.load(M + r)
    e = tl.where(m, tl.exp(xf - mv), 0.0)
    tl.store(S + r * tl.num_programs(1) + c, tl.sum(e, axis=0))


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
def _rank_probs_kernel(
    CAND, X, M, S, PROBS, IDX, N, k, TI: tl.constexpr, TJ: tl.constexpr
):
    """Emit (probs fp32, indices int64) in the descending-key order."""
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
    idx = 16777215 - (ki & 16777215)
    mv = tl.load(M + r)
    sv = tl.load(S + r)
    xf = tl.load(X + r * N + idx, mask=mi, other=0.0).to(tl.float32)
    # +inf logit: exp(inf - inf) is NaN under IEEE, but the row exp-sum then
    # is also +inf so a canonical formulation must produce p=1 for that
    # (unique) winner. We know the sum reduction already saturated on that
    # entry, so guard: if the logit equals m and s == +inf, emit 1.0.
    diff = xf - mv
    p = tl.where(sv == float("inf"), tl.where(diff == 0.0, 1.0, 0.0), tl.exp(diff) / sv)
    tl.store(IDX + r * k + rank.to(tl.int64), idx, mask=mi)
    tl.store(PROBS + r * k + rank.to(tl.int64), p, mask=mi)


def _fused_impl(logits: torch.Tensor, k: int):
    R, N = logits.shape
    if not 1 <= k <= N:
        raise ValueError(f"k={k} out of range for N={N}")
    if N > IDX_MASK + 1:
        raise ValueError(f"oracle supports N <= {IDX_MASK + 1}, got {N}")
    logits = logits.contiguous()
    dev = logits.device
    nch = triton.cdiv(N, CHUNK)

    # 1) row max
    mchunks = torch.empty(R, nch, dtype=torch.float32, device=dev)
    _row_max_kernel[(R, nch)](logits, mchunks, N, CHUNK=CHUNK)
    m = mchunks.max(dim=1).values.contiguous()

    # 2) row expsum in fp32
    schunks = torch.empty(R, nch, dtype=torch.float32, device=dev)
    _row_expsum_kernel[(R, nch)](logits, m, schunks, N, CHUNK=CHUNK)
    s = schunks.sum(dim=1).contiguous()  # may be +inf if any logit == +inf

    # 3) radix descent to find the k-th largest logit key per row
    prefix = torch.zeros(R, dtype=torch.int64, device=dev)
    krem = torch.full((R,), k, dtype=torch.int32, device=dev)
    hist = torch.zeros(R, 256, dtype=torch.int32, device=dev)
    for p in range(6, -1, -1):
        _hist_kernel[(R, nch)](logits, prefix, hist, N, shift=8 * p, CHUNK=CHUNK)
        _select_digit_kernel[(R,)](hist, prefix, krem)

    # 4) count + compact
    cnt = torch.empty(R, nch, dtype=torch.int32, device=dev)
    _count_ge_kernel[(R, nch)](logits, prefix, cnt, N, nch, CHUNK=CHUNK)
    offs = (torch.cumsum(cnt, dim=1) - cnt).to(torch.int32).contiguous()
    cand = torch.empty(R, k, dtype=torch.int64, device=dev)
    _compact_kernel[(R, nch)](logits, prefix, offs, cand, N, k, nch, CHUNK=CHUNK)

    # 5) rank + probs emission
    probs = torch.empty(R, k, dtype=torch.float32, device=dev)
    out_idx = torch.empty(R, k, dtype=torch.int64, device=dev)
    _rank_probs_kernel[(R, triton.cdiv(k, RANK_TI))](
        cand, logits, m, s, probs, out_idx, N, k, TI=RANK_TI, TJ=RANK_TJ
    )
    return probs, out_idx


def fused_softmax_topk(
    logits: torch.Tensor,  # [R, N] float32 or bfloat16, NaN-free
    k: int,
) -> tuple:  # (probs [R, k] fp32, indices [R, k] int64)
    return _fused_impl(logits, int(k))
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 a67c7147dadb026ca22380fe0ae29b13aae8013ae3d9ade671aa63d3f3914cd7
# FORGE-CANARY-SLOT-1 36d74bacec9865f8221688e19ad0abadb6fdde1389675a1d4a627add6157d4d0
# FORGE-CANARY-SLOT-2 31183c6bdaa23318a7d211bd3c1b4dbf19e583763b2a433fc87d64c4edf95ff5
# FORGE-CANARY-SLOT-3 bff5b333d2ac75c2d25ea6280dd8e7bf0268981cc2191487f9467eaf2b50256c
# FORGE-CANARY-END
