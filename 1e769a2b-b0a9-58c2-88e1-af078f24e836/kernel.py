"""
Starter -- row-wise descending top-k over PRIME row length N, forward only.

This file is yours. It is deliberately basic: three small Triton kernels do
all the real work over the [R, N] input, the slowest reasonable way -- build
a composite key per element, extract the row maximum with a chunked reduce,
cross it out, and rescan the whole row, k times over. Correct, deterministic,
and slow by design. Your job is to make it fast.

The entry point the harness calls is `topk_prime_row`, and its signature and
semantics must not change: descending top-k per row, ties broken by LOWER
original index first, values returned as bit-exact gathers of the input,
indices int64. N is guaranteed to be a PRIME at every graded shape --
between 5003 and just over 2^20 -- so any code that assumes N divides a
tile, or that virtually pads to the next power of two, must handle that
correctly under the exact tie-break rule (see the negative controls).
reference.py defines those semantics; agreement with it is graded exactly,
with no tolerance.

The trick worth keeping: (value bits << 21) | (2097151 - col) is unique per
row and orders the row by (value desc, index asc); the library selection
and sorting routines named in instruction.md are off-limits here; the
from-scratch max reductions below are the allowed kind of primitive.

STATUS: authored-draft starter -- structure mirrors the calibrated `topk`
category starter (GPU-proven there), applied over primes. Not re-timed in
this folder; the timing target and adoption headroom are pending GPU
calibration.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
CHUNK = 4096
PBLOCK = 512  # >= max chunk count: cdiv(2**21, CHUNK)
IDX_BITS = 21  # supports N up to 2**21 = 2,097,152
IDX_MASK = (1 << IDX_BITS) - 1  # 2097151


@triton.jit
def _rank_key_kernel(X, KEYS, total, N, BLOCK: tl.constexpr):
    """key = (ordered_bits(float32(x)) << 21) | (2097151 - col): unique per
    row, so 'largest key' IS 'largest value, lowest index'. -0.0 -> +0.0
    first. 64-bit offsets so R*N can exceed 2**31."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < total
    xf = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
    xf = tl.where(xf == 0.0, 0.0, xf)
    b = xf.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
    col = offs % N
    key = (u << 21) | (2097151 - col)
    tl.store(KEYS + offs, key, mask=m)


@triton.jit
def _chunk_max_kernel(KEYS, PM, N, nch, CHUNK: tl.constexpr):
    """Per-(row, chunk) maximum key; the whole row is rescanned on every one
    of the k steps -- the deliberate naivety of this starter."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < N
    kv = tl.load(KEYS + r * N + offs, mask=m, other=-1)
    tl.store(PM + r * nch + c, tl.max(kv, axis=0))


@triton.jit
def _pick_and_mask_kernel(PM, KEYS, OUT_IDX, N, nch, k, j, PBLOCK: tl.constexpr):
    """One program per row reduces the chunk maxima, decodes the winning
    column, records it as output j, and crosses it out (-1). Keys are unique
    per row, so the max is unique and every step is deterministic."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, PBLOCK).to(tl.int64)
    pm = tl.load(PM + r * nch + c, mask=c < nch, other=-1)
    mx = tl.max(pm, axis=0)
    col = 2097151 - (mx & 2097151)
    tl.store(OUT_IDX + r * k + j, col)
    tl.store(KEYS + r * N + col, mx * 0 - 1)


def topk_prime_row(
    x: torch.Tensor,  # [R, N] float32 or bfloat16, NaN-free; N prime
    k: int,
) -> tuple:  # (values [R, k] in x.dtype, indices [R, k] int64)
    """Entry point. Signature and semantics are fixed; the body is yours."""
    k = int(k)
    R, N = x.shape
    if not (1 <= k <= N):
        raise ValueError(f"k={k} out of range for N={N}")
    if N > IDX_MASK + 1:
        raise ValueError(f"starter supports N <= {IDX_MASK + 1}, got {N}")
    x = x.contiguous()

    keys = torch.empty(R, N, dtype=torch.int64, device=x.device)
    total = R * N
    _rank_key_kernel[(triton.cdiv(total, BLOCK),)](
        x.view(-1), keys.view(-1), total, N, BLOCK=BLOCK
    )

    nch = triton.cdiv(N, CHUNK)
    pm = torch.empty(R, nch, dtype=torch.int64, device=x.device)
    out_idx = torch.empty(R, k, dtype=torch.int64, device=x.device)
    for j in range(k):
        _chunk_max_kernel[(R, nch)](keys, pm, N, nch, CHUNK=CHUNK)
        _pick_and_mask_kernel[(R,)](pm, keys, out_idx, N, nch, k, j, PBLOCK=PBLOCK)

    values = x.gather(1, out_idx)
    return values, out_idx
