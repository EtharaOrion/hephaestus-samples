"""
Starter -- row-wise descending top-k over VERY LARGE N with SMALL k, fwd only.

This file is yours. It is deliberately basic: three small Triton kernels do
all the real work over the [R, N] input, the slowest reasonable way -- build
a composite key per element, extract the row maximum with a chunked reduce,
cross it out, and rescan the whole (multi-million-element) row, k times over.
That is O(k * N) HBM traffic and is exactly the wall the giant-N regime
punishes. Your job is to make it fast.

The entry point the harness calls is `topk_giant_row`, and its signature and
semantics must not change: descending top-k per row, ties broken by LOWER
original index first, values returned as bit-exact gathers of the input,
indices int64. N is guaranteed >= 262144 at every graded shape (up to a
1M-plus tenant) and k is small (16..128); do not shortcut on either.
reference.py defines the semantics; agreement with it is graded exactly,
with no tolerance.

The key trick worth keeping: (float bits reordered << 23) | (max_col - col)
is unique per row and orders it by (value desc, index asc). The 23-bit
index slot fits any graded N up to 2**23 - 1 = 8,388,607. The library
selection and sorting routines named in instruction.md are off-limits;
the from-scratch max reductions below are the allowed kind of primitive.

STATUS: authored-draft starter -- structure mirrors the calibrated `topk`
category starter (GPU-proven there), widened to a 23-bit index slot for
the giant-N regime. Not re-timed in this folder; the timing target and
adoption headroom are pending GPU calibration.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
CHUNK = 8192
PBLOCK = 1024  # >= max chunk count: cdiv(2**23, CHUNK)
IDX_BITS = 23  # supports N up to 2**23 = 8,388,608
IDX_MASK = (1 << IDX_BITS) - 1  # 8388607


@triton.jit
def _rank_key_kernel(
    X,
    KEYS,
    total,
    N,
    IDX_BITS: tl.constexpr,
    IDX_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """key = (ordered_bits(float32(x)) << IDX_BITS) | (IDX_MASK - col):
    unique per row, so 'largest key' IS 'largest value, lowest index'.
    -0.0 -> +0.0 first. 64-bit offsets so R*N (up to ~6.7e7 here but
    computed uniformly with the family) never wraps."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < total
    xf = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
    xf = tl.where(xf == 0.0, 0.0, xf)
    b = xf.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
    col = offs % N
    key = (u << IDX_BITS) | (IDX_MASK - col)
    tl.store(KEYS + offs, key, mask=m)


@triton.jit
def _chunk_max_kernel(KEYS, PM, N, nch, CHUNK: tl.constexpr):
    """Per-(row, chunk) maximum key; the whole row is rescanned on every
    one of the k steps -- the deliberate naivety of this starter, and
    exactly what makes it slow in the giant-N regime."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < N
    kv = tl.load(KEYS + r * N + offs, mask=m, other=-1)
    tl.store(PM + r * nch + c, tl.max(kv, axis=0))


@triton.jit
def _pick_and_mask_kernel(
    PM, KEYS, OUT_IDX, N, nch, k, j, IDX_MASK: tl.constexpr, PBLOCK: tl.constexpr
):
    """One program per row reduces the chunk maxima, decodes the winning
    column, records it as output j, crosses it out (-1). Keys are unique
    per row, so the max is unique and every step is deterministic."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, PBLOCK).to(tl.int64)
    pm = tl.load(PM + r * nch + c, mask=c < nch, other=-1)
    mx = tl.max(pm, axis=0)
    col = IDX_MASK - (mx & IDX_MASK)
    tl.store(OUT_IDX + r * k + j, col)
    tl.store(KEYS + r * N + col, mx * 0 - 1)


def topk_giant_row(
    x: torch.Tensor,  # [R, N] float32 or bfloat16, NaN-free
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
        x.view(-1),
        keys.view(-1),
        total,
        N,
        IDX_BITS=IDX_BITS,
        IDX_MASK=IDX_MASK,
        BLOCK=BLOCK,
    )

    nch = triton.cdiv(N, CHUNK)
    pm = torch.empty(R, nch, dtype=torch.int64, device=x.device)
    out_idx = torch.empty(R, k, dtype=torch.int64, device=x.device)
    for j in range(k):
        _chunk_max_kernel[(R, nch)](keys, pm, N, nch, CHUNK=CHUNK)
        _pick_and_mask_kernel[(R,)](
            pm, keys, out_idx, N, nch, k, j, IDX_MASK=IDX_MASK, PBLOCK=PBLOCK
        )

    values = x.gather(1, out_idx)
    return values, out_idx
