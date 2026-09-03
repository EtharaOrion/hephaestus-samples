"""
Starter -- row-wise argmax with mixed input dtype (fp32 / bf16 / fp16), fwd only.

This file is yours. It is deliberately basic: two Triton kernels build a
composite key over the row (upcast the input to fp32, reorder its bits into
a monotone unsigned key, pack the local column into the low bits with the
lower-index-first rule), then reduce the row max in a chunked pass. That is
correct and one pass over the row, but the reduce path is a
generic-int64-max that leaves headroom on the table -- there is no fp16
special-casing, no vectorized load, no cooperative-warp shuffle. Your job
is to make it fast.

The entry point the harness calls is `argmax_mixed_dtype`, and its
signature and semantics must not change: return the position of the largest
value in each row of `x[R, N]`, ties broken by LOWER original index first,
`-0.0 == +0.0` one tie group, indices int64. Input dtype is one of
float32, bfloat16, float16 drawn per invocation. reference.py defines the
semantics; agreement with it is graded exactly, with no tolerance.

The trick worth keeping: upcasting fp16/bf16 to fp32 BEFORE the bit
reorder is what makes ONE key format work across all three dtypes -- the
alternative (fp16 keys in int16, bf16 keys in int16, fp32 keys in int32)
needs three separate kernels and the negative controls include an
nc_fp16_upcast arm that zeroes at G2 by keeping the composite key at
input precision. The library argmax and sort primitives are off-limits;
top-1 primitives (`tl.max`, `tl.argmax`) and the from-scratch max reduce
below are the allowed kind of primitive.

STATUS: authored-draft starter -- structure mirrors the calibrated `topk`
category starter (GPU-proven there), collapsed to a single-pick argmax.
Not re-timed in this folder; the timing target and adoption headroom are
pending GPU calibration.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
CHUNK = 8192
PBLOCK = 1024  # >= max chunk count: cdiv(2**24, CHUNK)
IDX_BITS = 24  # supports N up to 2**24 = 16,777,216
IDX_MASK = (1 << IDX_BITS) - 1


@triton.jit
def _rank_key_kernel(
    X,
    KEYS,
    total,
    N,
    BLOCK: tl.constexpr,
    IDX_BITS: tl.constexpr,
    IDX_MASK: tl.constexpr,
):
    """key = (ordered_bits(fp32(x)) << IDX_BITS) | (IDX_MASK - col). Upcast
    to fp32 uniformly across fp32/bf16/fp16 inputs so ONE key format covers
    every dtype. -0.0 -> +0.0 first. 64-bit offsets so R*N above 2**31 is
    safe."""
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
    """Per-(row, chunk) maximum key. One pass over the row per rank; here
    rank is k=1 so this is called exactly once per row."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < N
    kv = tl.load(KEYS + r * N + offs, mask=m, other=-1)
    tl.store(PM + r * nch + c, tl.max(kv, axis=0))


@triton.jit
def _pick_kernel(PM, OUT_IDX, nch, IDX_MASK: tl.constexpr, PBLOCK: tl.constexpr):
    """One program per row reduces the chunk maxima and decodes the winning
    column. Keys are unique per row, so the max is unique and the pick is
    deterministic -- no atomics, no racy tie winner."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, PBLOCK).to(tl.int64)
    pm = tl.load(PM + r * nch + c, mask=c < nch, other=-1)
    mx = tl.max(pm, axis=0)
    col = IDX_MASK - (mx & IDX_MASK)
    tl.store(OUT_IDX + r, col)


def argmax_mixed_dtype(
    x: torch.Tensor,  # [R, N] float32, bfloat16, or float16, NaN-free
) -> torch.Tensor:  # indices [R] int64
    """Entry point. Signature and semantics are fixed; the body is yours."""
    if x.dim() != 2:
        raise ValueError(f"x must be [R, N], got {tuple(x.shape)}")
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"unsupported dtype {x.dtype}")
    R, N = x.shape
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
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
        BLOCK=BLOCK,
        IDX_BITS=IDX_BITS,
        IDX_MASK=IDX_MASK,
    )

    nch = triton.cdiv(N, CHUNK)
    pm = torch.empty(R, nch, dtype=torch.int64, device=x.device)
    out_idx = torch.empty(R, dtype=torch.int64, device=x.device)
    _chunk_max_kernel[(R, nch)](keys, pm, N, nch, CHUNK=CHUNK)
    _pick_kernel[(R,)](pm, out_idx, nch, IDX_MASK=IDX_MASK, PBLOCK=PBLOCK)
    return out_idx
