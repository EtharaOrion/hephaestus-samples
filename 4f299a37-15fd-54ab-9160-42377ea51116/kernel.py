"""
Starter -- segmented descending top-k within fixed-length row segments, fwd.

This file is yours. It is deliberately basic: three small Triton kernels do
all the real work over the flattened [R*G, S] segment view, the slowest
reasonable way -- extract the per-segment maximum key, cross it out, and
rescan the whole segment, k times over. Correct, deterministic, and slow by
design. Your job is to make it fast.

The entry point the harness calls is `topk_segmented`, and its signature and
semantics must not change: each row is cut into G = N // segment contiguous
length-S segments and a descending top-k is taken inside every segment;
`values`/`indices` are [R, G, k]; indices are GLOBAL row positions (segment
offset added); ties broken by LOWER original index first; values are bit-exact
gathers of the input; indices int64. reference.py defines those semantics;
agreement with it is graded exactly, with no tolerance.

The trick worth keeping: within a segment the LOCAL order and the GLOBAL order
agree, so a composite key over the LOCAL column -- (value bits << 21) |
(2097151 - local) -- is unique and orders each segment by (value desc, index
asc); the global index is just s*S + local, added at the end. The library
selection and sorting routines named in instruction.md are off-limits here;
the from-scratch max reductions below are the allowed kind of primitive.

STATUS: authored-draft starter -- structure mirrors the calibrated `topk`
category starter (GPU-proven there), applied per segment. Not re-timed in this
folder; the timing target and adoption headroom are pending GPU calibration.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
CHUNK = 4096
PBLOCK = 512                     # >= max chunk count: cdiv(2**21, CHUNK)
IDX_BITS = 21                    # supports segment length S up to 2**21
IDX_MASK = (1 << IDX_BITS) - 1   # 2097151


@triton.jit
def _rank_key_kernel(X, KEYS, total, S, BLOCK: tl.constexpr):
    """key = (ordered_bits(float32(x)) << 21) | (2097151 - local): unique
    within a segment, so 'largest key' IS 'largest value, lowest LOCAL index'.
    -0.0 -> +0.0 first. local = offs % S. 64-bit offsets: R*N can exceed
    2**31."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < total
    xf = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
    xf = tl.where(xf == 0.0, 0.0, xf)
    b = xf.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
    local = offs % S
    key = (u << 21) | (2097151 - local)
    tl.store(KEYS + offs, key, mask=m)


@triton.jit
def _chunk_max_kernel(KEYS, PM, S, nch, CHUNK: tl.constexpr):
    """Per-(flat-segment, chunk) maximum key; the whole segment is rescanned on
    every one of the k steps -- the deliberate naivety of this starter."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < S
    kv = tl.load(KEYS + r * S + offs, mask=m, other=-1)
    tl.store(PM + r * nch + c, tl.max(kv, axis=0))


@triton.jit
def _pick_and_mask_kernel(PM, KEYS, OUT_LOCAL, S, nch, k, j, PBLOCK: tl.constexpr):
    """One program per flat segment reduces the chunk maxima, decodes the
    winning LOCAL column, records it as output j, and crosses it out (-1). Keys
    are unique per segment, so the max is unique and every step is
    deterministic -- no atomics, no racy tie winner."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, PBLOCK).to(tl.int64)
    pm = tl.load(PM + r * nch + c, mask=c < nch, other=-1)
    mx = tl.max(pm, axis=0)
    local = 2097151 - (mx & 2097151)
    tl.store(OUT_LOCAL + r * k + j, local)
    tl.store(KEYS + r * S + local, mx * 0 - 1)        # int64 -1 sentinel


def topk_segmented(
    x: torch.Tensor,     # [R, N] float32 or bfloat16, NaN-free
    k: int,
    segment: int,        # S; divides N; 1 <= k <= S
) -> tuple:              # (values [R, G, k] in x.dtype, indices [R, G, k] int64)
    """Entry point. Signature and semantics are fixed; the body is yours."""
    k, S = int(k), int(segment)
    R, N = x.shape
    if S <= 0 or N % S != 0:
        raise ValueError(f"segment={S} must be positive and divide N={N}")
    if not (1 <= k <= S):
        raise ValueError(f"k={k} out of range for segment length S={S}")
    if S > IDX_MASK + 1:
        raise ValueError(f"starter supports segment <= {IDX_MASK + 1}, got {S}")
    G = N // S
    M = R * G
    xv = x.contiguous().view(M, S)                    # flat-segment view (no copy)

    keys = torch.empty(M, S, dtype=torch.int64, device=x.device)
    total = M * S
    _rank_key_kernel[(triton.cdiv(total, BLOCK),)](xv.reshape(-1), keys.view(-1),
                                                   total, S, BLOCK=BLOCK)

    # The slow part: extract the maximum key k times per flat segment,
    # rescanning the whole segment each time.
    nch = triton.cdiv(S, CHUNK)
    pm = torch.empty(M, nch, dtype=torch.int64, device=x.device)
    out_local = torch.empty(M, k, dtype=torch.int64, device=x.device)
    for j in range(k):
        _chunk_max_kernel[(M, nch)](keys, pm, S, nch, CHUNK=CHUNK)
        _pick_and_mask_kernel[(M,)](pm, keys, out_local, S, nch, k, j,
                                    PBLOCK=PBLOCK)

    # Globalise: local column -> s*S + local. Gather bit-exact values.
    local = out_local.view(R, G, k)
    seg_off = (torch.arange(G, device=x.device, dtype=torch.int64) * S).view(1, G, 1)
    idx = local + seg_off
    values = x.gather(1, idx.reshape(R, G * k)).reshape(R, G, k)
    return values, idx
