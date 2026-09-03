"""
Starter -- ragged descending top-k over variable-length row slices, fwd only.

This file is yours. It is deliberately basic: build a composite key per
element (locally within the row), then iterate the row's own slice k times
extracting the maximum key and crossing it out. One row = one Triton
program grid launch pair per rank; that is O(k * L_r) HBM traffic per row.
Correct, deterministic, and slow by design. Your job is to make it fast.

The entry point the harness calls is `topk_ragged`, and its signature and
semantics must not change: for each row r whose slice is
`x[seg_offsets[r]:seg_offsets[r+1]]`, return the k largest values (bit-exact
gathers of the input) and their GLOBAL indices (int64, i.e. `local + lo`),
descending, ties broken by LOWER original global index first. Row lengths
are ARBITRARY; every row has L_r >= k by construction. reference.py defines
those semantics; agreement with it is graded exactly, with no tolerance.

The trick worth keeping: within a row the LOCAL order and the GLOBAL order
agree, so a composite key over the LOCAL column -- (value bits << 23) |
(2**23 - 1 - local) -- is unique and orders each row by (value desc, index
asc); the global index is `lo + local` added at the end. The library
selection and sorting routines named in instruction.md are off-limits;
the from-scratch max reductions below are the allowed kind of primitive.

STATUS: authored-draft starter -- structure mirrors the calibrated `topk`
category starter (GPU-proven there), applied per variable-length row via a
row-major flat key buffer. Not re-timed in this folder; the timing target
and adoption headroom are pending GPU calibration.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
CHUNK = 4096
PBLOCK = 1024  # >= max chunk count for the tested Lmax
IDX_BITS = 23
IDX_MASK = (1 << IDX_BITS) - 1  # 8388607


@triton.jit
def _rank_key_kernel(
    X, SEG, KEYS, R, BLOCK: tl.constexpr, IDX_BITS: tl.constexpr, IDX_MASK: tl.constexpr
):
    """One program per (row, chunk-in-row). key = (ordered_bits(float32(x))
    << IDX_BITS) | (IDX_MASK - local): unique within a row, so 'largest key'
    IS 'largest value, lowest local index'. -0.0 -> +0.0 first. 64-bit
    offsets so NNZ can exceed 2**31."""
    r = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1).to(tl.int64)
    lo = tl.load(SEG + r).to(tl.int64)
    hi = tl.load(SEG + r + 1).to(tl.int64)
    L = hi - lo
    offs = b * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < L
    xf = tl.load(X + lo + offs, mask=m, other=0.0).to(tl.float32)
    xf = tl.where(xf == 0.0, 0.0, xf)
    bits = xf.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(bits >= 0, bits ^ 0x80000000, (~bits) & 0xFFFFFFFF)
    key = (u << IDX_BITS) | (IDX_MASK - offs)
    tl.store(KEYS + lo + offs, key, mask=m)


@triton.jit
def _chunk_max_kernel(KEYS, SEG, PM, nch, CHUNK: tl.constexpr):
    """Per-(row, chunk) maximum key over the row's own slice; the whole slice
    is rescanned on every one of the k steps -- the deliberate naivety of
    this starter."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    lo = tl.load(SEG + r).to(tl.int64)
    hi = tl.load(SEG + r + 1).to(tl.int64)
    L = hi - lo
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < L
    kv = tl.load(KEYS + lo + offs, mask=m, other=-1)
    tl.store(PM + r * nch + c, tl.max(kv, axis=0))


@triton.jit
def _pick_and_mask_kernel(
    PM, KEYS, SEG, OUT_IDX, k, j, nch, IDX_MASK: tl.constexpr, PBLOCK: tl.constexpr
):
    """One program per row reduces the chunk maxima, decodes the winning
    LOCAL column, records the GLOBAL index as output j, crosses it out (-1).
    Keys are unique per row, so the max is unique and every step is
    deterministic."""
    r = tl.program_id(0).to(tl.int64)
    lo = tl.load(SEG + r).to(tl.int64)
    c = tl.arange(0, PBLOCK).to(tl.int64)
    pm = tl.load(PM + r * nch + c, mask=c < nch, other=-1)
    mx = tl.max(pm, axis=0)
    local = IDX_MASK - (mx & IDX_MASK)
    tl.store(OUT_IDX + r * k + j, lo + local)
    tl.store(KEYS + lo + local, mx * 0 - 1)


def topk_ragged(
    x: torch.Tensor,  # [NNZ] float32 or bfloat16, NaN-free
    seg_offsets: torch.Tensor,  # [R+1] int64 prefix sum
    k: int,
) -> tuple:  # (values [R, k] in x.dtype, indices [R, k] int64)
    """Entry point. Signature and semantics are fixed; the body is yours."""
    k = int(k)
    if x.dim() != 1:
        raise ValueError(f"x must be 1D flat, got {tuple(x.shape)}")
    if seg_offsets.dtype != torch.int64 or seg_offsets.dim() != 1:
        raise ValueError("seg_offsets must be an int64 vector [R+1]")
    R = int(seg_offsets.shape[0]) - 1
    NNZ = int(x.shape[0])
    lens = seg_offsets[1:] - seg_offsets[:-1]
    Lmax = int(lens.max().item())
    if Lmax > IDX_MASK + 1:
        raise ValueError(f"starter supports Lmax <= {IDX_MASK + 1}, got {Lmax}")
    x = x.contiguous()
    seg_offsets = seg_offsets.contiguous()

    keys = torch.full((NNZ,), -1, dtype=torch.int64, device=x.device)
    max_blocks = triton.cdiv(Lmax, BLOCK)
    _rank_key_kernel[(R, max_blocks)](
        x, seg_offsets, keys, R, BLOCK=BLOCK, IDX_BITS=IDX_BITS, IDX_MASK=IDX_MASK
    )

    nch = triton.cdiv(Lmax, CHUNK)
    pm = torch.empty(R, nch, dtype=torch.int64, device=x.device)
    out_idx = torch.empty(R, k, dtype=torch.int64, device=x.device)
    for j in range(k):
        _chunk_max_kernel[(R, nch)](keys, seg_offsets, pm, nch, CHUNK=CHUNK)
        _pick_and_mask_kernel[(R,)](
            pm, keys, seg_offsets, out_idx, k, j, nch, IDX_MASK=IDX_MASK, PBLOCK=PBLOCK
        )

    values = x.gather(0, out_idx.view(-1)).view(R, k)
    return values, out_idx
