"""Private oracle -- from-scratch Triton k=1 row-wise argmax with mixed dtype.

UNVERIFIED IN THIS FOLDER: single-pass composite-key reduction with a per-row
atomic-max on int64. Handles fp32, bf16, fp16 uniformly by upcasting to fp32
inside the kernel before composite-key derivation (preserves the exact
lower-original-index tie-break under all three formats and does not lose
tie-break bits on fp16 subnormals -- the nc_fp16_upcast negative control
zeroes at G2 by precisely this shortcut). No GPU was available at authoring
time; the headroom / full-reward claim is PENDING GPU CALIBRATION at the
golden stage.

Never shipped to an agent. Establishes that full reward is reachable through
every live gate (G1 symbol scan naming both the free-function and the tensor
method spellings of the selection anchors; exact G2 int64-index semantics;
G3 timing; G4 adoption) with no library selection or sorting call anywhere.

Algorithm (deterministic, no value-dependent branching):

  key(r, i) = (ordered_bits32(fp32(x[r, i])) << 24) | (16777215 - i)

  Every key is unique per row, so "largest key" IS "largest value, lowest
  index". -0.0 -> +0.0 before the bit transform so both zeros form one tie
  group ordered by index.

  1. One streaming pass. Each (row, chunk) program computes its chunk's max
     key and does one tl.atomic_max into a per-row int64 buffer. Order-free
     integer max is associative and commutative, so the result is
     deterministic across every launch shape and every scheduler ordering.
  2. Decode: idx = 16777215 - (row_max_key & 16777215).

  IDX_BITS = 24 supports N up to 2**24 = 16,777,216 (max hidden N is
  1,200,000). 64-bit offsets throughout: c6 (R=2048 * N=1,200,000) crosses
  2**31 elements.
"""

import torch
import triton
import triton.language as tl

CHUNK = 4096
IDX_BITS = 24
IDX_MASK = (1 << IDX_BITS) - 1  # 16777215
NEG_INF_KEY = -(1 << 62)  # every real key strictly greater


@triton.jit
def _chunk_atomic_max_kernel(X, ROWMAX, N, CHUNK: tl.constexpr, NEG_INF_KEY: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
    m = offs < N
    xf = tl.load(X + r * N + offs, mask=m, other=0.0).to(tl.float32)
    xf = tl.where(xf == 0.0, 0.0, xf)  # -0.0 -> +0.0
    b = xf.to(tl.int32, bitcast=True).to(tl.int64)
    u = tl.where(b >= 0, b ^ 0x80000000, (~b) & 0xFFFFFFFF)
    key = (u << 24) | (16777215 - offs)
    # Inactive lanes carry NEG_INF_KEY so they never win the local max.
    key_safe = tl.where(m, key, NEG_INF_KEY)
    local_max = tl.max(key_safe, axis=0)
    tl.atomic_max(ROWMAX + r, local_max)


def _argmax_impl(x: torch.Tensor):
    R, N = x.shape
    if N > IDX_MASK + 1:
        raise ValueError(f"oracle supports N <= {IDX_MASK + 1}, got {N}")
    if N == 0:
        raise ValueError("empty row")
    x = x.contiguous()
    dev = x.device
    nch = triton.cdiv(N, CHUNK)

    row_max = torch.full((R,), NEG_INF_KEY, dtype=torch.int64, device=dev)
    _chunk_atomic_max_kernel[(R, nch)](x, row_max, N, CHUNK=CHUNK, NEG_INF_KEY=NEG_INF_KEY)
    idx = (16777215 - (row_max & 16777215)).to(torch.int64)
    return idx


def argmax_mixed_dtype(
    x: torch.Tensor,  # [R, N] float32, bfloat16, or float16, NaN-free
) -> torch.Tensor:  # indices [R] int64
    return _argmax_impl(x)
# FORGE-CANARY-BEGIN forge-canary/v1
# FORGE-CANARY-SLOT-0 ddc04cc54b21b837ac5b0f07e0ace9920ce0f3a7163e1ed357b5c660ca2f8ef2
# FORGE-CANARY-SLOT-1 6913886b93a188cfab568dd96c3f71af289aa8d364d7b9c026c64facdfeea081
# FORGE-CANARY-SLOT-2 1a48efddf7e826c02c9a973b42b72c2edbd1a5da7375d6d69eccb4dd028744f7
# FORGE-CANARY-SLOT-3 e008db3b35ab885bdd02db1cb573c074c22bd70c5261bd9c40eb517316ca451c
# FORGE-CANARY-END
