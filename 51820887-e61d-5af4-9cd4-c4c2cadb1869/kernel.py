"""
Starter -- fused row-wise softmax + top-k, forward only.

This file is yours. It is deliberately basic: three Triton kernels build a
composite key over the logits (bit-reordered fp32 << 23 | inverse-index),
extract the max k times over, then compute the softmax reduction in a
separate pass and gather probs at the winning indices. That is FOUR passes
over the row where the sota does one; making it fast is a real fusion
exercise, not a micro-tuning one.

The entry point the harness calls is `fused_softmax_topk`, and its
signature and semantics must not change: canonical fp32 softmax over each
row (max-subtract, exp, sum-normalize), returned probs are the softmax
probabilities AT THE TOP-K INDICES in fp32, indices int64. Ties on the
LOGITS resolve to lower-original-index; softmax preserves order, so ties
on the probs resolve the same way. reference.py defines the semantics;
indices are graded EXACT and probs by a small dtype-dependent tolerance.

The library selection and sorting routines named in instruction.md are
off-limits here, and so are the softmax spellings (`torch.softmax`,
`F.softmax`, `_softmax`) -- the softmax reduction must live in your own
kernel path.

STATUS: authored-draft starter -- correct and deterministic, structure
mirrors the calibrated `topk` category starter (GPU-proven there) with an
added softmax reduction pass. Not re-timed in this folder; the timing
target and adoption headroom are pending GPU calibration.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
CHUNK = 4096
PBLOCK = 512
IDX_BITS = 23
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
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, PBLOCK).to(tl.int64)
    pm = tl.load(PM + r * nch + c, mask=c < nch, other=-1)
    mx = tl.max(pm, axis=0)
    col = IDX_MASK - (mx & IDX_MASK)
    tl.store(OUT_IDX + r * k + j, col)
    tl.store(KEYS + r * N + col, mx * 0 - 1)


@triton.jit
def _row_softmax_stats_kernel(X, MAX_OUT, LSE_OUT, N, CHUNK: tl.constexpr):
    """Per-row fp32 (max, log-sum-exp): one program per row, chunked online
    reduction (Chan-Golub-LeVeque style). Deterministic under the chunk
    tiling but NOT bit-identical to the reference's whole-row reduction --
    that is why probs are graded with a tolerance."""
    r = tl.program_id(0).to(tl.int64)
    nch = tl.cdiv(N, CHUNK)
    m_acc = tl.load(X + r * N).to(tl.float32)
    s_acc = 0.0
    for c in range(nch):
        offs = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
        mk = offs < N
        xf = tl.load(X + r * N + offs, mask=mk, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m_acc, tl.max(xf, axis=0))
        s_acc = s_acc * tl.exp(m_acc - m_new) + tl.sum(tl.exp(xf - m_new) * mk, axis=0)
        m_acc = m_new
    tl.store(MAX_OUT + r, m_acc)
    tl.store(LSE_OUT + r, tl.log(s_acc) + m_acc)


@triton.jit
def _gather_probs_kernel(X, OUT_IDX, MAX_IN, LSE_IN, PROBS, N, k, KBLOCK: tl.constexpr):
    """Per-row: probs[r, j] = exp(x[r, out_idx[r, j]] - lse[r])."""
    r = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, KBLOCK).to(tl.int64)
    m = j < k
    idx = tl.load(OUT_IDX + r * k + j, mask=m, other=0)
    xf = tl.load(X + r * N + idx, mask=m, other=0.0).to(tl.float32)
    lse = tl.load(LSE_IN + r)
    p = tl.exp(xf - lse)
    tl.store(PROBS + r * k + j, p, mask=m)


def fused_softmax_topk(
    logits: torch.Tensor,  # [R, N] float32 or bfloat16, NaN-free
    k: int,
) -> tuple:  # (probs [R, k] fp32, indices [R, k] int64)
    """Entry point. Signature and semantics are fixed; the body is yours."""
    k = int(k)
    R, N = logits.shape
    if not (1 <= k <= N):
        raise ValueError(f"k={k} out of range for N={N}")
    if N > IDX_MASK + 1:
        raise ValueError(f"starter supports N <= {IDX_MASK + 1}, got {N}")
    logits = logits.contiguous()

    keys = torch.empty(R, N, dtype=torch.int64, device=logits.device)
    total = R * N
    _rank_key_kernel[(triton.cdiv(total, BLOCK),)](
        logits.view(-1),
        keys.view(-1),
        total,
        N,
        BLOCK=BLOCK,
        IDX_BITS=IDX_BITS,
        IDX_MASK=IDX_MASK,
    )

    nch = triton.cdiv(N, CHUNK)
    pm = torch.empty(R, nch, dtype=torch.int64, device=logits.device)
    out_idx = torch.empty(R, k, dtype=torch.int64, device=logits.device)
    for j in range(k):
        _chunk_max_kernel[(R, nch)](keys, pm, N, nch, CHUNK=CHUNK)
        _pick_and_mask_kernel[(R,)](
            pm, keys, out_idx, N, nch, k, j, IDX_MASK=IDX_MASK, PBLOCK=PBLOCK
        )

    max_r = torch.empty(R, dtype=torch.float32, device=logits.device)
    lse_r = torch.empty(R, dtype=torch.float32, device=logits.device)
    _row_softmax_stats_kernel[(R,)](logits, max_r, lse_r, N, CHUNK=CHUNK)

    probs = torch.empty(R, k, dtype=torch.float32, device=logits.device)
    kblock = 1
    while kblock < k:
        kblock *= 2
    _gather_probs_kernel[(R,)](
        logits, out_idx, max_r, lse_r, probs, N, k, KBLOCK=kblock
    )
    return probs, out_idx
