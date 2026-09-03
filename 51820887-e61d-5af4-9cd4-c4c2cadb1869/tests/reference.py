"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is a fused row-wise softmax followed by a top-k selection,
canonical shape of a speculative-decoding gate:

    fused_softmax_topk(logits[R, N], k) -> (probs[R, k] float32, indices[R, k] int64)

Semantics, exactly:

  * Row-wise softmax (canonical fp32 reduction):
        m       = logits.max(dim=-1, keepdim=True).values.float()
        e       = (logits.float() - m).exp()
        s       = e.sum(dim=-1, keepdim=True)
        probs_full = e / s                                            # [R, N] fp32
    The intermediate math is float32 regardless of input dtype; this
    prevents bf16 subtract-max cancellation and is the standard fused
    softmax numerical policy.
  * `indices[r]` holds the positions of the k LARGEST values of
    logits[r, :] (equivalently, of probs_full[r, :]) in [0, N), ordered
    DESCENDING by value, ties broken by LOWER ORIGINAL INDEX first,
    `-0.0 == +0.0` one tie group. Softmax is strictly increasing in the
    input, so top-k over the logits is exactly top-k over the probs;
    tie groups are preserved.
  * `probs[r, :] = probs_full[r].gather(0, indices[r])` in float32.
    Values are the softmax probabilities at the selected indices; they
    are compared with a dtype-dependent tolerance (see taskdef.TOL).
  * `indices` is int64.
  * Inputs are guaranteed NaN-free; `-inf`/`+inf` are legal on the logits
    (a `+inf` logit yields a probability of 1 at that index, everything
    else 0, and it wins the top-k unambiguously).

Correctness grading: indices EXACT (integer equality); probs by tolerance
per dtype (TOL in taskdef.py). The `compare` override enforces this split;
a candidate that matches indices but returns probs computed with a
different reduction policy (e.g. bf16 accumulator) is caught by the probs
tolerance on bfloat16 inputs.

The implementation runs a stable descending argsort on logits, gathers the
first k local indices, and computes the softmax reduction on the row's
fp32 upcast then gathers probs at those indices. Deliberately simple and
slow.
"""

import torch


def fused_softmax_topk_ref(logits: torch.Tensor, k: int):
    """Sequential ground truth. logits: [R, N] float32 or bfloat16, NaN-free.
    1 <= k <= N. Returns (probs[R, k] float32, indices[R, k] int64)."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be [R, N], got {tuple(logits.shape)}")
    R, N = logits.shape
    k = int(k)
    if not (1 <= k <= N):
        raise ValueError(f"k={k} out of range for N={N}")
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)  # [R, N]
    idx = order[:, :k].contiguous().to(torch.int64)
    xf = logits.float()
    m = xf.max(dim=-1, keepdim=True).values
    e = (xf - m).exp()
    s = e.sum(dim=-1, keepdim=True)
    probs_full = e / s
    probs = torch.gather(probs_full, -1, idx).contiguous()
    return probs, idx


if __name__ == "__main__":

    def _brute(row, k):
        return sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:k]

    inf = float("inf")
    logits = torch.tensor(
        [
            [3.0, 1.0, 3.0, -0.0, 0.0, inf, -inf, 2.0, 5.0, 5.0, 1.0, 1.0],
            [-1.0, -1.0, -2.0, 0.0, 4.0, 4.0, 4.0, -inf, 2.0, 7.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    for k in (1, 3, 6):
        p, idx = fused_softmax_topk_ref(logits, k)
        assert p.shape == (logits.shape[0], k) and idx.shape == (logits.shape[0], k)
        assert idx.dtype == torch.int64 and p.dtype == torch.float32
        for r in range(logits.shape[0]):
            want = _brute(logits[r].tolist(), k)
            assert idx[r].tolist() == want, (r, k, idx[r].tolist(), want)
        # Row-wise probs sum to 1 on rows without +inf logits; row 0 has +inf
        # at index 5, which owns all mass in canonical fp32 softmax.
        p_full_r1 = torch.softmax(logits[1].float(), dim=-1)
        want_p_r1 = p_full_r1.gather(0, idx[1])
        assert torch.allclose(p[1], want_p_r1, atol=1e-6, rtol=1e-5)
    pb, ib = fused_softmax_topk_ref(logits.to(torch.bfloat16), 2)
    assert pb.dtype == torch.float32 and ib.dtype == torch.int64
    print(
        "fused_softmax_topk reference CPU sanity OK: indices/tie-break/probs validated"
    )
