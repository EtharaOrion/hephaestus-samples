"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is SEGMENTED top-k: each row is cut into contiguous fixed-length
segments and a descending top-k is taken inside every segment independently:

    topk_segmented(x[R, N], k, segment) -> (values[R, G, k], indices[R, G, k])

with segment length S = `segment`, S dividing N, and G = N // S segments per
row. 1 <= k <= S.

Semantics, exactly:

  * For row r and segment s (columns [s*S, s*S + S) of x[r]), `indices[r, s]`
    holds the positions of the k LARGEST elements WITHIN that segment, ordered
    by DESCENDING value, ties broken by LOWER ORIGINAL INDEX first, `-0.0 ==
    +0.0` one tie group.
  * Indices are GLOBAL positions within the row (in [0, N)): index = s*S + (the
    local position inside the segment). Global order and local order agree
    inside a segment, so the tie-break is unambiguous.
  * `values[r, s] = x[r].gather(indices[r, s])`, bit for bit, in the INPUT
    dtype. No arithmetic is performed on the values.
  * `indices` is int64.
  * Inputs are guaranteed NaN-free by construction; `-inf`/`+inf` are legal.

There is no single library builtin for this; the timing anchor applies
torch.topk to the [R, G, S] segment view (that is the denominator only, and its
tie-break differs from the rule above, so it is never used for correctness).
Correctness is EXACT equality against this function.

The implementation reshapes to the [R, G, S] segment view, stable-descending
argsorts the last (length-S) axis, gathers the first k, and adds the segment
offset s*S to make indices global. Deliberately simple and slow.
"""

import torch


def topk_segmented_ref(x: torch.Tensor, k: int, segment: int):
    """Sequential ground truth for segmented descending top-k.

    x: [R, N] float32 or bfloat16, NaN-free. S = segment divides N. 1 <= k <= S.
    Returns (values[R, G, k] in x.dtype, indices[R, G, k] int64) with G = N // S
    and GLOBAL (row-relative) indices.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [R, N], got {tuple(x.shape)}")
    R, N = x.shape
    S, k = int(segment), int(k)
    if S <= 0 or N % S != 0:
        raise ValueError(f"segment={S} must be positive and divide N={N}")
    if not (1 <= k <= S):
        raise ValueError(f"k={k} out of range for segment length S={S}")
    G = N // S
    xr = x.contiguous().view(R, G, S)                 # segment view (no copy)
    # Stable descending sort within each length-S segment.
    local = torch.argsort(xr, dim=-1, descending=True, stable=True)[..., :k]  # [R,G,k]
    vals = torch.gather(xr, -1, local)                # [R,G,k], bit-exact
    seg_off = (torch.arange(G, device=x.device, dtype=torch.int64) * S).view(1, G, 1)
    idx_global = local.to(torch.int64) + seg_off       # global row positions
    return vals.contiguous(), idx_global.contiguous()


if __name__ == "__main__":
    # CPU-ONLY sanity check on tiny shapes: cross-validate against a brute-force
    # per-segment (value desc, index asc) order. No GPU/Triton is touched here.
    def _brute_seg(row, k, S):
        G = len(row) // S
        out = []
        for s in range(G):
            seg_pos = list(range(s * S, s * S + S))
            order = sorted(seg_pos, key=lambda i: row[i], reverse=True)[:k]
            out.append(order)
        return out

    inf, ninf = float("inf"), float("-inf")
    x = torch.tensor([
        [3.0, 1.0, 3.0, -0.0, 0.0, inf, ninf, 2.0, 5.0, 5.0, 1.0, 1.0],
        [-1.0, -1.0, -2.0, 0.0, 4.0, 4.0, 4.0, ninf, inf, 7.0, 2.0, 2.0],
    ], dtype=torch.float32)
    for S in (3, 4, 6, 12):
        G = x.shape[1] // S
        for k in range(1, S + 1):
            v, idx = topk_segmented_ref(x, k, S)
            assert v.shape == (x.shape[0], G, k) and idx.shape == (x.shape[0], G, k), (S, k, v.shape)
            assert idx.dtype == torch.int64 and v.dtype == torch.float32
            for r in range(x.shape[0]):
                want = _brute_seg(x[r].tolist(), k, S)
                assert idx[r].tolist() == want, (r, S, k, idx[r].tolist(), want)
                assert torch.equal(v[r].reshape(-1), x[r].gather(0, idx[r].reshape(-1)))
    vb, ib = topk_segmented_ref(x.to(torch.bfloat16), 2, 4)
    assert vb.dtype == torch.bfloat16 and ib.dtype == torch.int64
    print("topk_segmented reference CPU sanity OK: shapes/dtypes/global-index/bit-exact validated")
