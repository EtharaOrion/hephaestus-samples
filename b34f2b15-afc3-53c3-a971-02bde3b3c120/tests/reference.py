"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is a row-wise DESCENDING top-k over VERY LARGE rows (N in the
256K to 4M range) with SMALL k (16 to 128). The huge N is the hardness
lever: the row does not fit in shared memory, an iterative masked-max
extraction pays O(k * N) HBM bandwidth (crushing for small k), and even a
radix select must land its histogram/scatter work in a small number of
streaming passes.

    topk_giant_row(x[R, N], k) -> (values[R, k], indices[R, k])

Semantics, exactly:

  * For row r, `indices[r]` holds the positions of the k LARGEST values in
    x[r, :], ordered DESCENDING by value, ties broken by LOWER ORIGINAL
    INDEX first, `-0.0 == +0.0` one tie group.
  * Indices are int64 positions in [0, N).
  * `values[r, :] = x[r].gather(0, indices[r])`, bit for bit, in the INPUT
    dtype. No arithmetic is performed on the values.
  * Inputs are guaranteed NaN-free; `-inf`/`+inf` are legal.
  * N >= 262144 at every graded shape; 1 <= k <= 128 at every graded shape.

There is one correct answer per input, so correctness is graded as EXACT
equality against this function -- indices by integer equality, values at
the bit level (via a raw-bits view). The `compare` override in taskdef.py
enforces this and rejects any dense-tolerance escape hatch.

The implementation is a stable descending argsort over the whole row
followed by a gather of the first k. Deliberately simple and slow; on the
graded shapes it materializes multi-hundred-megabyte permutations, which
is exactly why this task exists.
"""

import torch


def topk_giant_row_ref(x: torch.Tensor, k: int):
    """Sequential ground truth for giant-N descending row-wise top-k.

    x: [R, N] float32 or bfloat16, NaN-free. 1 <= k <= N. Returns
    (values[R, k] in x.dtype, indices[R, k] int64).
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [R, N], got {tuple(x.shape)}")
    R, N = x.shape
    k = int(k)
    if not (1 <= k <= N):
        raise ValueError(f"k={k} out of range for N={N}")
    order = torch.argsort(x, dim=-1, descending=True, stable=True)  # [R, N]
    idx = order[:, :k].contiguous().to(torch.int64)
    vals = torch.gather(x, -1, idx)
    return vals.contiguous(), idx


if __name__ == "__main__":
    # CPU-ONLY sanity on tiny shapes (semantics do not depend on N being large).
    def _brute(row, k):
        return sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:k]

    inf, ninf = float("inf"), float("-inf")
    x = torch.tensor(
        [
            [
                3.0,
                1.0,
                3.0,
                -0.0,
                0.0,
                inf,
                ninf,
                2.0,
                5.0,
                5.0,
                1.0,
                1.0,
                3.0,
                4.0,
                5.0,
                5.0,
            ],
            [
                -1.0,
                -1.0,
                -2.0,
                0.0,
                4.0,
                4.0,
                4.0,
                ninf,
                inf,
                7.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                3.0,
            ],
        ],
        dtype=torch.float32,
    )
    for k in (1, 3, 6, 16):
        v, idx = topk_giant_row_ref(x, k)
        assert v.shape == (x.shape[0], k) and idx.shape == (x.shape[0], k)
        assert idx.dtype == torch.int64 and v.dtype == torch.float32
        for r in range(x.shape[0]):
            want = _brute(x[r].tolist(), k)
            assert idx[r].tolist() == want, (r, k, idx[r].tolist(), want)
            assert torch.equal(v[r], x[r].gather(0, idx[r]))
    vb, ib = topk_giant_row_ref(x.to(torch.bfloat16), 2)
    assert vb.dtype == torch.bfloat16 and ib.dtype == torch.int64
    print(
        "topk_giant_row reference CPU sanity OK: shapes/dtypes/tie-break/bit-exact validated"
    )
