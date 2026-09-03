"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is a row-wise DESCENDING top-k over rows whose length N is a
PRIME (guaranteed non-power-of-two by construction). The prime shape is the
hardness lever: bitonic and radix-select kernels tuned for power-of-two n
must virtually pad or mask against a length that is coprime with 2^m and
never divides evenly by any common tile.

    topk_prime_row(x[R, N], k) -> (values[R, k], indices[R, k])

Semantics, exactly:

  * For row r, `indices[r]` holds the positions of the k LARGEST values in
    x[r, :], ordered DESCENDING by value, ties broken by LOWER ORIGINAL
    INDEX first, `-0.0 == +0.0` one tie group.
  * Indices are int64 positions in [0, N).
  * `values[r, :] = x[r].gather(0, indices[r])`, bit for bit, in the INPUT
    dtype. No arithmetic is performed on the values.
  * Inputs are guaranteed NaN-free; `-inf`/`+inf` are legal.
  * N is guaranteed PRIME >= 5 at every graded and correctness shape.
    1 <= k <= N.

There is one correct answer per input, so correctness is graded as EXACT
equality against this function -- indices by integer equality, values at the
bit level (via a raw-bits view). The `compare` override in taskdef.py enforces
this and rejects any dense-tolerance escape hatch.

The implementation is a stable descending argsort over the whole row followed
by a gather of the first k. Deliberately simple and slow.
"""

import torch


def topk_prime_row_ref(x: torch.Tensor, k: int):
    """Sequential ground truth for prime-N descending row-wise top-k.

    x: [R, N] float32 or bfloat16, NaN-free, with N a PRIME. 1 <= k <= N.
    Returns (values[R, k] in x.dtype, indices[R, k] int64).
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [R, N], got {tuple(x.shape)}")
    R, N = x.shape
    k = int(k)
    if not (1 <= k <= N):
        raise ValueError(f"k={k} out of range for N={N}")
    if N < 2 or not _is_prime(N):
        raise ValueError(f"topk_prime_row requires N prime and >= 2, got N={N}")
    # Stable descending: (value desc, index asc) is a total order.
    order = torch.argsort(x, dim=-1, descending=True, stable=True)  # [R, N]
    idx = order[:, :k].contiguous().to(torch.int64)
    vals = torch.gather(x, -1, idx)
    return vals.contiguous(), idx


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


if __name__ == "__main__":
    # CPU-ONLY sanity check on tiny PRIME shapes: cross-validate against a
    # brute-force (value desc, index asc) order. No GPU/Triton is touched here.
    def _brute(row, k):
        return sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:k]

    inf, ninf = float("inf"), float("-inf")
    for N in (5, 7, 11, 13, 17, 23, 29, 31):
        assert _is_prime(N)
    # A row of length 13 (prime) with duplicates, +/-0, +/-inf.
    x = torch.tensor(
        [
            [3.0, 1.0, 3.0, -0.0, 0.0, inf, ninf, 2.0, 5.0, 5.0, 1.0, 1.0, 3.0],
            [-1.0, -1.0, -2.0, 0.0, 4.0, 4.0, 4.0, ninf, inf, 7.0, 2.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    N = x.shape[1]
    assert _is_prime(N), N
    for k in (1, 3, 6, 13):
        v, idx = topk_prime_row_ref(x, k)
        assert v.shape == (x.shape[0], k) and idx.shape == (x.shape[0], k)
        assert idx.dtype == torch.int64 and v.dtype == torch.float32
        for r in range(x.shape[0]):
            want = _brute(x[r].tolist(), k)
            assert idx[r].tolist() == want, (r, k, idx[r].tolist(), want)
            assert torch.equal(v[r], x[r].gather(0, idx[r]))
    vb, ib = topk_prime_row_ref(x.to(torch.bfloat16), 2)
    assert vb.dtype == torch.bfloat16 and ib.dtype == torch.int64
    print(
        "topk_prime_row reference CPU sanity OK: prime-N shapes/dtypes/tie-break/bit-exact validated"
    )
