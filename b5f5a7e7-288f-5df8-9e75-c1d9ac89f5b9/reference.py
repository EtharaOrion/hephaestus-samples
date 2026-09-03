"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is the k=1 degenerate row-wise argmax over a row with a hidden
input dtype drawn from {float32, bfloat16, float16}. That combination is
the hardness lever: fp16 argmax needs care around subnormals and the
narrower exponent range, and the same kernel must produce bit-exact
tie-break behaviour under all three formats.

    argmax_mixed_dtype(x[R, N]) -> indices[R] int64

Semantics, exactly:

  * For row r, `indices[r]` is the position of the largest value in
    x[r, :]. Ties broken by LOWER ORIGINAL INDEX first. `-0.0 == +0.0`
    one tie group.
  * Indices are int64 positions in [0, N). No values are returned; this
    is the k=1 degenerate fast path of the family top-k operator.
  * Inputs are guaranteed NaN-free; `-inf`/`+inf` are legal (a `+inf`
    wins its row unambiguously).
  * Input dtype is one of float32, bfloat16, float16, drawn per graded
    invocation from the hidden dtype set for that shape.

There is one correct answer per input, so correctness is graded as EXACT
equality against this function (int64 equality on indices). The `compare`
override in taskdef.py enforces this and rejects any dense-tolerance
escape hatch.

The implementation runs a stable descending argsort over each row and
returns the first index -- deliberately slow and pedagogical; the whole
point of this task is that any candidate strictly beats a row argsort.
"""

import torch


def argmax_mixed_dtype_ref(x: torch.Tensor):
    """Sequential ground truth for row-wise argmax with lower-index tie-break.

    x: [R, N] float32, bfloat16, or float16, NaN-free. Returns indices[R] int64.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [R, N], got {tuple(x.shape)}")
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"unsupported dtype {x.dtype}")
    order = torch.argsort(x, dim=-1, descending=True, stable=True)
    return order[:, 0].contiguous().to(torch.int64)


if __name__ == "__main__":

    def _brute(row):
        return max(range(len(row)), key=lambda i: (row[i], -i))

    inf, ninf = float("inf"), float("-inf")
    x = torch.tensor(
        [
            [3.0, 1.0, 3.0, -0.0, 0.0, inf, ninf, 2.0, 5.0, 5.0, 1.0, 1.0],
            [-1.0, -1.0, -2.0, 0.0, 4.0, 4.0, 4.0, ninf, inf, 7.0, 2.0, 2.0],
            [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
        ],
        dtype=torch.float32,
    )
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        xd = x.to(dt)
        idx = argmax_mixed_dtype_ref(xd)
        assert idx.dtype == torch.int64
        assert idx.shape == (x.shape[0],)
        for r in range(x.shape[0]):
            want = _brute(xd[r].float().tolist())
            assert int(idx[r]) == want, (dt, r, int(idx[r]), want)
    print(
        "argmax_mixed_dtype reference CPU sanity OK: fp32/bf16/fp16 tie-break validated"
    )
