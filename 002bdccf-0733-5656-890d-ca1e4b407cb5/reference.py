"""
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is RAGGED top-k: a descending top-k taken inside each row of a
flat tensor whose row boundaries are given by an int64 prefix-sum:

    topk_ragged(x[NNZ], seg_offsets[R+1] int64, k)
        -> (values[R, k], indices[R, k])

Semantics, exactly:

  * `seg_offsets` is a monotone non-decreasing int64 vector of length R+1
    with `seg_offsets[0] == 0` and `seg_offsets[R] == NNZ`. Row r consists
    of the flat positions in `[seg_offsets[r], seg_offsets[r+1])` and has
    length L_r = seg_offsets[r+1] - seg_offsets[r].
  * Every row is guaranteed to have L_r >= k by construction; row lengths
    are ARBITRARY (different from topk_segmented, which uses one fixed
    length S per row).
  * For row r, `indices[r]` holds the GLOBAL positions in [0, NNZ) of the
    k largest values within that row, ordered DESCENDING by value, ties
    broken by LOWER ORIGINAL GLOBAL INDEX first, `-0.0 == +0.0` one tie
    group. Because the row occupies a contiguous slice, global-order and
    local-order agree inside a row, so the tie-break is unambiguous.
  * `values[r, :] = x.gather(0, indices[r])`, bit for bit, in the INPUT
    dtype. No arithmetic is performed on the values.
  * `indices` is int64.
  * Inputs are guaranteed NaN-free; `-inf`/`+inf` are legal.

There is no single library builtin for ragged top-k. The timing anchor
pads all rows to the maximum length with `-inf` and calls `torch.topk`;
that is a denominator only, its `-inf` fill collides with legitimately
`-inf` valid entries and its tie-break can differ, so it is never used
for correctness. This reference is the definition.

The implementation slices each row and runs a stable descending argsort
over that slice. Sequential; no fused kernel. Deliberately simple.
"""

import torch


def topk_ragged_ref(x: torch.Tensor, seg_offsets: torch.Tensor, k: int):
    """Sequential ground truth for ragged descending top-k.

    x: [NNZ] float32 or bfloat16, NaN-free. seg_offsets: [R+1] int64,
    monotone, seg_offsets[0]=0, seg_offsets[-1]=NNZ, every row L_r >= k.
    1 <= k <= min row length. Returns (values[R, k] in x.dtype,
    indices[R, k] int64 GLOBAL positions).
    """
    if x.dim() != 1:
        raise ValueError(f"x must be 1D flat, got {tuple(x.shape)}")
    if seg_offsets.dtype != torch.int64:
        raise ValueError(f"seg_offsets must be int64, got {seg_offsets.dtype}")
    if seg_offsets.dim() != 1:
        raise ValueError(f"seg_offsets must be 1D, got {tuple(seg_offsets.shape)}")
    R = int(seg_offsets.shape[0]) - 1
    if R < 1:
        raise ValueError(f"need >= 1 row, seg_offsets len={seg_offsets.shape[0]}")
    off = seg_offsets.tolist()
    NNZ = int(x.shape[0])
    if off[0] != 0 or off[-1] != NNZ:
        raise ValueError(f"seg_offsets must span [0, {NNZ}], got [{off[0]}, {off[-1]}]")
    k = int(k)
    vals = torch.empty(R, k, dtype=x.dtype, device=x.device)
    idx = torch.empty(R, k, dtype=torch.int64, device=x.device)
    for r in range(R):
        lo, hi = off[r], off[r + 1]
        if hi - lo < k:
            raise ValueError(f"row {r} has length {hi - lo} < k={k}")
        row = x[lo:hi]
        order_local = torch.argsort(row, descending=True, stable=True)[:k]
        idx[r] = order_local.to(torch.int64) + lo
        vals[r] = x.gather(0, idx[r])
    return vals.contiguous(), idx.contiguous()


if __name__ == "__main__":

    def _brute(row, k):
        return sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:k]

    inf, ninf = float("inf"), float("-inf")
    # Three rows of lengths 5, 3, 8; NNZ = 16.
    x = torch.tensor(
        [
            3.0,
            1.0,
            3.0,
            -0.0,
            0.0,  # row 0
            inf,
            ninf,
            2.0,  # row 1
            5.0,
            5.0,
            1.0,
            1.0,
            3.0,
            4.0,
            5.0,
            5.0,  # row 2
        ],
        dtype=torch.float32,
    )
    seg = torch.tensor([0, 5, 8, 16], dtype=torch.int64)
    for k in (1, 2, 3):
        v, idx = topk_ragged_ref(x, seg, k)
        R = seg.shape[0] - 1
        assert v.shape == (R, k) and idx.shape == (R, k)
        assert idx.dtype == torch.int64 and v.dtype == torch.float32
        off = seg.tolist()
        for r in range(R):
            row = x[off[r] : off[r + 1]].tolist()
            want_local = _brute(row, k)
            want_global = [i + off[r] for i in want_local]
            assert idx[r].tolist() == want_global, (r, k, idx[r].tolist(), want_global)
            assert torch.equal(v[r], x.gather(0, idx[r]))
    vb, ib = topk_ragged_ref(x.to(torch.bfloat16), seg, 2)
    assert vb.dtype == torch.bfloat16 and ib.dtype == torch.int64
    print(
        "topk_ragged reference CPU sanity OK: variable lengths / global indices / bit-exact validated"
    )
