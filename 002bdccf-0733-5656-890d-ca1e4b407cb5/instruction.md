# Task: implement a fast ragged (variable-length) top-k selection

`kernel.py` contains a working but deliberately slow implementation of descending top-k over rows whose lengths vary, described by an int64 prefix-sum. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `topk_ragged(x, seg_offsets, k)`:

- `x` is a **1D flat** tensor of shape `[NNZ]` in float32 or bfloat16.
- `seg_offsets` is an int64 vector of shape `[R+1]`, monotone non-decreasing, with `seg_offsets[0] == 0` and `seg_offsets[R] == NNZ`. Row `r` is the slice `x[seg_offsets[r] : seg_offsets[r+1]]` of length `L_r = seg_offsets[r+1] - seg_offsets[r]`.
- `1 <= k <= min L_r`. Every row is guaranteed to satisfy `L_r >= k`.

There is no backward pass.

## The operator

For each row `r`, `topk_ragged` returns `(values, indices)`, both `[R, k]`:

- `indices[r]` holds the **global** positions in `[0, NNZ)` of the `k` largest values inside row `r`, ordered by **descending value**, tie-break **lower original global index first**, `-0.0 == +0.0` one tie group.
- `values = x.gather(0, indices[r])`, bit for bit, in the input dtype.
- `indices` is `int64`.

Because each row is a contiguous slice, global order and local order agree inside a row, so the tie-break rule is unambiguous.

Row lengths are ARBITRARY per invocation. Every graded shape carries `Lmin`, `Lmax` bounds and draws lengths uniformly from `[Lmin, Lmax]`, so the row-length distribution is different every call — there is no fixed padded shape a candidate can precompute.

`(value desc, index asc)` is a total order, so each input has exactly one correct answer, and correctness is graded as **exact equality** — indices by integer equality, values at the bit level. There is no tolerance.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- forbidden-symbol scan clean (see Constraints).
- output matches `reference.py` exactly on the hidden graded shape set, plus a correctness-only set that includes: a prime-length row where `k == L`, a 4-column row, a duplicate-saturated draw, a specials draw, and one input whose flat `NNZ` exceeds `2^31` (int64 offsets mandatory).
- at least **60%** of graded device time in Triton kernels your module declares.

Above the gates, the score is the geomean fraction of `(production kernel time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator pads every row to `Lmax` with `-inf` and calls `torch.topk` over the padded tensor.** That anchor pays a bandwidth tax proportional to the length skew of every batch — a batch with one long row and many short ones spends most of its time on padding. Your kernel need not pay that tax, and the reachable fraction is largest on the most-skewed hidden shape (g4).

## The correctness harness, in five stages

1. **Smoke.** A small ragged batch (R=32, L in [128, 512], k=8).
2. **Shape sweep.** The hidden graded ragged set, across float32 and bfloat16.
3. **Stability.** All-equal plateau, `-inf` salt, six-decade magnitude, extreme length skew (one giant row plus many short), and eight-level duplicates. Gates on finiteness only.
4. **Determinism.** Same input three times, bitwise identical values and indices.
5. **Edge cases.** `k == L`, 4-column rows, per-row duplicate saturation, specials in row 0, and the `> 2^31`-element probe.

## The rubric check

An LLM from a different model family reads your trajectory and final `kernel.py`. Every rubric must pass.

## Requirements

**Exact per-row semantics.** Select within each row's own slice; add the row's global offset `seg_offsets[r]` to translate local into global positions. The negative controls include an `nc_missing_row_offset` arm that zeroes at G2 by omitting the offset addition.

**No fixed-length assumption.** The rows are ragged. The negative controls include `nc_fixed_length_assumption` that hardcodes row 0's length for every row — zeroes at G2.

**No pad-to-max necessary.** You may pad if you want, but you must not pay the pad tax the anchor pays. If you pad you must still produce the exact tie-break rule under all inputs (a valid `-inf` entry ties any padding sentinel, and the lower-index rule then puts the padded index into your output — a bug).

**Full selection.** Every element of every row participates.

**Recomputation, data independence, determinism, index width.** As in every family task.

**k=1 fast path is allowed but must be uniform.** Any argmax-only path must still break ties by lower original global index.

**Address the kernel, not the reader.**

**Honest measurement.** CUDA events or explicit sync; timed on the exact bytes shipped.

## Working loop

`./bench.py` checks exact correctness (including a length-skew shape), determinism, and adoption, then times against the pad-to-max `torch.topk` anchor on published ragged shapes.

## Disclosed thresholds

| rule | value |
|---|---|
| correctness | **exact**: indices by integer equality, values bit-identical to gathered input elements |
| row structure | ragged, `seg_offsets` int64 prefix-sum; every row `L_r >= k` |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| forbidden calls | word-boundary scan over comment-stripped source |
| dtypes | float32 and bfloat16 |
| determinism | **3** runs bitwise identical after warmup |
| tie-break | lower original global index first; `±0.0` one tie group |
| index width | one hidden input's flat NNZ exceeds `2^31` |
| rubric gate | **every** rubric must pass |

## Constraints

- One H100. **No internet.**
- Do not change the signature or output dtypes.
- Forbidden names (word-boundary scan over comment-stripped source; strings and docstrings count): `torch.topk`, `torch.sort`, `torch.argsort`, `torch.msort`, `torch.kthvalue`, `torch.unique`, `torch.median`, `torch.nanmedian`, `torch.quantile`, `torch.nanquantile`, and the attribute spellings `.topk`, `.sort`, `.argsort`, `.msort`, `.kthvalue`, `.unique`, `.median`, `.quantile` (which includes `tl.sort`). Top-1 primitives (`torch.max`, `torch.argmax`, `tl.max`, `tl.argmax`), reshape/`view`, and scan/histogram primitives (`tl.cumsum`, `tl.histogram`) are allowed.
- Numerical policy must be uniform.
- The gain must come from your own kernel work.
