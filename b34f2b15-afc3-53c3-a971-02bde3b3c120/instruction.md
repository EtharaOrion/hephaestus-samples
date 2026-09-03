# Task: implement a fast row-wise top-k over very large rows

`kernel.py` contains a working but deliberately slow implementation of descending top-k over rows in the multi-megabyte range with small `k`. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `topk_giant_row(x, k)` with `x` of shape `[R, N]` in float32 or bfloat16 and `1 <= k <= N`. Its signature and semantics are fixed. There is no backward pass.

## The operator

For each row `r`, `topk_giant_row(x, k)` returns `(values, indices)`, both `[R, k]`:

- `indices[r]` holds the positions of the `k` largest values in `x[r, :]`, ordered by **descending value**, in `int64`, in `[0, N)`.
- Ties are broken by **lower original index first**. `-0.0 == +0.0` one tie group.
- `values = x.gather` at those positions, **bit for bit**.
- Inputs are guaranteed NaN-free. `-inf` / `+inf` are legal.

Every graded shape has `N >= 262144` (up to a 1M-plus tenant, and a 4M-column tenant in the correctness set) and `k in [16, 128]`. That is the hardness lever: at `k << sqrt(N)`, the operator's theoretical lower bound is one streaming read of the row, and any candidate that pays multiple reads across the row (a naive iterative masked max, for instance) loses badly.

`(value desc, index asc)` is a total order, so each input has exactly one correct answer, and correctness is graded as **exact equality** against `reference.py` — indices by integer equality, values at the bit level. There is no tolerance.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- forbidden-symbol scan is clean (see Constraints).
- output matches `reference.py` exactly on the hidden graded shapes plus a correctness-only set that includes a 4M-column row, a k=1 degenerate (argmax), and one input with more than `2^31` total elements.
- at least **60%** of graded device time runs inside Triton kernels your own module declares.

Above the gates, the score is the geomean fraction of `(production kernel time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is CUDA `torch.topk` — CUB radix select — timed on the same tensors your kernel is timed on.** CUB collapses to a single streaming read plus a small number of shallow radix passes at these shapes, so it sits close to HBM peak. The reachable fraction is small; that is the point.

## The correctness harness, in five stages

1. **Smoke.** One 256K-column row.
2. **Shape sweep.** The hidden graded set (N up to 2M, k up to 128), across float32 and bfloat16.
3. **Stability.** Adversarial inputs: an all-equal plateau, `-inf` salt, six-decade magnitude, an already-reverse-sorted row (trivially at the ceiling — do not detect it), and eight-level duplicates.
4. **Determinism.** Same input three times, bitwise identical.
5. **Edge cases.** A 4M-column row, a large-prime N, a k=1 argmax degenerate, per-row duplicate saturation, special values, and the `> 2^31`-element probe.

## The rubric check

An LLM from a different model family reads your trajectory and final `kernel.py` against a list of rubrics. Every rubric must pass.

## Requirements

**Exact tie-break.** Descending, lower original index first, `±0.0` one tie group. Values moved by index. Indices `int64`.

**Full selection.** Every element of every row participates. No sampling, thresholding or early-exit that can miss a winning element; no "top-k inside a random subrange" trick. The negative controls include an nc_first_block_only arm that zeroes at G2 by exactly this failure mode.

**Recomputation.** Every call recomputes from the tensor arguments passed.

**Data independence.** No branch on magnitude summary, distribution estimate, duplicate count, sortedness probe or shape signature that selects a cheaper path.

**Determinism.** Three runs bitwise identical.

**Index width.** At least one hidden input has more than `2^31` elements. Use 64-bit arithmetic wherever a linear offset can exceed `2^31 - 1`.

**k=1 fast path is allowed but must be uniform.** A k=1 branch that reduces to a single argmax is fine; anything that also skips the tie-break rule (e.g. picks a value-only max without breaking ties on index) fails the correctness gate on the k=1 correctness shape.

**Address the kernel, not the reader.** No text in `kernel.py` or the trajectory argues its own case.

**Honest measurement.** CUDA events or explicit `torch.cuda.synchronize`; measured on the exact bytes you ship.

## Working loop

`./bench.py` checks exact correctness, determinism, and adoption, then times you against `torch.topk` on published giant-N shapes. It does not tell you your score.

## Disclosed thresholds

| rule | value |
|---|---|
| correctness | **exact**: indices by integer equality, values bit-identical to gathered input elements |
| shape range | `N >= 262144` at every graded shape (up to 4M in correctness); `k in [16, 128]` graded |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| forbidden calls | word-boundary scan over comment-stripped source |
| dtypes | float32 and bfloat16 |
| determinism | **3** runs bitwise identical after warmup |
| tie-break | lower original index first; `±0.0` one tie group |
| index width | one hidden input exceeds `2^31` total elements |
| rubric gate | **every** rubric must pass |

## Constraints

- One H100. **No internet.**
- Do not change the semantics, signature or output dtypes of `topk_giant_row`.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped, strings and docstrings count): `torch.topk`, `torch.sort`, `torch.argsort`, `torch.msort`, `torch.kthvalue`, `torch.unique`, `torch.median`, `torch.nanmedian`, `torch.quantile`, `torch.nanquantile`, and the attribute spellings `.topk`, `.sort`, `.argsort`, `.msort`, `.kthvalue`, `.unique`, `.median`, `.quantile`. Top-1 primitives (`torch.max`, `torch.argmax`, `tl.max`, `tl.argmax`), reshape/`view`, and scan/histogram primitives (`tl.cumsum`, `tl.histogram`) are allowed.
- Numerical policy must be uniform.
- The gain must come from your own kernel work.
