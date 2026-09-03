# Task: implement a fast row-wise top-k over prime row length

`kernel.py` contains a working but deliberately slow implementation of descending top-k over rows whose length `N` is a **prime**. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `topk_prime_row(x, k)` with `x` of shape `[R, N]` in float32 or bfloat16 and `1 <= k <= N`. Its signature and semantics are fixed. There is no backward pass: this operator is graded forward-only.

## The operator

For each row `r`, `topk_prime_row(x, k)` returns `(values, indices)`, both `[R, k]`:

- `indices[r]` holds the positions of the `k` largest values in `x[r, :]`, ordered by **descending value**, in `int64`, in `[0, N)`.
- Ties are broken by **lower original index first**. `-0.0` and `+0.0` compare equal, so a mix of the two is a single tie group and the index rule orders it.
- `values = x.gather` at those positions, **bit for bit**: every returned value is the input element at the returned index, in the input dtype, with no arithmetic performed on it.
- Inputs are guaranteed NaN-free. `-inf` and `+inf` are legal.

Every hidden shape has **prime `N`** — from Mersenne `M13 = 8191` through the Fermat prime `F4 = 65537` and a large 20-bit prime `1048583`, up to a 1M-plus prime tenant. That is the hardness lever: `N` is coprime with every common SM tile, so any code that assumes `N` divides a block or pads to the next power of two must handle the tie-break rule correctly on the padded slots.

`(value descending, index ascending)` is a total order, so each input has exactly one correct answer, and correctness is graded as **exact equality** against `reference.py` — indices by integer equality, values at the bit level. There is no tolerance anywhere in this task.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not call any library selection or sorting routine. The banned names are listed under Constraints below and are checked by a word-boundary scan of your comment-stripped source. Reading library source is allowed and encouraged; calling it is not.
- your output — indices **and** bit-exact values — agrees with `reference.py` exactly, on a hidden shape set that is larger than, and different from, the shapes `bench.py` lets you measure. The hidden set includes a Mersenne prime `N`, the Fermat prime `F4`, `k = N`, per-row duplicate-saturated inputs, special values (`±inf`, `±0.0`, subnormals) planted inside a row, and one input with more than `2^31` total elements.
- **at least 60% of graded device time runs inside Triton kernels your own module declares.** This is measured by device-time attribution after the fact, not by reading your source. A submission that delegates the selection to framework operators scores zero on this gate.

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production kernel time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is CUDA `torch.topk` — CUB radix select — timed on the same tensors your kernel is timed on.** The anchor is a timing denominator only: its tie-break may differ from the semantics above, so matching its output is not the goal and would fail the correctness gate.

Both the magnitude **and the distribution shape** of the input are drawn at random from hidden distributions on every invocation, and the tensors that are timed are the same tensors that are checked.

## The correctness harness, in five stages

1. **Smoke.** One small prime-`N` input.
2. **Shape sweep.** The hidden graded prime-`N` shape set, across float32 and bfloat16.
3. **Stability.** Adversarial inputs: an all-equal plateau, rows salted with `-inf`, six decades of magnitude, a monotonic ramp, and eight-level duplicates. Gates on finiteness, not agreement.
4. **Determinism.** The same input three times, after a warmup, must give **bitwise identical** values and indices. The determinism input is saturated with duplicates on purpose.
5. **Edge cases.** Mersenne primes, Fermat prime, `k = N`, guaranteed-duplicate and special-value inputs, and the `> 2^31`-element tensor.

## The rubric check

Separately from the gates above, an LLM from a different model family reads your trajectory and your final `kernel.py` against a list of natural-language rubrics, and judges each one satisfied or not. **Every rubric must pass.** A single rubric judged unsatisfied scores zero.

## Requirements

**Exact tie-break.** Descending by value, lower original index first, `±0.0` one tie group. Values are moved by index, not recomputed. Indices are `int64`.

**Prime-length safety.** If you pad to the next power of two (or any convenient multiple), the padded slots must not appear in the output under any input, including one with a valid `-inf` entry. The negative controls include an nc_pad_to_pow2 arm that zeroes at G2 by exactly this failure mode.

**Full selection.** Every element of every row participates. No sampling, thresholding or early-exit path that can miss a winning element.

**Recomputation.** Every call recomputes from the tensor arguments it was passed.

**Data independence.** No branch on a magnitude summary, distribution estimate, duplicate count or sortedness probe.

**Determinism.** Three runs on one input must be bitwise identical.

**Index width.** At least one hidden input has more than `2^31` elements; use 64-bit arithmetic where an offset can exceed `2^31 - 1`.

**Address the kernel, not the reader.** Nothing in `kernel.py` or in your trajectory is addressed to the grader.

**Honest measurement.** Any speedup figure you report was produced by a run on the exact bytes you shipped, with CUDA events or an explicit `torch.cuda.synchronize` around the timed region.

## Working loop

`./bench.py` checks exact correctness (including a duplicate-saturated Mersenne shape), determinism, and your written-kernel share, then times you against `torch.topk` on published prime-`N` shapes. It does not tell you your score, and the published shapes are not the graded shapes. `--json <path>` writes machine-readable results.

## Disclosed thresholds

| rule | value |
|---|---|
| correctness | **exact**: indices by integer equality, values bit-identical to gathered input elements; no tolerance exists |
| row length | **prime** at every graded and correctness shape |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| forbidden calls | word-boundary scan over comment-stripped source; banned names listed under Constraints |
| dtypes | float32 and bfloat16, both checked **and** both timed |
| determinism | **3** runs bitwise identical after warmup |
| tie-break | lower original index first; `±0.0` one tie group |
| index width | one hidden input exceeds `2^31` total elements |
| rubric gate | **every** rubric must pass |

## Constraints

- One H100. **No internet.** torch, triton, einops and pytest are installed.
- Do not change the semantics, signature or output dtypes of `topk_prime_row`.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped, strings and docstrings count): `torch.topk`, `torch.sort`, `torch.argsort`, `torch.msort`, `torch.kthvalue`, `torch.unique`, `torch.median`, `torch.nanmedian`, `torch.quantile`, `torch.nanquantile`, and the attribute spellings `.topk`, `.sort`, `.argsort`, `.msort`, `.kthvalue`, `.unique`, `.median`, `.quantile` — which includes Triton's `tl.sort`. Top-1 primitives (`torch.max`, `torch.argmax`, `tl.max`, `tl.argmax`), reshape/`view`, and scan/histogram primitives (`tl.cumsum`, `tl.histogram`) are allowed.
- Numerical policy must be uniform: no branch on input magnitude, distribution, shape or seed that changes which algorithm runs.
- The gain must come from your own kernel work.
