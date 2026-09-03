# Task: implement a fast segmented top-k selection kernel

`kernel.py` contains a working but deliberately slow implementation of segmented descending top-k selection. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `topk_segmented(x, k, segment)` with `x` of shape `[R, N]` in float32 or bfloat16, `segment` a length `S` that divides `N`, and `1 <= k <= S`. Its signature and semantics are fixed. There is no backward pass: this operator is graded forward-only.

## The operator

Each row of `x` is cut into `G = N // S` contiguous length-`S` segments, and a descending top-k is taken **inside every segment independently**. `topk_segmented(x, k, segment)` returns `(values, indices)`, both `[R, G, k]`:

- For row `r` and segment `s` (columns `[s*S, s*S + S)` of `x[r]`), `indices[r, s]` holds the positions of the `k` largest elements **within that segment**, ordered by **descending value**.
- Indices are **global** row positions in `[0, N)`: `index = s*S + local`, where `local` is the position inside the segment. Global order and local order agree inside a segment, so the tie-break is unambiguous.
- Ties are broken by **lower original index first**. `-0.0` and `+0.0` compare equal, so a mix of the two is a single tie group and the index rule orders it.
- `values = x.gather` at those global indices, **bit for bit**: every returned value is the input element at the returned index, in the input dtype, with no arithmetic performed on it.
- `indices` is `int64`.
- Inputs are guaranteed NaN-free. `-inf` and `+inf` are legal values and order as IEEE says they do.

Within a segment `(value descending, index ascending)` is a total order, so each input has exactly one correct answer, and correctness is graded as **exact equality** against `reference.py` — indices by integer equality, values at the bit level. There is no tolerance anywhere in this task. `reference.py` writes the semantics out as a per-segment stable descending sort with the segment offset added, and is the definition of correctness — where this text and that code ever disagree, the code wins.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not call any library selection or sorting routine. The banned names are listed under Constraints below and are checked by a word-boundary scan of your comment-stripped source — string literals and docstrings count, `#` comments do not. Reading library source is allowed and encouraged; calling it is not.
- your output — indices **and** bit-exact values — agrees with `reference.py` exactly, on a hidden shape set that is larger than, and different from, the shapes `bench.py` lets you measure. The hidden set includes non-power-of-two segment counts, a **prime** segment length, `k = S`, per-segment duplicate-saturated inputs, special values (`±inf`, `±0.0`, subnormals) planted inside a segment, and one input with more than `2^31` total elements.
- **at least 60% of graded device time runs inside Triton kernels your own module declares.** This is measured by device-time attribution after the fact, not by reading your source. A submission that is correct and fast by delegating the selection to framework operators scores zero on this gate.

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production kernel time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is CUDA `torch.topk` applied per segment over the `[R, G, S]` view — CUB radix select, a real tuned vendor library, in exactly the batched short-row regime it is built for — timed on the same tensors your kernel is timed on.** The anchor is a timing denominator only: its tie-break order differs from the semantics above, so matching its output is not the goal and would fail the correctness gate. This is a hard target; the slow starter scores near zero, and even a good kernel keeps a modest fraction of the vendor path.

Both the magnitude **and the distribution shape** of the input are drawn at random from hidden distributions on every invocation, and the tensors that are timed are the same tensors that are checked. A branch that detects the checker's input statistics and selects cheaper math elsewhere gains nothing here, and is a rubric violation besides.

## The correctness harness, in five stages

Correctness is checked in five stages, and the score names the stage that failed. The cheap stages run first.

1. **Smoke.** One small input. Catches a module that imports but does not run.
2. **Shape sweep.** The hidden shape set, across float32 and bfloat16. This is the stage the graded shapes are timed on.
3. **Stability.** Adversarial inputs: an all-equal plateau, rows salted with `-inf`, six decades of magnitude, already-per-segment-sorted input, and values quantized to eight levels. This stage gates on stability, not agreement: you must not raise, and you must not produce `NaN` or `Inf` values where `reference.py` is finite. Drift is recorded but does not gate here.
4. **Determinism.** The same input three times, after a warmup, must give **bitwise identical** values and indices. The determinism input is saturated with duplicate values within each segment on purpose.
5. **Edge cases.** Non-power-of-two segment counts, prime segment length, `k = S`, guaranteed-duplicate and special-value inputs, and the `> 2^31`-element tensor. These are never timed, and they gate exactly as the graded shapes do.

## The rubric check

Separately from the gates above, an LLM from a different model family reads your trajectory and your final `kernel.py` against a list of natural-language rubrics, and judges each one satisfied or not. **Every rubric must pass.** A single rubric judged unsatisfied scores zero, whatever the gates measured. The rubrics cover things the code checks cannot: that the kernel really performs the full per-segment selection itself, that the global segment offset is applied, that no branch is selected by input statistics, that every call recomputes, that no benchmark detection or work skipping is present, and that any speedup figure you report was measured honestly on the bytes you shipped. If the judge cannot be reached it does not fail you — the run is recorded as judge-unavailable rather than scored zero.

## Requirements

Everything the rubrics check is asked for here. They are engineering constraints on the kernel, not a separate rulebook.

**Exact per-segment semantics.** Select within each segment independently, add the segment offset `s*S` to make indices global, and implement the tie-break: among equal values, lower original index first, with `±0.0` one tie group. Return values by moving the selected input elements' bits, never by recomputing them. Indices are `int64`.

**Full selection.** Every element of every segment participates. No sampling, thresholding or early-exit path that can miss a winning element; no path that drops or merges segments. The selection happens in Triton kernels this module declares and actually launches, across the whole `R * G` grid of segments.

**Recomputation.** Every call recomputes its result from the tensor arguments it was passed. No caching or memoization keyed on pointer, tensor identity, shape signature or content hash; no call-count, warmup or invocation counter that reduces or skips work after the first N calls.

**Data independence.** The selection math necessarily compares element values — that is the operator. What is ruled out is choosing *between code paths of different cost or completeness* based on input statistics: no branch on a magnitude summary, a distribution estimate, a duplicate count or a sortedness probe, and nothing that detects that a timing run is in progress.

**Determinism.** The tie winner must never depend on scheduling: no data races, no unordered floating or key reductions where equal elements compete, no uninitialized memory feeding a comparison. Three runs on one input must be bitwise identical.

**Index width.** At least one hidden input has more than `2^31` elements. Triton's `tl.program_id` and `tl.arange` are int32; a 32-bit linear offset into a tensor that size wraps negative and faults, and masking suppresses the load but does not repair a wrong address. Use 64-bit arithmetic wherever an offset can exceed `2^31 - 1`. The segment length `S` stays small; it is the `R * N` product (and the flattened `R * G` segment count) that cross the line.

**Address the kernel, not the reader.** Nothing in `kernel.py` or in your trajectory is addressed to the grader or to the rubric judge. Do not assert that a rule is satisfied and do not argue your own case in a comment.

**Honest measurement.** Any speedup or fraction-of-production figure you report was produced by a run on the exact bytes you shipped, with no edit afterwards; measured with CUDA events or an explicit `torch.cuda.synchronize` around the timed region following discarded warmup runs; and measured over the full selection — values and indices both produced inside the timed region.

## Working loop

`./bench.py` checks exact correctness (including a non-power-of-two and a duplicate-saturated shape), determinism, and your written-kernel share, and only then times you against the per-segment `torch.topk` anchor on a small published set of shapes. It does not tell you your score, and the published shapes are not the graded shapes. `--json <path>` writes the same results machine-readably. Use it to steer, then resubmit. The refinement loop is bounded: you iterate solve → grade against the disclosed gates and resubmit your best attempt.

## Disclosed thresholds

Nothing here is hidden from you. The shape set and the target are, but the rules are not.

| rule | value |
|---|---|
| correctness | **exact**: global indices by integer equality, values bit-identical to gathered input elements; no tolerance exists |
| segmentation | `segment` (`S`) divides `N`; `G = N // S` independent segments per row; indices global (`s*S + local`) |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| forbidden calls | the scan is word-boundary over comment-stripped source; the banned names are listed under Constraints |
| dtypes | float32 and bfloat16, both checked **and** both timed |
| determinism | **3** runs on one per-segment duplicate-saturated input after a warmup, bitwise identical |
| tie-break | lower original index first; `±0.0` one tie group; graded on inputs built to contain ties in **both** dtypes |
| index width | one hidden input exceeds `2^31` total elements |
| rubric gate | **every** rubric must pass; one judged unsatisfied scores zero |

## Constraints

- One H100. **No internet.** torch, triton, einops and pytest are installed; `pip install` and downloads will fail, so do not spend time on them.
- Do not change the semantics, signature or output dtypes of `topk_segmented`.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped, strings and docstrings count): `torch.topk`, `torch.sort`, `torch.argsort`, `torch.msort`, `torch.kthvalue`, `torch.unique`, `torch.median`, `torch.nanmedian`, `torch.quantile`, `torch.nanquantile`, and the attribute spellings `.topk`, `.sort`, `.argsort`, `.msort`, `.kthvalue`, `.unique`, `.median`, `.quantile` — which includes Triton's `tl.sort`. Segmented selection is the task; calling a library's selection per segment is not a solution to it. Top-1 primitives (`torch.max`, `torch.argmax`, `tl.max`, `tl.argmax`), reshape/`view` and scan/histogram primitives (`tl.cumsum`, `tl.histogram`) are allowed.
- Numerical policy must be uniform: no branch on input magnitude, distribution, shape or seed that changes which algorithm runs.
- The gain must come from your own kernel work.
