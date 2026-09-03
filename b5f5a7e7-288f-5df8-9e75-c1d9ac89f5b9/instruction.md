# Task: implement a fast row-wise argmax with mixed input dtype

`kernel.py` contains a working implementation of row-wise argmax that supports float32, bfloat16, and float16 inputs. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `argmax_mixed_dtype(x)` with `x` of shape `[R, N]` in float32, bfloat16, OR float16. Its signature and semantics are fixed. There is no backward pass.

## The operator

For each row `r`, `argmax_mixed_dtype(x)` returns `indices[R]`, an int64 vector:

- `indices[r]` is the position in `[0, N)` of the largest value in `x[r, :]`, tie-break **lower original index first**, `-0.0 == +0.0` one tie group.
- Inputs are guaranteed NaN-free. `-inf` / `+inf` are legal on all three dtypes (a `+inf` wins its row).

Every graded shape is drawn under every dtype in its dtype list; each dtype is graded independently.

Correctness is **exact** — int64 equality on `reference.py`'s output. There is no tolerance.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- forbidden-symbol scan clean (see Constraints; `torch.argmax` and `.argmax` are banned in addition to the selection primitives).
- output matches `reference.py` exactly on the hidden graded set plus a correctness set that includes an `N = 1` degenerate (index always 0), a duplicate-saturated draw across every dtype, a specials draw, a unique-winner probe, and one input whose flat `R * N` exceeds `2^31`.
- at least **60%** of graded device time in Triton kernels your module declares.

Above the gates, the score is the geomean fraction of `(production kernel time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is CUDA `torch.argmax`** — a single-pass CUB tree reduction that sits close to HBM peak. The reachable fraction is small; that is the point.

## The correctness harness, in five stages

1. **Smoke.** A small row across every dtype in `SMOKE_SHAPE`'s dtype list.
2. **Shape sweep.** The hidden graded set: four shapes, three dtypes each.
3. **Stability.** All-equal, `-inf` salt, subnormal-saturated (dtype-specific magnitudes), a reverse ramp (winner at index 0), and eight-level duplicates.
4. **Determinism.** Same input three times, bitwise identical.
5. **Edge cases.** Large-prime N, N=1, duplicate-saturated, specials, unique-winner probe, and the `> 2^31`-element input.

## The rubric check

Every rubric must pass.

## Requirements

**Exact tie-break under every dtype.** Descending, lower original index first, `±0.0` one tie group. This must hold under fp16 (subnormals ~6e-8) and bf16 (subnormals ~9e-41) as well as fp32.

**Dtype-agnostic key derivation.** The composite-key strategy in the starter upcasts every input to fp32 before deriving the sortable key. That is one way to solve the mixed-dtype requirement in one kernel; you may write three specialised kernels instead, but if you keep the key at input precision on fp16 you will lose distinguishing bits and fail the tie-break on duplicate-saturated fp16 (the `nc_fp16_upcast` negative control zeroes at G2 by exactly this mistake).

**Full row.** Every column of every row participates.

**Recomputation, data independence, determinism.** As for every family task.

**Index width.** One hidden input's `R * N` exceeds `2^31`; use 64-bit arithmetic wherever a linear offset can exceed `2^31 - 1`.

**N = 1 degenerate.** The correctness set includes an `N = 1` shape. The answer is index 0 for every row; ensure your kernel does not divide by `N`, mod by zero, or launch with a zero-block grid.

**Address the kernel, not the reader.**

**Honest measurement.** CUDA events or explicit sync; timed on the exact bytes shipped.

## Working loop

`./bench.py` checks correctness on all three dtypes (bfloat16 by default in the CHECK shapes plus an fp16 correctness probe), determinism, and adoption, then times against `torch.argmax` on published shapes.

## Disclosed thresholds

| rule | value |
|---|---|
| correctness | **exact** int64 equality |
| dtypes | float32, bfloat16, **and float16** — every graded shape drawn under every listed dtype |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| forbidden calls | word-boundary scan over comment-stripped source; `torch.argmax` and `.argmax` also banned |
| determinism | **3** runs bitwise identical after warmup |
| tie-break | lower original index first; `±0.0` one tie group |
| index width | one hidden input's `R * N` exceeds `2^31` |
| rubric gate | **every** rubric must pass |

## Constraints

- One H100. **No internet.**
- Do not change the signature or output dtype of `argmax_mixed_dtype`.
- Forbidden names in `kernel.py` (word-boundary scan, comments stripped, strings/docstrings count): the selection primitives (`torch.topk`, `torch.sort`, `torch.argsort`, `torch.msort`, `torch.kthvalue`, `torch.unique`, `torch.median`, `torch.nanmedian`, `torch.quantile`, `torch.nanquantile`, and the `.` attribute spellings — includes `tl.sort`) PLUS `torch.argmax` and `.argmax`. Top-1 primitives (`torch.max`, `tl.max`, `tl.argmax`), reshape/`view`, and scan/histogram primitives (`tl.cumsum`, `tl.histogram`) are allowed.
- Numerical policy must be uniform across dtypes.
- The gain must come from your own kernel work.
