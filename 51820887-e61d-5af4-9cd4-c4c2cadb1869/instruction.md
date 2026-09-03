# Task: implement a fused row-wise softmax + top-k

`kernel.py` contains a working but deliberately unfused implementation of a softmax-then-top-k operator. Your job is to fuse the two reductions and make it fast.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `fused_softmax_topk(logits, k)` with `logits` of shape `[R, N]` in float32 or bfloat16 and `1 <= k <= N`. Its signature and semantics are fixed. There is no backward pass.

## The operator

For each row `r`, `fused_softmax_topk` returns `(probs, indices)`, both `[R, k]`:

1. **Canonical fp32 softmax** over the row:
   ```
   m   = logits.max(dim=-1, keepdim=True).values.float()
   e   = (logits.float() - m).exp()
   s   = e.sum(dim=-1, keepdim=True)
   probs_full = e / s          # [R, N] fp32
   ```
   The intermediate arithmetic is fp32 regardless of input dtype.
2. **Top-k of the row**: `indices[r]` holds the positions of the k largest values of `logits[r]` (equivalently, of `probs_full[r]`), ordered by **descending value**, tie-break **lower original index first**, `-0.0 == +0.0` one tie group. Softmax is strictly increasing, so top-k of the logits IS top-k of the probs; tie groups are preserved.
3. `probs[r, :] = probs_full[r].gather(0, indices[r])` in **float32**.
4. `indices` is `int64`.

Inputs are guaranteed NaN-free; `-inf` and `+inf` are legal on logits (a `+inf` logit yields probability 1 at that index).

Correctness is graded with a **mixed** rule enforced by the `compare` override in the taskdef:

- **indices**: exact integer equality against `reference.py`.
- **probs**: dtype-dependent tolerance — float32 `atol=1e-6, rtol=1e-5`; bfloat16 `atol=5e-4, rtol=5e-3`.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- forbidden-symbol scan clean. The banned list includes the softmax spellings (`torch.softmax`, `F.softmax`, `torch.nn.functional.softmax`, `.softmax`, `log_softmax`, `_softmax`) in addition to the selection primitives — the softmax reduction must live in your own kernel path, not be delegated.
- output matches `reference.py` on the hidden graded set plus a correctness set that includes a prime-N draw, a k=1 degenerate, k == N, duplicate-saturated, specials, a peaked (near-onehot) row, and a `> 2^31`-element probe.
- at least **60%** of graded device time runs inside Triton kernels your own module declares.

Above the gates, the score is the geomean fraction of `(production kernel time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is `torch.softmax + torch.topk`** — a three-pass composition that reads the row three times. A fused kernel reads the row once and folds the two reductions together; the reachable fraction reflects that gap.

## The correctness harness, in five stages

1. **Smoke.** One small vocab-scale input.
2. **Shape sweep.** Hidden graded LLM-vocab-scale shapes (N up to 200k), across float32 and bfloat16.
3. **Stability.** All-equal, `-inf` salt, large magnitude range (drift-tests fp32 accumulator vs bf16 accumulator), a peaked (near-onehot) row, eight-level duplicates.
4. **Determinism.** Same input three times, bitwise identical (indices) and within tolerance (probs).
5. **Edge cases.** Prime N, k=1, k=N, duplicate-saturated, specials, peaked, and the > 2^31-element probe.

## The rubric check

Every rubric must pass.

## Requirements

**Fused reduction.** The softmax reduction (max + sum-exp) must be fused into the same pass(es) as the top-k selection, or into a small number of your own kernels — no `torch.softmax` call is available and no composition through it will pass the adoption floor.

**fp32 accumulator for softmax.** The bf16 tolerance is set so that a candidate keeping the softmax accumulator in bf16 will drift outside it on bf16 inputs. Use fp32 for the max, exp, and sum; the negative controls include `nc_bf16_accumulator` that zeroes at G2 by exactly this mistake.

**Exact tie-break on logits.** Descending value, lower original index first, `±0.0` one tie group. Softmax preserves order, so implement the tie-break on the LOGIT KEY, not on the probs (which lose distinguishing bits under the fp32 divide).

**Full row.** Every column of every row participates in both the softmax reduction and the top-k selection.

**No branch on input statistics.** Even though max-subtract is a magnitude-dependent operation, it must be applied uniformly — no path that skips the max-subtract on "small" rows or uses a different reduction order on "peaked" rows.

**k=1 fast path is allowed but must be uniform.** A k=1 branch that returns argmax + its softmax probability is fine as long as it uses the same fp32 reduction policy.

**Determinism.** Three runs must give bitwise identical indices; probs must be equal within the tolerance (but a deterministic kernel will give bitwise identical probs too).

**Index width.** One hidden input exceeds `2^31` total elements.

**Address the kernel, not the reader.**

**Honest measurement.** CUDA events or explicit sync; timed on the exact bytes shipped; timing covers the full selection AND softmax.

## Working loop

`./bench.py` checks correctness, determinism, and adoption, then times against the `torch.softmax + torch.topk` composition on published LLM-vocab shapes.

## Disclosed thresholds

| rule | value |
|---|---|
| indices | **exact** integer equality |
| probs | fp32: `atol=1e-6, rtol=1e-5`; bf16: `atol=5e-4, rtol=5e-3` |
| probs dtype | always `torch.float32` regardless of input dtype |
| softmax policy | canonical fp32 max-subtract + exp + sum-normalize |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| forbidden calls | word-boundary scan; softmax spellings also banned |
| dtypes | float32 and bfloat16 |
| determinism | **3** runs bitwise identical (indices), within tol (probs) |
| tie-break | lower original index first; `±0.0` one tie group |
| index width | one hidden input exceeds `2^31` total elements |
| rubric gate | **every** rubric must pass |

## Constraints

- One H100. **No internet.**
- Do not change the signature, semantics or output dtypes of `fused_softmax_topk`.
- Forbidden names in `kernel.py` (word-boundary scan, comments stripped, strings/docstrings count): all selection primitives (`torch.topk`, `torch.sort`, `torch.argsort`, `torch.msort`, `torch.kthvalue`, `torch.unique`, `torch.median`, `torch.nanmedian`, `torch.quantile`, `torch.nanquantile`, and the `.` attribute spellings — includes `tl.sort`) PLUS the softmax spellings (`torch.softmax`, `F.softmax`, `torch.nn.functional.softmax`, `.softmax`, `log_softmax`, `_softmax`). Top-1 primitives (`torch.max`, `torch.argmax`, `tl.max`, `tl.argmax`, `torch.exp`, `torch.log`, `tl.exp`, `tl.log`), reshape/`view`, and scan/histogram primitives (`tl.cumsum`, `tl.histogram`) are allowed.
- The gain must come from your own kernel work.
