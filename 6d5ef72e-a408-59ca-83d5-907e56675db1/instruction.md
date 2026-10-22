# Task: implement a fast routed fused mixture-of-experts kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of a routed MoE layer — the routing decision lives INSIDE the operator. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `fused_moe(x, router_w, router_b, w1, w2, A)` with `x: [T, D]`, `router_w: [D, E]` (same dtype as `x`), `router_b: [E]` float32, `w1: [E, D, 2F]`, `w2: [E, F, D]`, and `A` the number of experts per token. It returns `y: [T, D]` in the input dtype. Its signature and semantics are fixed. The graded surface is forward AND backward: autograd must work through your implementation, and the gradients of `x`, `router_w`, `w1` and `w2` are graded alongside the output. `router_b` and `A` carry no gradient.

## The operator

Everything happens inside the one call — scores, selection, dispatch, expert compute, combine:

- **Scores.** `s = sigmoid(x @ router_w)`, computed in float32 from the input tensors.
- **Selection.** Each token takes the top `A` experts by the BIASED score `s + router_b`. Ties are broken by **lower expert index first**; `-0.0` and `+0.0` compare equal. `router_b` is a selection bias only: it decides which experts are picked and then vanishes — it never enters the combine weights, and it has no gradient (selection is piecewise-constant in it).
- **Combine weights.** `w_a = s[t, sel_a] / sum_a s[t, sel_a]` — the RAW sigmoid scores of the selected experts, normalized. Not the biased scores.
- **Experts.** `f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]`, and `y[t] = sum_a w_a * f(sel_a, x[t])`.
- **Arithmetic.** Inputs are cast up to float32; every matmul, the sigmoid, the silu, the normalization and the combine accumulate in float32; only the final `y` is cast back to the input dtype.
- **Gradients.** Selection indices are treated as locally constant; gradients flow through the gathered raw scores of the selected experts (through the sigmoid into `router_w` and `x`) and through the expert computation. `reference.py`'s autograd graph is the definition of those gradients.

`reference.py` writes all of this out as a slow expert-by-expert loop and is the definition of correctness — where this text and that code ever disagree, the code wins.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not call a grouped-GEMM library or an MoE serving stack. The banned names are listed under Constraints and are checked by a word-boundary scan of your comment-stripped source — string literals and docstrings count, `#` comments do not. Reading library source is allowed and encouraged; calling it is not.
- your output **and all four gradients** agree with `reference.py` under the disclosed per-dtype tolerances, on a hidden shape set that is larger than, and different from, the shapes `bench.py` lets you measure. The hidden set spans expert counts from 8-ish to 160-ish, `A` from 1 to `E`, non-power-of-two `T`, `D`, `F` and `E`, and inputs built to contain **exact** score ties in both dtypes.
- **at least 60% of graded forward+backward device time runs inside Triton kernels your own module declares.** This is measured by device-time attribution after the fact, not by reading your source. One narrow exception is carved out and disclosed: `torch.topk` / `torch.sort` / `torch.argsort` on the small `[T, E]` score matrix or the `[T*A]` routing metadata are permitted — routing bookkeeping is a rounding error of the runtime, and the floor keeps the expert compute, dispatch arithmetic and combine in kernels you wrote. Framework grouped or batched matmuls are not the permitted kind of call.

Above those gates, the score is the geometric mean across the hidden graded shapes — float32 and bfloat16 rows both — of `(production time) / (your time)` for one forward+backward invocation, normalized against a hidden target and saturating at 1. **The denominator is a production composition: the router matmul plus integer-key top-A in torch, feeding a `torch._grouped_mm` expert pipeline over tokens sorted by expert — a real CUTLASS grouped-GEMM path, timed on the same tensors your kernel is timed on.** Matching the slow starter scores near zero.

Three properties of the graded draws are disclosed because they shape correct solutions:

- **The decision margin.** On every graded draw, each token's `A`-th and `(A+1)`-th biased scores are either exactly equal (a manufactured tie, decided by the index rule) or separated by at least `1e-4`. Float32 score arithmetic sits orders of magnitude inside that margin; TF32 or bfloat16 score arithmetic does not — it flips selections and fails correctness outright. There is no reward for guessing scores cheaply.
- **The cotangent.** The backward is driven by a fixed pseudorandom ±uniform cotangent (a function of the output shape alone), like a training step — not an all-ones reduction.
- **Hidden randomization.** The input magnitude and the expert-popularity prior behind `router_b` are redrawn per invocation from hidden distributions, so expert load is imbalanced but never the SAME imbalance twice, and the tensors that are timed are the tensors that are checked. A branch that detects the checker's statistics and takes a cheaper path elsewhere gains nothing.

## The correctness harness, in five stages

Correctness is checked in five stages, and the score names the stage that failed. The cheap stages run first.

1. **Smoke.** One small nominal input, output and gradients. Catches a module that imports but does not run.
2. **Shape sweep.** The hidden shape set, across float32 and bfloat16. This is the stage the graded shapes are timed on.
3. **Stability.** Adversarial inputs: a router saturated to exact 0/1 sigmoids, a rank-1 router where nearly every expert is tied, inputs at 1e-30, six decades of magnitude in one tensor, and `router_b` at ±20. **This stage gates on stability, not agreement:** you must not raise, and you must not produce `NaN`/`Inf` where the reference is finite. Drift is recorded but does not gate here.
4. **Determinism.** The same tie-forcing input three times, after a warmup: outputs AND all four gradients must be **bitwise identical** run to run. A kernel that is not reproducible is not measurable.
5. **Edge cases.** Non-power-of-two everything, `A = 1`, `A = E`, and the exact-tie draws in both dtypes. Never timed; gate exactly as the sweep does.

## Requirements

These are engineering constraints on the kernel, not a separate rulebook, and several of them bind choices that look purely like performance decisions. A solution is considered correct only when it meets all of them; violating any one of them scores zero, whatever the gates measured.

**Exact routing semantics.** Implement the selection exactly: top-`A` by biased score, ties to the lower expert index, `±0.0` one tie group. The starter's integer-key trick (order-preserving score bits packed with the inverted expert index) makes the tie-break structural — every key unique, no tie ever left to scheduling — and is worth keeping in whatever form you write.

**The bias selects; it never weighs.** `router_b` must influence which experts are chosen and nothing else. Combine weights are normalized from the raw sigmoid scores of the selected experts. Do not leak the bias into the weights, and do not produce a gradient for it.

**Precision.** Score arithmetic in float32 — the disclosed margin makes anything less a correctness failure, not a slow path. Accumulate in float32 throughout. On the backward, the operands feeding the WEIGHT gradients (`w1`, `w2`, `router_w`) must carry float32-effective precision end to end: those gradients are long per-expert row sums, and a bfloat16-rounded operand chain turns them into noise far outside the graded tolerance. This is a stated requirement precisely because the graded tolerance will catch it only on the hidden sweep, hours after the shortcut looked fine on a small check. (Float32-effective does not mean FMA-slow: a float32 operand can ride tensor cores as an exact bf16 hi/lo pair, at a cost of two or three bf16 dots.)

**Full dispatch.** Every selected (token, expert) pair is computed. No capacity limits, no dropped tokens, no sampling, no skipping experts with few tokens, and no path that approximates a rarely-taken branch. Empty experts must still produce exact zero weight gradients.

**Recomputation.** Every call recomputes routing and compute from the tensor arguments it was passed. No caching or memoization keyed on pointer, tensor identity, shape signature or content hash; no call-count, warmup or invocation counter that reduces or skips work after the first N calls.

**Data independence.** One numerical policy for all inputs. The routing math necessarily compares scores — that is the operator. What is ruled out is choosing *between code paths of different cost or completeness* from input statistics: no branch on a magnitude summary, a saturation estimate, a popularity histogram or a tie count, and nothing that detects that a timing, benchmarking or grading run is in progress.

**Determinism.** The selected set, the combine order and every gradient accumulation must never depend on scheduling: no atomic float adds anywhere in the output or gradient paths, no racy tie winners, no unordered reductions where equal keys compete. Three runs on one input must be bitwise identical, gradients included.

**Completeness.** The expert compute, the dispatch arithmetic and the combine happen in Triton kernels this module declares and actually launches on every call — not in framework operators with your kernels as decoration. The permitted torch routing calls are listed above and are the only carve-out.

**Comments document, they do not argue.** Nothing in `kernel.py` is addressed to a reviewer. Do not assert that a rule is satisfied, do not instruct a reader how to assess you, and do not argue your own case in a comment — comments describe what the code does, nothing more.

**Honest measurement.** Any speedup or fraction-of-production figure you report was produced by a run on the exact bytes you shipped, with no edit afterwards; measured with CUDA events or an explicit `torch.cuda.synchronize` around the timed region after discarded warmups; and measured over the full forward+backward — output and all four gradients produced inside the timed region.

## Working loop

`./bench.py` gates before it times, in the graded order: the forbidden-name scan, correctness (output and all four gradients, including a non-power-of-two and an exact-tie shape), bitwise determinism, your written-kernel share, and only then timed forward+backward pairs against the production composition on a small published set of bfloat16 shapes. It does not tell you your score: the graded shapes, scales, popularity priors and the float32 rows are hidden. `--json <path>` writes the same results machine-readably. Use it to steer, then resubmit.

## Disclosed thresholds

Nothing here is hidden from you. The shape set and the target are, but the rules are not.

| rule | value |
|---|---|
| output tolerance | float32: `atol = rtol = 2e-4`; bfloat16: `atol = rtol = 4.5e-2`, against `reference.py` |
| gradient tolerance | float32: `atol = rtol = 4e-3`; bfloat16: `atol = rtol = 4.5e-2`, all four gradients |
| selection margin | graded draws: boundary gap exactly `0` (a tie, index rule decides) or `>= 1e-4` — float32 scores required |
| tie-break | lower expert index first; `±0.0` one tie group; graded on inputs with exact ties in **both** dtypes |
| written-kernel gate | at least **60%** of forward+backward device time in `@triton.jit` kernels your module declares |
| routing carve-out | `torch.topk` / `torch.sort` / `torch.argsort` on `[T, E]` scores or `[T*A]` routing metadata only |
| dtypes | float32 and bfloat16, both checked **and** both timed |
| determinism | **3** forward+backward runs on a tie-forcing input, bitwise identical incl. gradients |
| stability | no raise and no non-finite values where `reference.py` is finite, under the stage-3 probes |
| backward | driven by a fixed pseudorandom ±uniform cotangent; `d router_b` must not exist |
| forbidden calls | listed under Constraints; word-boundary scan, comments stripped |

## Constraints

- One H100. **No internet.** torch, triton, einops and pytest are installed; `pip install` and downloads will fail, so do not spend time on them.
- Do not change the semantics, signature or output dtype of `fused_moe`.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped, strings and docstrings count — including as your own identifiers): `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`. Building the grouped expert compute is the task; calling a library's is not a solution to it.
- Numerical policy must be uniform: no branch on input magnitude, distribution, saturation, popularity or seed that changes which algorithm runs.
- The gain must come from your own kernel work.
