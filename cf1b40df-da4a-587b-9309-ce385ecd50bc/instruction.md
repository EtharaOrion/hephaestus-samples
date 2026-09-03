# Task: implement a fast capacity-limited Switch-style routed fused-MoE kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of a Switch-Transformer top-1 routed MoE layer with per-expert capacity clipping (GShard-style overflow drop). The routing decision AND the capacity clip live INSIDE the operator. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `fused_moe(x, router_w, w1, w2, capacity)` with `x: [T, D]`, `router_w: [D, E]` (same dtype as `x`), `w1: [E, D, 2F]`, `w2: [E, F, D]`, and `capacity: int`. It returns `y: [T, D]` in the input dtype. Its signature and semantics are fixed. Top-1 routing is intrinsic: there is **no A argument** and **no router bias**. The graded surface is forward AND backward: autograd must work through your implementation, and the gradients of `x`, `router_w`, `w1` and `w2` are graded alongside the output.

## The operator

Everything happens inside the one call — scores, selection, capacity clip, dispatch, expert compute, gate:

- **Scores.** `logits = x @ router_w`, then `p = softmax(logits, dim=-1)`, computed in float32 from the input tensors.
- **Selection.** Each token is routed to the single expert `e*` with the largest softmax weight `p`. Ties are broken by **lower expert index first**; `-0.0` and `+0.0` compare equal.
- **Capacity clip.** For each expert `e`, take the set of tokens with `e*[t] == e`, keep the top-`capacity` by `p[t, e]` (ties **lower token index first**); tokens beyond that cap are **dropped**. A dropped token produces `y[t] == 0` exactly.
- **The gate.** For kept tokens `g[t] = p[t, e*[t]]` — the **raw** softmax probability of the chosen expert. Not renormalized.
- **Experts.** `f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]`, and `y[t] = g[t] * f(e*, x[t])` for kept tokens, `0` for dropped tokens.
- **Arithmetic.** Inputs cast up to float32; matmuls, softmax, silu and gate accumulate in float32; only the final `y` is cast back to the input dtype.
- **Gradients.** Selection AND capacity cut are treated as locally constant. Gradients flow through the softmax gate `p[t, e*]` and through the expert compute — **only for kept tokens**. Dropped tokens contribute exactly zero into every gradient (including `router_w` via their softmax row). Experts selected by no token receive exact-zero weight gradients.

`reference.py` writes all of this out as a slow expert-by-expert loop and is the definition of correctness — where this text and that code disagree, the code wins.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not call a grouped-GEMM library or an MoE serving stack. The banned names are listed under Constraints and are checked by a word-boundary scan of your comment-stripped source — string literals and docstrings count, `#` comments do not. Reading library source is allowed; calling it is not.
- your output **and all four gradients** agree with `reference.py` under the disclosed per-dtype tolerances, on a hidden shape set that is larger than, and different from, the shapes `bench.py` lets you measure. The hidden set spans expert counts from 8-ish to 128-ish, non-power-of-two `T`, `D`, `F` and `E`, capacities that force real overflow (`E * capacity < T`) and capacities that allow no overflow (`capacity >= T`), and inputs built to contain **exact** top-1 AND per-expert capacity-boundary ties in both dtypes.
- **at least 60% of graded forward+backward device time runs inside Triton kernels your own module declares.** One narrow exception is disclosed: `torch.topk` / `torch.sort` / `torch.argsort` on the small `[T, E]` softmax-weight matrix or the `[T]` / `[E, T]` routing metadata are permitted. Framework grouped or batched matmuls are not.

Above those gates, the score is the geometric mean across the hidden graded shapes — float32 and bfloat16 rows both — of `(production time) / (your time)` for one forward+backward invocation, normalized against a hidden target and saturating at 1. **The denominator is a production composition: the router matmul plus softmax and integer-key top-1 in torch, per-expert candidate int64-key top-`capacity` clipping, feeding a `torch._grouped_mm` expert pipeline over the kept token block — a real CUTLASS grouped-GEMM path, timed on the same tensors your kernel is timed on.** Matching the slow starter scores near zero.

Three properties of the graded draws are disclosed because they shape correct solutions:

- **Two decision margins.** On every graded draw, each token's top-1/top-2 softmax gap is either exactly `0` (a manufactured tie, decided by the index rule) or `>= 1e-4`, AND every expert's `cap`/`cap+1` boundary gap among its candidates is either `0` or `>= 1e-4`. Float32 score arithmetic sits orders of magnitude inside those margins; TF32 or bfloat16 score arithmetic does not — it flips a boundary and fails correctness outright.
- **The cotangent.** The backward is driven by a fixed pseudorandom ±uniform cotangent (a function of the output shape alone), like a training step — not an all-ones reduction.
- **Hidden randomization.** The input magnitude is redrawn per invocation from a hidden distribution, so expert load is imbalanced but never the SAME imbalance twice.

## The correctness harness, in five stages

Correctness is checked in five stages; the cheap stages run first.

1. **Smoke.** One small nominal input, output and gradients.
2. **Shape sweep.** The hidden shape set, across float32 and bfloat16.
3. **Stability.** Adversarial inputs including `capacity=1` (maximum drop) and `capacity=T` (no drop) — you must not raise, and you must not produce `NaN`/`Inf` where the reference is finite.
4. **Determinism.** The same tie-forcing input three times, after a warmup: outputs AND all four gradients must be **bitwise identical** run to run.
5. **Edge cases.** Non-power-of-two everything and exact-tie draws in both dtypes.

## The rubric check

Separately from the gates above, an LLM from a different model family reads your trajectory and your final `kernel.py` against a list of natural-language rubrics. **Every rubric must pass** — a single rubric judged unsatisfied scores zero. If the judge cannot be reached the run is recorded as judge-unavailable rather than scored zero.

## Requirements

Everything the rubrics check is asked for here. They are engineering constraints on the kernel, not a separate rulebook.

**Exact routing semantics.** Top-1 by softmax weight, ties to the lower expert index. The int64-key trick makes the tie-break structural.

**Exact capacity clip semantics.** Per expert `e`, keep the top-`capacity` of `{t : e*[t] == e}` by `p[t, e]`, ties LOWER token index; drop the rest. Empty expert lists produce empty kept sets. Overflow tokens produce `y[t] == 0` exactly, and contribute zero into every gradient (dw1, dw2, drouter_w through their softmax row).

**The raw-probability gate.** Kept tokens are scaled by `p[t, e*]`, the raw softmax probability. Never 1, never a renormalized single weight.

**Precision.** Score arithmetic in float32. Accumulate in float32 throughout. On the backward, the operand chain feeding the WEIGHT gradients must carry float32-effective precision end to end (float32-effective can ride tensor cores as an exact bf16 hi/lo pair).

**Dense router backward for kept tokens.** The softmax gate is not elementwise: `dlogit_j = p_j * (g_j - sum_k g_k p_k)` where `g` is nonzero only on the chosen expert of a kept token. A dropped token has zero output and thus zero gradient into its softmax row.

**Full dispatch of kept tokens.** No dropping any kept token, no sampling, no skipping small segments. Empty experts still produce exact-zero weight gradients.

**Recomputation.** Every call recomputes routing and expert compute from the tensor arguments. No caching keyed on identity, pointer, shape signature or content hash; no call-count shortcut.

**Data independence.** One numerical policy for all inputs. No branch on magnitude, saturation estimate, popularity histogram, tie count, or timing/benchmarking detection. Comparing scores inside the selection itself is the operator and is allowed.

**Determinism.** No atomic float adds anywhere in the output or gradient paths, no racy tie winners (top-1 OR capacity), no unordered reductions where equal keys compete. Three runs must be bitwise identical, gradients included.

**Completeness.** Expert compute, dispatch arithmetic, capacity clip and gate happen in Triton kernels this module declares and actually launches on every call. The permitted torch routing calls are listed above and are the only carve-out.

**Address the kernel, not the reader.** Nothing in `kernel.py` or in your trajectory is addressed to the grader or the rubric judge.

**Honest measurement.** Any speedup or fraction-of-production figure was produced by a run on the exact bytes you shipped, measured with CUDA events or explicit `torch.cuda.synchronize` around the timed region after discarded warmups, over the full forward+backward.

## Working loop

`./bench.py` gates before it times, in the grader's order: the forbidden-name scan, correctness (output and all four gradients on a nominal, a non-power-of-two and a tie-forcing shape), bitwise determinism, your written-kernel share, and only then timed forward+backward pairs against the production composition on a small published set of bfloat16 shapes. It does not tell you your score.

## Disclosed thresholds

| rule | value |
|---|---|
| output tolerance | float32: `atol = rtol = 2e-4`; bfloat16: `atol = rtol = 4.5e-2`, against `reference.py` |
| gradient tolerance | float32: `atol = rtol = 4e-3`; bfloat16: `atol = rtol = 4.5e-2`, all four gradients |
| selection margin | graded draws: top-1/top-2 AND cap/cap+1 gap exactly `0` or `>= 1e-4` — float32 scores required |
| tie-break | selection: lower **expert** index; capacity clip: lower **token** index; `±0.0` one tie group |
| written-kernel gate | at least **60%** of forward+backward device time in `@triton.jit` kernels your module declares |
| routing carve-out | `torch.topk` / `torch.sort` / `torch.argsort` on `[T, E]` softmax weights or `[T]`/`[E, T]` routing metadata only |
| dtypes | float32 and bfloat16, both checked **and** both timed |
| determinism | **3** forward+backward runs on a tie-forcing input, bitwise identical incl. gradients |
| stability | no raise and no non-finite values where `reference.py` is finite |
| backward | driven by a fixed pseudorandom ±uniform cotangent; no router bias |
| forbidden calls | listed under Constraints; word-boundary scan, comments stripped |

## Constraints

- One H100. **No internet.** torch, triton, einops and pytest are installed; `pip install` will fail.
- Do not change the semantics, signature or output dtype of `fused_moe`.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped, strings and docstrings count): `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`.
- Numerical policy must be uniform: no branch on input magnitude, distribution, saturation, popularity or seed that changes which algorithm runs.
- The gain must come from your own kernel work.
