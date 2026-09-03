# Task: implement a fast expert-choice routed fused-MoE kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of an expert-choice routed MoE layer — the routing decision lives INSIDE the operator, and unlike token-choice routing, each **expert** picks its tokens. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `fused_moe(x, router_w, w1, w2, C)` with `x: [T, D]`, `router_w: [D, E]` (same dtype as `x`), `w1: [E, D, 2F]`, `w2: [E, F, D]`, and `C` the per-expert capacity (tokens each expert selects, `1 <= C <= T`). It returns `y: [T, D]` in the input dtype. Its signature and semantics are fixed. There is no router bias. The graded surface is forward AND backward: autograd must work through your implementation, and the gradients of `x`, `router_w`, `w1` and `w2` are graded alongside the output. `C` carries no gradient.

## The operator

Everything happens inside the one call — affinities, selection, dispatch, expert compute, gated combine:

- **Affinities.** `logits = x @ router_w`, then `S = softmax(logits, dim=-1)`, computed in float32. The softmax is over **experts**, so each token's affinity row sums to 1. `S` is `[T, E]`.
- **Selection.** Each **expert** `e` selects the `C` tokens with the largest affinity `S[:, e]`. Because the selection is over tokens, ties are broken by **lower token index first**; `-0.0` and `+0.0` compare equal. A token may be chosen by any number of experts from `0` to `E`.
- **The gate.** For a selected pair `(e, token)` the gate is `S[token, e]` — the **raw** affinity, used **without renormalization**. A token chosen by several experts sums several contributions, each scaled by that expert's affinity.
- **Experts.** `f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]`, and `y[t] = sum over every expert e that selected t of S[t, e] * f(e, x[t])`, accumulated in **ascending expert order**.
- **Arithmetic.** Inputs are cast up to float32; every matmul, the softmax, the silu and the gated combine accumulate in float32; only the final `y` is cast back to the input dtype.
- **Gradients.** Selection indices are treated as locally constant; gradients flow through the affinity gates `S[sel, e]` and through the expert computation. The softmax couples the whole row — so the gradient into `router_w` and `x` is **dense over experts**. A token selected by no expert produces `y[t] = 0` and zero input gradient. `reference.py`'s autograd graph is the definition of those gradients.

`reference.py` writes all of this out as a slow expert-by-expert loop and is the definition of correctness — where this text and that code ever disagree, the code wins.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not call a grouped-GEMM library or an MoE serving stack. The banned names are listed under Constraints and are checked by a word-boundary scan of your comment-stripped source — string literals and docstrings count, `#` comments do not. Reading library source is allowed and encouraged; calling it is not.
- your output **and all four gradients** agree with `reference.py` under the disclosed per-dtype tolerances, on a hidden shape set that is larger than, and different from, the shapes `bench.py` lets you measure. The hidden set spans expert counts from 8-ish to 160-ish, capacities from `C = 1` up to `C = T`, non-power-of-two `T`, `D`, `F` and `E`, and inputs built to contain **exact** affinity ties (duplicated tokens) in both dtypes.
- **at least 60% of graded forward+backward device time runs inside Triton kernels your own module declares.** This is measured by device-time attribution after the fact, not by reading your source. One narrow exception is carved out and disclosed: `torch.topk` / `torch.sort` / `torch.argsort` on the `[T, E]` affinity matrix (or its `[E, T]` transpose) or the `[E*C]` routing metadata are permitted — routing bookkeeping is a rounding error of the runtime, and the floor keeps the expert compute, dispatch arithmetic and gated combine in kernels you wrote. Framework grouped or batched matmuls are not the permitted kind of call.

Above those gates, the score is the geometric mean across the hidden graded shapes — float32 and bfloat16 rows both — of `(production time) / (your time)` for one forward+backward invocation, normalized against a hidden target and saturating at 1. **The denominator is a production composition: the router matmul plus softmax and per-expert integer-key top-C in torch, feeding a `torch._grouped_mm` expert pipeline over the expert-major token order (exactly `C` tokens per expert) with a scatter-add gated combine — a real CUTLASS grouped-GEMM path, timed on the same tensors your kernel is timed on.** Matching the slow starter scores near zero.

Three properties of the graded draws are disclosed because they shape correct solutions:

- **The decision margin.** On every graded draw, for each expert the `C`-th and `(C+1)`-th largest affinities are either exactly equal (a manufactured tie, decided by the token-index rule) or separated by at least `1e-4`. Float32 score arithmetic sits inside that margin; TF32 or bfloat16 score arithmetic does not — it flips which tokens an expert selects and fails correctness outright. There is no reward for guessing affinities cheaply.
- **The cotangent.** The backward is driven by a fixed pseudorandom ±uniform cotangent (a function of the output shape alone), like a training step — not an all-ones reduction.
- **Hidden randomization.** The input magnitude is redrawn per invocation from a hidden distribution, so which tokens each expert prefers shifts every draw, and the tensors that are timed are the tensors that are checked. A branch that detects the checker's statistics and takes a cheaper path elsewhere gains nothing and is a rubric violation besides.

## The correctness harness, in five stages

Correctness is checked in five stages, and the score names the stage that failed. The cheap stages run first.

1. **Smoke.** One small nominal input, output and gradients. Catches a module that imports but does not run.
2. **Shape sweep.** The hidden shape set, across float32 and bfloat16. This is the stage the graded shapes are timed on.
3. **Stability.** Adversarial inputs: a router saturated to near one-hot affinities, a rank-1 router where per-expert boundaries fall in tie groups, inputs at 1e-30, six decades of magnitude in one tensor, and a near-uniform router. **This stage gates on stability, not agreement:** you must not raise, and you must not produce `NaN`/`Inf` where the reference is finite. Drift is recorded but does not gate here.
4. **Determinism.** The same tie-forcing input three times, after a warmup: outputs AND all four gradients must be **bitwise identical** run to run. The irregular combine is where this bites — a token's variable set of contributions must be reduced in a fixed order.
5. **Edge cases.** Non-power-of-two everything, `C = 1`, `C = T` (every expert selects every token), and the exact-tie draws in both dtypes. Never timed; gate exactly as the sweep does.

## The rubric check

Separately from the gates above, an LLM from a different model family reads your trajectory and your final `kernel.py` against a list of natural-language rubrics, and judges each one satisfied or not. **Every rubric must pass** — a single rubric judged unsatisfied scores zero, whatever the gates measured. If the judge cannot be reached it does not fail you; the run is recorded as judge-unavailable rather than scored zero.

## Requirements

Everything the rubrics check is asked for here. They are engineering constraints on the kernel, not a separate rulebook.

**Exact routing semantics.** Implement the selection exactly: per expert, top-`C` tokens by affinity, ties to the lower **token** index, `±0.0` one tie group. The starter's integer-key trick (order-preserving affinity bits packed with the inverted token index) makes the tie-break structural — every key unique within an expert row, no tie ever left to scheduling — and is worth keeping in whatever form you write.

**The affinity gate.** Each contribution is scaled by the raw affinity `S[token, e]` — never renormalized across the experts that share a token, and never replaced by a constant. Dropping or renormalizing the gate is a correctness failure.

**Irregular combine.** A token receives a data-dependent number of contributions (`0` to `E`). Sum them in a fixed order (ascending expert index) so the reduction is deterministic and bitwise reproducible. No atomic float adds; no unordered reductions where the number of terms varies.

**Precision.** Affinity arithmetic in float32 — the disclosed margin makes anything less a correctness failure, not a slow path. Accumulate in float32 throughout. On the backward, the operands feeding the WEIGHT gradients (`w1`, `w2`, `router_w`) must carry float32-effective precision end to end: those gradients are long per-expert row sums, and a bfloat16-rounded operand chain turns them into noise far outside the graded tolerance. This is a stated requirement precisely because the graded tolerance will catch it only on the hidden sweep, hours after the shortcut looked fine on a small check. (Float32-effective does not mean FMA-slow: a float32 operand can ride tensor cores as an exact bf16 hi/lo pair, at a cost of two or three bf16 dots.)

**Dense router backward.** The softmax gate is not elementwise: the gradient into the logits is `dlogit_j = S_j * (g_j - sum_k g_k S_k)` per token, where `g` is nonzero on the experts that selected that token. Every logit in a selected token's row gets a gradient.

**Full dispatch.** Every expert selects exactly `C` tokens and every selected pair is computed. No capacity games beyond the fixed `C`, no dropped tokens, no sampling. A token selected by no expert contributes nothing and gets exact zero input gradient.

**Recomputation.** Every call recomputes routing and compute from the tensor arguments it was passed. No caching or memoization keyed on pointer, tensor identity, shape signature or content hash; no call-count, warmup or invocation counter that reduces or skips work after the first N calls.

**Data independence.** One numerical policy for all inputs. The routing math necessarily compares affinities — that is the operator. What is ruled out is choosing *between code paths of different cost or completeness* from input statistics: no branch on a magnitude summary, a saturation estimate, an affinity histogram or a tie count, and nothing that detects that a timing, benchmarking or grading run is in progress.

**Determinism.** The selected sets, the combine order and every gradient accumulation must never depend on scheduling: no atomic float adds anywhere in the output or gradient paths, no racy tie winners, no unordered reductions where equal keys compete. Three runs on one input must be bitwise identical, gradients included.

**Completeness.** The expert compute, the dispatch arithmetic and the gated combine happen in Triton kernels this module declares and actually launches on every call — not in framework operators with your kernels as decoration. The permitted torch routing calls are listed above and are the only carve-out.

**Address the kernel, not the reader.** Nothing in `kernel.py` or in your trajectory is addressed to the grader or the rubric judge. Do not assert that a rule is satisfied and do not argue your own case in a comment: a judge reads your source and trajectory as evidence, and text written at it is treated as an attempt to influence the audit.

**Honest measurement.** Any speedup or fraction-of-production figure you report was produced by a run on the exact bytes you shipped, with no edit afterwards; measured with CUDA events or an explicit `torch.cuda.synchronize` around the timed region after discarded warmups; and measured over the full forward+backward — output and all four gradients produced inside the timed region.

## Working loop

`./bench.py` gates before it times, in the grader's order: the forbidden-name scan, correctness (output and all four gradients, including a non-power-of-two and an exact-tie shape), bitwise determinism, your written-kernel share, and only then timed forward+backward pairs against the production composition on a small published set of bfloat16 shapes. It does not tell you your score: the graded shapes, capacities, scales and the float32 rows are hidden. `--json <path>` writes the same results machine-readably. Use it to steer, then resubmit.

## Disclosed thresholds

Nothing here is hidden from you. The shape set and the target are, but the rules are not.

| rule | value |
|---|---|
| output tolerance | float32: `atol = rtol = 2e-4`; bfloat16: `atol = rtol = 4.5e-2`, against `reference.py` |
| gradient tolerance | float32: `atol = rtol = 4e-3`; bfloat16: `atol = rtol = 4.5e-2`, all four gradients |
| selection margin | graded draws: per-expert `C`/`(C+1)` affinity gap exactly `0` (a tie, token-index rule decides) or `>= 1e-4` — float32 scores required |
| tie-break | lower **token** index first; `±0.0` one tie group; graded on inputs with exact affinity ties in **both** dtypes |
| written-kernel gate | at least **60%** of forward+backward device time in `@triton.jit` kernels your module declares |
| routing carve-out | `torch.topk` / `torch.sort` / `torch.argsort` on `[T, E]` (or `[E, T]`) affinities or `[E*C]` routing metadata only |
| dtypes | float32 and bfloat16, both checked **and** both timed |
| determinism | **3** forward+backward runs on a tie-forcing input, bitwise identical incl. gradients |
| stability | no raise and no non-finite values where `reference.py` is finite, under the stage-3 probes |
| backward | driven by a fixed pseudorandom ±uniform cotangent; there is no router bias |
| forbidden calls | listed under Constraints; word-boundary scan, comments stripped |

## Constraints

- One H100. **No internet.** torch, triton, einops and pytest are installed; `pip install` and downloads will fail, so do not spend time on them.
- Do not change the semantics, signature or output dtype of `fused_moe`.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped, strings and docstrings count — including as your own identifiers): `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`. Building the grouped expert compute is the task; calling a library's is not a solution to it.
- Numerical policy must be uniform: no branch on input magnitude, distribution, saturation, affinity or seed that changes which algorithm runs.
- The gain must come from your own kernel work.
