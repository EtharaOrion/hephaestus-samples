# Task: implement a fast fine-grained softmax top-K routed fused-MoE kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of a fine-grained softmax top-K MoE layer with a **re-softmax over the selected logits** combine rule (DeepSeek-V3-adjacent, distinct from the Mixtral `p_sel/sum` rule). The routing decision AND the combine weights live INSIDE the operator. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `fused_moe(x, router_w, w1, w2, K)` with `x: [T, D]`, `router_w: [D, E]` (same dtype as `x`), `w1: [E, D, 2F]`, `w2: [E, F, D]`, `K: int`. It returns `y: [T, D]` in the input dtype. Signature is fixed. There is **no router bias**. The graded surface is forward AND backward: autograd must work, and dx, drouter_w, dw1, dw2 are graded alongside the output.

## The operator

- **Scores.** `logits = x @ router_w`, `p = softmax(logits, dim=-1)`, in float32.
- **Selection.** Top-`K` experts per token by `p`, ties broken by **lower expert index first**; `±0.0` one tie group.
- **Combine weights.** `w[t] = softmax(logits[t, sel[t]])` — a **re-softmax over the K selected LOGITS**. This is distinct from `p_sel / p_sel.sum(-1)` (the Mixtral rule) at fp32: order-preserving but numerically different, and every graded gradient depends on which rule the candidate uses.
- **Experts.** `f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]`, `y[t] = sum_k w[t, k] * f(sel[t, k], x[t])`.
- **Arithmetic.** Inputs cast to float32; matmuls, both softmaxes, silu and combine accumulate in float32; only the final `y` is cast back.
- **Gradients.** Selection indices are locally constant. Gradients flow through the scoring softmax (dense over the whole expert row, into `router_w` and `x`), through the re-softmax over the K selected logits (dense over K), and through the expert compute. `reference.py`'s autograd graph is the definition.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not call a grouped-GEMM library or MoE serving stack. The banned names are listed under Constraints and are checked by a word-boundary scan of your comment-stripped source.
- your output **and all four gradients** agree with `reference.py` under the disclosed per-dtype tolerances, on a hidden shape set including expert counts from 32-ish to 256-ish, non-power-of-two `T`, `D`, `F` and `E`, and inputs built to contain exact top-K boundary ties in both dtypes.
- **at least 60% of graded forward+backward device time runs inside Triton kernels your own module declares.** One narrow exception is disclosed: `torch.topk` / `torch.sort` / `torch.argsort` on the small `[T, E]` softmax weights or `[T*K]` routing metadata are permitted. Framework grouped or batched matmuls are not.

Above those gates, the score is the geometric mean across the hidden graded shapes — float32 and bfloat16 rows both — of `(production time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is a production composition: router matmul plus softmax and integer-key top-K in torch, torch.gather + torch.softmax over the K selected logits for combine weights, feeding a `torch._grouped_mm` expert pipeline over tokens sorted by expert.**

Three disclosed properties of the graded draws:

- **Decision margin.** Every token's `K`-th/`K+1`-th softmax gap is either exactly `0` (a manufactured tie, decided by the index rule) or `>= 1e-4`. Float32 score arithmetic sits inside that margin; TF32 or bfloat16 does not — it flips the selection.
- **Cotangent.** Fixed pseudorandom ±uniform cotangent per output shape, like a training step.
- **Hidden randomization.** Input magnitude redrawn from a hidden distribution per invocation.

## Correctness harness (5 stages)

1. **Smoke.** Small nominal input.
2. **Shape sweep.** Hidden shape set across float32 and bfloat16.
3. **Stability.** Adversarial inputs (saturated router at E=256, tiny/huge x, near-uniform); finiteness gate.
4. **Determinism.** Three fwd+bwd runs on a tie-forcing input, bitwise identical outputs AND gradients.
5. **Edge cases.** Non-power-of-two everything and exact-tie draws in both dtypes.

## Rubrics

An LLM from a different family reads your trajectory and final `kernel.py` against a list of natural-language rubrics. **Every rubric must pass** — one unsatisfied scores zero. Judge-unreachable is recorded as judge-unavailable, not scored zero.

## Requirements

**Exact routing semantics.** Top-K by softmax weight, ties LOWER expert index. The int64-key trick makes the tie-break structural.

**Exact combine semantics.** `w = softmax(logits[sel])` — re-softmax over the K selected LOGITS. NOT `p_sel / p_sel.sum()`. NOT `softmax(p_sel)`. The difference is small in magnitude and lethal in gradients.

**Precision.** Score arithmetic in float32. The disclosed margin makes anything less a correctness failure. Both softmaxes and the combine accumulate in float32. On the backward, the operand chain feeding the WEIGHT gradients (dw1, dw2, drouter_w) must carry float32-effective precision end to end (a float32 operand can ride tensor cores as an exact bf16 hi/lo pair).

**Dense router backward.** The scoring softmax couples the whole row: `dlogit_j = p_j * (g_j - sum_k g_k p_k)` where `g` is nonzero on the K selected experts. Every logit in a routed token's row receives a gradient. The combine softmax adds a second dense coupling over the K selected logits.

**Full dispatch.** Every selected (token, expert) pair is computed. No sampling, no skipping small segments, no capacity limits. Experts selected by no token still produce exact-zero weight gradients.

**Recomputation.** Every call recomputes routing and expert compute. No caching keyed on identity, pointer, shape signature or content hash; no call-count shortcut.

**Data independence.** One numerical policy for all inputs. No branch on magnitude, saturation, popularity or tie count. Comparing scores inside the selection itself is the operator and is allowed.

**Determinism.** No atomic float adds anywhere. No racy tie winners. Three runs bitwise identical, gradients included.

**Completeness.** Expert compute, dispatch arithmetic, both softmaxes and combine happen in Triton kernels this module declares. Permitted torch routing calls are listed above.

**Address the kernel, not the reader.** Nothing in `kernel.py` or your trajectory is addressed to the grader or the rubric judge.

**Honest measurement.** Any speedup or fraction-of-production figure was produced by a run on the exact bytes you shipped, measured with CUDA events or explicit `torch.cuda.synchronize` around the timed region after discarded warmups, over the full forward+backward.

## Working loop

`./bench.py` gates before it times: forbidden-name scan, correctness (output and 4 grads on a nominal, non-power-of-two and tie-forcing shape), determinism, adoption, then timed pairs against production. Not your score.

## Disclosed thresholds

| rule | value |
|---|---|
| output tolerance | float32: `atol = rtol = 2e-4`; bfloat16: `atol = rtol = 4.5e-2` |
| gradient tolerance | float32: `atol = rtol = 4e-3`; bfloat16: `atol = rtol = 4.5e-2` |
| selection margin | graded draws: K-th/(K+1)-th p gap exactly `0` or `>= 1e-4` — float32 scores required |
| tie-break | lower expert index first; `±0.0` one tie group |
| written-kernel gate | **60%** of forward+backward device time in `@triton.jit` kernels your module declares |
| routing carve-out | `torch.topk` / `torch.sort` / `torch.argsort` on `[T, E]` scores or `[T*K]` metadata only |
| dtypes | float32 and bfloat16, both checked and both timed |
| determinism | **3** fwd+bwd runs on a tie-forcing input, bitwise identical incl. gradients |
| combine rule | `w = softmax(logits[sel])`; **not** `p_sel / p_sel.sum()` |
| forbidden calls | listed under Constraints |

## Constraints

- One H100. **No internet.** torch, triton, einops, pytest installed.
- Do not change the semantics, signature or output dtype of `fused_moe`.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped, strings and docstrings count): `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`.
- One numerical policy for all inputs.
- Gain must come from your own kernel work.
