# Task: implement a fast sigmoid top-K routed fused-MoE kernel with sum-renormalized combine, forward and backward

`kernel.py` contains a working but deliberately slow sigmoid-routed top-K MoE with a sum-renormalized combine (DeepSeek-V3-adjacent, distinct from the sibling `sigmoid_topk_glm` which ships a raw sigmoid combine plus a selection bias, and distinct from `softmax_topk_mixtral` which uses `p_sel/sum` on softmax scores).

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `fused_moe(x, router_w, w1, w2, K)` with `x: [T, D]`, `router_w: [D, E]` (same dtype as `x`), `w1: [E, D, 2F]`, `w2: [E, F, D]`, `K: int`. It returns `y: [T, D]` in the input dtype. Signature is fixed. **No router bias.** The graded surface is forward AND backward; dx, drouter_w, dw1, dw2 are graded alongside the output.

## The operator

- **Scores.** `logits = x @ router_w`, `s = sigmoid(logits)`, in float32.
- **Selection.** Top-`K` experts per token by `s`, ties broken by **lower expert index first**; `±0.0` one tie group.
- **Combine weights.** `w[t] = s[t, sel[t]] / s[t, sel[t]].sum(-1)` — sum-renormalized so `w.sum(-1) == 1` per token.
- **Experts.** `f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]`, `y[t] = sum_k w[t, k] * f(sel[t, k], x[t])`.
- **Arithmetic.** float32 accumulation; only final `y` cast back.
- **Gradients.** The sigmoid backward is elementwise (unselected experts get zero from the router-side path), but the sum-renorm COUPLES the K selected `s` gradients through both numerator and denominator. `reference.py`'s autograd graph is the definition.

## How you are graded

One float in `[0, 1]`. **Zero** unless every gate passes:

- No banned symbols in `kernel.py` (word-boundary scan, comments stripped).
- Output and all four gradients agree with `reference.py` under the disclosed tolerances on a hidden shape set.
- **At least 60%** of graded forward+backward device time in `@triton.jit` kernels your module declares. `torch.topk`/`torch.sort`/`torch.argsort` on `[T, E]` sigmoid scores or `[T*K]` routing metadata are permitted.

Above the gates, score is geomean across hidden graded shapes (fp32 AND bf16) of `(production time)/(your time)`, normalized against a hidden target, saturating at 1. **Denominator:** router matmul + sigmoid + integer-key top-K + gather + sum-renorm in torch, feeding a `torch._grouped_mm` expert pipeline.

Disclosed:
- Decision margin `1e-4` on every K/(K+1) sigmoid gap (or exactly 0, tie).
- Backward driven by a fixed pseudorandom ±uniform cotangent per output shape.
- Hidden per-invocation magnitude randomization.

## Correctness harness (5 stages)

Smoke → shape sweep (fp32+bf16, hidden set) → stability (finiteness under saturated router, sigmoid-zero regime, tiny/huge x, near-uniform) → determinism (3 runs bitwise identical incl. gradients) → edge cases (non-power-of-two + exact ties).

## Rubrics

An LLM from a different family reads your trajectory and final `kernel.py`. Every rubric must pass.

## Requirements

**Exact selection.** Top-K by sigmoid score, ties LOWER expert index; int64-key trick makes tie-break structural.

**Exact combine rule.** `w = s_sel / s_sel.sum(-1)`. NOT raw `s_sel`. NOT softmax. The magnitude and coupling structure of gradients depend on it.

**Precision.** float32 score arithmetic (the disclosed margin makes anything less a correctness failure). Weight-gradient operand chain (dw1, dw2, drouter_w) float32-effective end to end.

**Router backward through sigmoid + sum-renorm.** The K selected columns receive gradient through both the sigmoid derivative and the denominator coupling; unselected columns receive zero from this path.

**Full dispatch.** Every (token, expert) selected pair is computed. No sampling, no capacity, no skipping. Experts selected by no token still produce exact-zero weight gradients.

**Recomputation, data independence, determinism, completeness, no address-to-judge, honest measurement** — as in the family conventions.

## Working loop

`./bench.py` gates before it times.

## Disclosed thresholds

| rule | value |
|---|---|
| output tolerance | fp32: `atol=rtol=2e-4`; bf16: `4.5e-2` |
| gradient tolerance | fp32: `atol=rtol=4e-3`; bf16: `4.5e-2` |
| selection margin | K-th/(K+1)-th sigmoid gap exactly `0` or `>= 1e-4` |
| tie-break | lower expert index; `±0.0` one tie group |
| written-kernel gate | **60%** of fwd+bwd device time in `@triton.jit` kernels you declare |
| routing carve-out | `torch.topk`/`torch.sort`/`torch.argsort` on `[T, E]` scores or `[T*K]` metadata only |
| combine rule | `s_sel / s_sel.sum(-1)`, NOT raw sigmoid, NOT softmax |
| dtypes | fp32 and bf16, both checked and both timed |
| determinism | **3** fwd+bwd bitwise identical incl. gradients |
| forbidden calls | see Constraints |

## Constraints

- One H100, no internet.
- Do not change signature or output dtype.
- Names disallowed in `kernel.py` (word-boundary scan): `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`.
- One numerical policy for all inputs.
- Gain from your own kernel work.
