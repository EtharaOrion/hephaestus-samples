# Task: implement a fast shared-expert routed top-K fused-MoE kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of a shared-expert routed top-K MoE (DeepSeek-V2/Qwen-MoE style): a dense always-on SwiGLU shared expert plus a softmax-routed top-K sparse SwiGLU expert pool. Both live INSIDE the operator. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `fused_moe(x, router_w, w1, w2, w1_s, w2_s, K)` with `x: [T, D]`, `router_w: [D, E]`, routed `w1: [E, D, 2F]`, `w2: [E, F, D]`, shared `w1_s: [D, 2Fs]`, `w2_s: [Fs, D]`, `K: int`. It returns `y: [T, D]` in the input dtype. Signature is fixed. **No router bias.** The graded surface is forward AND backward: **six** gradients are graded — dx, drouter_w, dw1, dw2, dw1_s, dw2_s.

## The operator

- **Routed path.**
  - `logits = x @ router_w`, `p = softmax(logits, dim=-1)`, in float32.
  - Top-K by `p`, ties **lower expert index first**; `±0.0` one tie group.
  - Combine weights `w[t] = p[t, sel[t]] / p[t, sel[t]].sum(-1)` (Mixtral rule).
  - `y_r[t] = sum_k w[t, k] * f_r(sel[t, k], x[t])` where `f_r(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]`.
- **Shared path (always-on, dense).**
  - `y_s[t] = (silu(x[t] @ w1_s[:, :Fs]) * (x[t] @ w1_s[:, Fs:])) @ w2_s`.
- **Composition.** `y[t] = y_r[t] + y_s[t]`.
- **Arithmetic.** float32 accumulation throughout; only final `y` cast back.
- **Gradients.** Six graded gradients. Shared-path gradients into x are added to routed-path gradients into x. `reference.py`'s autograd graph is the definition.

## How you are graded

One float in `[0, 1]`. **Zero** unless every gate passes:

- No banned symbols in `kernel.py` (word-boundary scan, comments stripped).
- Output and **all six gradients** agree with `reference.py` under the disclosed tolerances on a hidden shape set (E from 8-ish to 160-ish, non-power-of-two `T`, `D`, `F`, `Fs`, `E`, plus exact-tie draws).
- **At least 60%** of graded forward+backward device time in `@triton.jit` kernels your module declares. `torch.topk` / `torch.sort` / `torch.argsort` on `[T, E]` scores or `[T*K]` routing metadata are permitted.

Above the gates, score is geomean across hidden graded shapes (fp32 AND bf16) of `(production time)/(your time)`, normalized against a hidden target, saturating at 1. **Denominator:** router matmul + softmax + int-key top-K + p_sel/sum + `torch._grouped_mm` routed SwiGLU pipeline **plus** dense `torch.matmul` shared SwiGLU pipeline.

Disclosed:
- Decision margin `1e-4` on every K/(K+1) softmax gap.
- Backward driven by a fixed pseudorandom ±uniform cotangent per output shape.
- Hidden per-invocation magnitude randomization.

## Correctness harness (5 stages)

Smoke → shape sweep (fp32+bf16) → stability (saturated router, shared-only regime, tiny/huge x, near-uniform) → determinism (3 runs bitwise identical incl. six gradients) → edge cases (non-power-of-two + exact ties).

## Requirements

**Exact routing semantics.** Top-K by softmax weight, ties LOWER expert index.

**Exact composition.** `y = y_routed + y_shared`. Dropping the shared path, moving it into the router gate, or scaling it by anything other than 1 is a correctness failure.

**Dense shared path.** `y_s` uses every input token unconditionally through `w1_s` / `w2_s`. No sampling, no gating.

**p_sel/sum combine for routed.** Kept routed tokens are scaled by `p_sel / p_sel.sum(-1)`.

**Precision.** float32 score arithmetic. Weight-gradient operand chains (dw1, dw2, drouter_w, **dw1_s, dw2_s**) all float32-effective end to end. This is doubled work: routed grads are per-expert reductions, shared grads are full-T reductions.

**Dense router backward.** Softmax couples the whole expert row.

**Full dispatch.** Every routed (token, expert) selected pair is computed. Experts selected by no token still produce exact-zero routed weight gradients.

**Recomputation, data independence, determinism, completeness, no address-to-judge, honest measurement** — as in the family conventions.

## Working loop

`./bench.py` gates before it times.

## Disclosed thresholds

| rule | value |
|---|---|
| output tolerance | fp32: `atol=rtol=2e-4`; bf16: `4.5e-2` |
| gradient tolerance | fp32: `atol=rtol=4e-3`; bf16: `4.5e-2` (all SIX grads) |
| selection margin | K-th/(K+1)-th softmax gap exactly `0` or `>= 1e-4` |
| tie-break | lower expert index; `±0.0` one tie group |
| written-kernel gate | **60%** of fwd+bwd device time in `@triton.jit` kernels you declare |
| routing carve-out | `torch.topk`/`torch.sort`/`torch.argsort` on `[T, E]` scores or `[T*K]` metadata only |
| composition | `y = y_routed + y_shared`; no gating on shared |
| dtypes | fp32 and bf16, both checked and both timed |
| determinism | **3** fwd+bwd bitwise identical incl. all six gradients |
| forbidden calls | see Constraints |

## Constraints

- One H100, no internet.
- Do not change signature or output dtype.
- Names disallowed in `kernel.py` (word-boundary scan): `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`.
- One numerical policy for all inputs.
- Gain from your own kernel work — including reuse of operand loads between the shared and routed paths.
