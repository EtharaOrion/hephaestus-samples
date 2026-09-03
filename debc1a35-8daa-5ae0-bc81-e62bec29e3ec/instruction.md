# Task: implement a fast fused MoE + SwiGLU + per-expert output bias kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of mixture-of-experts expert compute with a SwiGLU FFN, a **per-expert output bias `b2[E, D]` fused into the down-projection**, and precomputed top-k routing. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `fused_downproj_swiglu_moe(x, w1, w2, b2, topk_idx, topk_w)` and its signature and semantics are fixed. **Autograd must work through it**: the harness runs forward plus backward and grades **five** gradients — `dx`, `dw1`, `dw2`, `db2` (the per-expert output bias) and `dtopk_w`. `topk_idx` is integer routing and has no gradient.

## The operator

Shapes: `x [T, D]`, `w1 [E, D, 2F]`, `w2 [E, F, D]`, `b2 [E, D]`, `topk_idx [T, A]` int64, `topk_w [T, A]`. Output `y [T, D]`.

Per token `t`, with `h = x[t]` and `e = topk_idx[t, a]`:

- `f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]  +  b2[e]`;
- `y[t] = sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])`.

The bias is added **inside** the routing sum, BEFORE the `topk_w` scalar multiply. Duplicate experts per token are legal and each slot contributes independently — the same `b2[e]` is added once per routed slot that lands on expert `e`. Experts that no token routes to receive exactly zero `dw1`, `dw2` and `db2`.

`reference.py` computes in float32 and is the definition of correctness.

## How you are graded

One float in `[0, 1]`, zero unless every gate passes:

- no forbidden library call (word-boundary scan, comments stripped);
- output and all five gradients agree with `reference.py` under the disclosed tolerances on a hidden shape set;
- three runs on one input produce bitwise identical output and all five gradients;
- ≥ 60% of graded device time in your declared Triton kernels, forward+backward.

Above the gates, the score is a geomean of `(production time) / (your time)`, normalized against a hidden target. The **denominator is a cuBLAS grouped-GEMM composition with an eager `+b2` bias add on the segment**: the anchor's cuBLAS grouped_mm has no epilogue for per-group additive bias + per-token scalar, so the bias-add and the routing-weight multiply are two extra elementwise passes. That is the headroom a fused down-projection kernel captures.

## The correctness harness

1. **Smoke.** Small input, forward and backward.
2. **Shape sweep.** Hidden shape set, both dtypes, output and all five gradients per cell. Timed here.
3. **Stability.** Adversarial inputs (collapsed routing, tiny/huge inputs, six-decade weights, zero routing weights → y is zero here since there is no shared always-on path). Gates on finiteness only.
4. **Determinism.** Three runs, bitwise identical output and all five gradients; A=8 probe.
5. **Edge cases.** Non-power-of-two dims, `A = 1`, collapsed routing.

## The rubric check

Same rubric set as the family; every rubric must pass, judge-unavailable does not fail.

## Requirements

**Bias correctness.** `b2[e]` is added to every routed slot's down-projection output before its `topk_w` multiply; `db2[e]` sums that downstream signal over every slot routed to expert `e`, weighted by that slot's `topk_w`.

**Full routing, all five gradients, determinism, recomputation, data independence, kernels-that-launch, no address-the-reader, honest measurement** — see the family rubrics.

## Disclosed thresholds

| rule | value |
|---|---|
| operator | SwiGLU + per-expert output bias `b2` fused into down-proj |
| output tolerance | bfloat16: atol 0.16, rtol 0.10 · float32: atol 0.05, rtol 0.01 |
| gradient tolerance | bfloat16: atol 18.0, rtol 0.10 · float32: atol 5.0, rtol 0.02 — sized for `db2` which sums the downstream signal over routed slots |
| graded gradients | `dx`, `dw1`, `dw2`, `db2`, `dtopk_w` — all five, every graded cell |
| surface | forward + backward, timed and checked together; loss = sum over `y` |
| written-kernel gate | at least **60%** of graded device time in your `@triton.jit` kernels, fwd+bwd |
| determinism | 3 runs after warmup, bitwise identical output and all five gradients; A=8 probe |
| stability | no raise, no non-finite where reference is finite |
| dtypes | float32 and bfloat16, both checked and both timed |
| routing | precomputed; loads imbalanced under a hidden per-invocation-redrawn distribution |
| forbidden calls | listed under Constraints; word-boundary scan, comments stripped |
| rubric gate | every rubric must pass |

## Constraints

- One H100. **No internet.** torch, triton, einops and pytest are installed.
- Do not change the semantics, signature or output dtype of `fused_downproj_swiglu_moe`.
- Forbidden in `kernel.py`: `torch._grouped_mm`, `_grouped_mm`, `grouped_mm`, `fused_moe`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `cutlass`, `flashinfer`.
- Uniform numeric policy; no branching on input magnitude, load balance, or timing context.
- Weights are large; use 64-bit index arithmetic where offsets could cross `2^31`.
- The gain must come from your own kernel work.
