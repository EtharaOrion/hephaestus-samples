# Task: implement a fast fp8-e4m3 weight-only quantized SwiGLU MoE kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of mixture-of-experts expert compute with **fp8-e4m3 weight-only quantized** SwiGLU experts and precomputed top-k routing. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `fp8_e4m3_swiglu_moe(x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w)`; signature and semantics are fixed. Autograd must work through it: the harness grades **two** gradients — `dx` and `dtopk_w`. The quantized weight buffers `w1_q`, `w1_s`, `w2_q`, `w2_s` are inference-shaped inputs and carry no gradient. `topk_idx` has no gradient.

## The operator

Shapes: `x [T, D]`; `w1_q [E, D, 2F]` (weights on the fp8-e4m3 grid, stored in `x.dtype`); `w1_s [E, 2F]` per-output-channel scale; `w2_q [E, F, D]`; `w2_s [E, D]`; `topk_idx [T, A]` int64; `topk_w [T, A]`. Output `y [T, D]` in `x.dtype`.

Per token `t` with `h = x[t]` and `e = topk_idx[t, a]`:

- `w1_dq[e] = w1_q[e] * w1_s[e][None, :]` (elementwise per output channel);
- `w2_dq[e] = w2_q[e] * w2_s[e][None, :]`;
- `f(e, h) = (silu(h @ w1_dq[e][:, :F]) * (h @ w1_dq[e][:, F:])) @ w2_dq[e]`;
- `y[t] = sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])`.

The weight values ARE ALREADY snapped to the fp8-e4m3fn grid; the scale is symmetric per-output-channel (amax / 448). A fast kernel keeps the weights in fp8 and consumes them from Hopper's fp8-e4m3 tensor cores with the scale applied in the epilogue. A candidate that dequantizes eagerly to bf16 and runs a standard bf16 GEMM is correct but slow.

`reference.py` computes in float32 (over dequantized values) and is the definition of correctness.

## How you are graded

One float in `[0, 1]`; zero unless every gate passes:

- no forbidden library call (word-boundary scan);
- output, `dx` and `dtopk_w` agree with `reference.py` under the disclosed tolerances (output tolerance is widened to accommodate fp8 quant noise);
- three runs bitwise identical (output, `dx`, `dtopk_w`);
- ≥ 60% of graded device time in your declared Triton kernels, fwd+bwd.

The denominator is a **cuBLAS grouped-GEMM composition with an eager per-expert bf16 dequant pass** — that dequant traffic is the exact work a fp8-prologue-fusion kernel elides. This gives the family's widest headroom.

## The correctness harness

1. **Smoke.** Small input.
2. **Shape sweep.** Hidden shape set, both dtypes, output + two grads. Timed here.
3. **Stability.** Adversarial inputs; finiteness gate only.
4. **Determinism.** Three runs, bitwise identical output and both grads; A=8 probe.
5. **Edge cases.** Non-power-of-two dims, `A = 1`, collapsed routing.

## The rubric check

Same family rubric set. Every rubric must pass. Judge-unavailable does not fail.

## Requirements

**Weight-only quantization respected.** Consume `w1_q`/`w2_q` on the fp8 grid; scale is applied per output channel. Do not re-quantize x (activations stay in `x.dtype`).

**Full routing, both graded gradients (`dx`, `dtopk_w`), determinism, recomputation, data independence, kernels-that-launch, no address-the-reader, honest measurement** — family baseline.

## Disclosed thresholds

| rule | value |
|---|---|
| operator | fp8-e4m3 weight-only quantized SwiGLU MoE, per-out-channel symmetric scale |
| output tolerance | bfloat16: atol 0.35, rtol 0.15 · float32: atol 0.20, rtol 0.08 — widened for fp8 quant noise |
| gradient tolerance | bfloat16: atol 6.0, rtol 0.10 · float32: atol 2.0, rtol 0.02 |
| graded gradients | `dx`, `dtopk_w` only — quantized weight buffers carry no gradient |
| surface | forward + backward, timed and checked together; loss = sum over `y` |
| written-kernel gate | at least **60%** of graded device time in your `@triton.jit` kernels, fwd+bwd |
| determinism | 3 runs after warmup, bitwise identical output + both grads; A=8 probe |
| stability | no raise, no non-finite where reference is finite |
| dtypes | float32 and bfloat16, both checked and both timed |
| routing | precomputed; loads imbalanced under a hidden per-invocation-redrawn distribution |
| forbidden calls | listed under Constraints |
| rubric gate | every rubric must pass |

## Constraints

- One H100. **No internet.** torch, triton, einops, pytest installed.
- Do not change the semantics, signature or output dtype of `fp8_e4m3_swiglu_moe`.
- Forbidden in `kernel.py`: `torch._grouped_mm`, `_grouped_mm`, `grouped_mm`, `fused_moe`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `cutlass`, `flashinfer`.
- Uniform numeric policy; no branching on input magnitude, load balance, or timing context.
- Use 64-bit index arithmetic where offsets can cross `2^31`.
- The gain must come from your own kernel work.
