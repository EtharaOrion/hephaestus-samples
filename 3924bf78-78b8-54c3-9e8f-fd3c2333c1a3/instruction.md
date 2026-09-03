# Task: implement a fast int4 W4A16 weight-only quantized SwiGLU MoE kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of mixture-of-experts expert compute with **int4 (W4A16) weight-only quantized** SwiGLU experts under per-output-channel **asymmetric** (scale + zero-point) dequant. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `int4_wq_swiglu_moe(x, w1_pack, w1_s, w1_zp, w2_pack, w2_s, w2_zp, topk_idx, topk_w)`; signature and semantics fixed. Autograd must work: harness grades **two** gradients, `dx` and `dtopk_w`. Packed weight buffers, scales and zero-points carry no gradient.

## The operator

Shapes: `x [T, D]`; `w1_pack [E, D, F]` uint8 (nibble-packed along last dim, unpack doubles it to `2F`); `w1_s [E, 2F]` per-out-channel scale; `w1_zp [E, 2F]` per-out-channel zero-point (int8, values `[0, 15]`); `w2_pack [E, F, D/2]` uint8; `w2_s [E, D]`; `w2_zp [E, D]`; `topk_idx [T, A]` int64; `topk_w [T, A]`. Output `y [T, D]` in `x.dtype`.

**Nibble convention.** Low nibble of each byte is the **even** output column (`2j`), high nibble is the **odd** column (`2j+1`). Both `2F` and `D` are even in every graded shape.

Per token `t` with `h = x[t]` and `e = topk_idx[t, a]`:

- `w1_int[e] = unpack_nibbles(w1_pack[e])` — `[D, 2F]`, values in `[0, 15]`
- `w1_dq[e] = (w1_int[e] - w1_zp[e][None, :]) * w1_s[e][None, :]`
- `w2_int[e] = unpack_nibbles(w2_pack[e])` — `[F, D]`
- `w2_dq[e] = (w2_int[e] - w2_zp[e][None, :]) * w2_s[e][None, :]`
- `f(e, h) = (silu(h @ w1_dq[e][:, :F]) * (h @ w1_dq[e][:, F:])) @ w2_dq[e]`
- `y[t] = sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])`

A fast kernel fuses UNPACK + SUBTRACT-zp + MULTIPLY-scale into the MMA prologue, keeps weights half the bytes of bf16, and feeds bf16 / fp16 tensor cores. A candidate that unpacks eagerly to bf16 and runs a standard GEMM is correct but slow.

`reference.py` computes in float32 on the dequantized values and is the definition of correctness.

## How you are graded

One float in `[0, 1]`; zero unless every gate passes:

- no forbidden library call (word-boundary scan);
- output, `dx` and `dtopk_w` agree with `reference.py` under the disclosed tolerances (output tolerance is widened for int4 quant noise, bounded below by `(range/15)` per channel);
- three runs bitwise identical (output + two grads);
- ≥ 60% of graded device time in your declared Triton kernels, fwd+bwd.

The denominator is a **cuBLAS grouped-GEMM composition with an eager per-expert unpack + asymmetric dequant pass** — that traffic (unpack the packed bytes, subtract zp, multiply scale, all as separate elementwise passes) is exactly what a fused int4 prologue elides. Widest structural headroom in the family alongside fp8.

## The correctness harness

1. **Smoke.** Small input, forward and backward.
2. **Shape sweep.** Hidden shape set, both dtypes, output + two grads. Timed here.
3. **Stability.** Adversarial inputs; finiteness gate only.
4. **Determinism.** Three runs, bitwise identical; A=8 probe.
5. **Edge cases.** Non-power-of-two dims (D always even), `A = 1`, collapsed routing.

## The rubric check

Same family rubric set. Every rubric must pass. Judge-unavailable does not fail.

## Requirements

**Asymmetric dequant respected.** Subtract per-out-channel zero-point BEFORE multiplying by scale. Symmetric-only dequant (`w_int * scale`) shifts every output column by `zp * scale` and drifts far outside `OUT_TOL`.

**Nibble order respected.** Low nibble = even column, high nibble = odd column. Swapping them transposes every expert weight's even/odd columns.

**Full routing, both graded gradients, determinism, recomputation, data independence, kernels-that-launch, no address-the-reader, honest measurement** — family baseline.

## Disclosed thresholds

| rule | value |
|---|---|
| operator | int4 W4A16 asymmetric-quantized SwiGLU MoE, per-out-channel scale + zero-point |
| nibble convention | low = even column, high = odd column, along the last packed dim |
| output tolerance | bfloat16: atol 0.40, rtol 0.15 · float32: atol 0.25, rtol 0.08 — widened for int4 quant noise |
| gradient tolerance | bfloat16: atol 6.0, rtol 0.10 · float32: atol 2.0, rtol 0.02 |
| graded gradients | `dx`, `dtopk_w` only |
| surface | forward + backward, timed and checked together; loss = sum over `y` |
| written-kernel gate | at least **60%** of graded device time in your `@triton.jit` kernels, fwd+bwd |
| determinism | 3 runs after warmup, bitwise identical output + both grads; A=8 probe |
| stability | no raise, no non-finite where reference is finite |
| dtypes | float32 and bfloat16, both checked and both timed |
| routing | precomputed; loads imbalanced under a hidden per-invocation-redrawn distribution |
| forbidden calls | listed under Constraints; word-boundary scan, comments stripped |
| rubric gate | every rubric must pass |

## Constraints

- One H100. **No internet.** torch, triton, einops, pytest installed.
- Do not change the semantics, signature or output dtype of `int4_wq_swiglu_moe`.
- Forbidden in `kernel.py`: `torch._grouped_mm`, `_grouped_mm`, `grouped_mm`, `fused_moe`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `cutlass`, `flashinfer`, **and int4-serving stacks**: `bitsandbytes`, `auto_gptq`, `awq`.
- Uniform numeric policy; no branching on input magnitude, load balance, or timing context.
- 64-bit index arithmetic where offsets can cross `2^31`.
- The gain must come from your own kernel work.
