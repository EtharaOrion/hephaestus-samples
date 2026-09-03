# Task: implement a fast fused MoE + SwiGLU kernel for WIDE FFN, HIGH-E shapes, fwd/bwd

`kernel.py` contains a working but deliberately slow SwiGLU MoE implementation. The graded shapes here have **E up to 256 experts** with **strongly imbalanced token counts** and **wide FFN aspect** (`F ≈ 2–3 × D`). Your job is to make it fast without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `wide_expert_swiglu_moe(x, w1, w2, topk_idx, topk_w)`; signature and semantics are fixed. Autograd must work through it: the harness grades four gradients — `dx`, `dw1`, `dw2`, `dtopk_w`. `topk_idx` has no gradient.

## The operator

Shapes: `x [T, D]`, `w1 [E, D, 2F]`, `w2 [E, F, D]`, `topk_idx [T, A]` int64, `topk_w [T, A]`. Output `y [T, D]`. Per token `t` with `h = x[t]` and `e = topk_idx[t, a]`:

- `f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]`
- `y[t] = sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])`

`silu(z) = z · sigmoid(z)`. Duplicate experts per token legal; each slot contributes independently. Experts no token routes to receive exactly zero weight gradients. `reference.py` is the definition of correctness.

## How you are graded

One float in `[0, 1]`; zero unless every gate passes: no forbidden library call; output and all four gradients agree with `reference.py` under disclosed tolerances on a hidden shape set (with `E = 256`, non-power-of-two, `A = 1`, collapsed routing); three runs bitwise identical; ≥ 60% of graded device time in your declared Triton kernels, forward+backward.

Above the gates the score is a geomean of `(production time) / (your time)`, normalized against a hidden target. **The denominator is a cuBLAS grouped-GEMM composition on expert-sorted tokens**; at `E = 256` with strong zipf and wide `F`, that composition leaves the largest headroom in the family — its scheduler has no persistent-CTA policy for imbalanced-segment grouped GEMMs, and it materializes the wide intermediate `act[T·A_eff, F]`.

The routing is drawn from a hostile popularity distribution (strength `2.0 – 3.5`, hidden per invocation), with a hidden per-invocation input scale. The tensors that are timed are the tensors that are checked.

## The correctness harness

1. **Smoke.** Small input, forward and backward.
2. **Shape sweep.** Hidden shape set (`E` up to 256, `F ≈ 2–3 × D`), both dtypes, output and all four grads per cell. Timed here.
3. **Stability.** Adversarial inputs. Finiteness gate only.
4. **Determinism.** Three runs, bitwise identical output and all four grads; A=8 probe.
5. **Edge cases.** Non-power-of-two dims, `A = 1`, collapsed routing at `E = 256`.

## The rubric check

Same family rubric set. Every rubric must pass. Judge-unavailable is not zero.

## Requirements

**Full routing, all four gradients, determinism, recomputation, data independence, kernels-that-launch, no address-the-reader, honest measurement** — family baseline. Additionally: **skip no expert**, however cold. Truncating the zipf tail (a tempting shortcut at `E = 256`) drops slots that carry both `y` and gradient contributions.

## Disclosed thresholds

| rule | value |
|---|---|
| operator | SwiGLU MoE, wide FFN (`F ≈ 2–3 × D`), high `E` (up to 256) |
| output tolerance | bfloat16: atol 0.16, rtol 0.10 · float32: atol 0.05, rtol 0.01 |
| gradient tolerance | bfloat16: atol 20.0, rtol 0.10 · float32: atol 6.0, rtol 0.02 — the hot expert can hold tens of thousands of routed rows |
| graded gradients | `dx`, `dw1`, `dw2`, `dtopk_w` |
| surface | forward + backward, timed and checked together; loss = sum over `y` |
| written-kernel gate | at least **60%** of graded device time in your `@triton.jit` kernels, fwd+bwd |
| determinism | 3 runs after warmup, bitwise identical output and all four gradients; A=8 probe |
| stability | no raise, no non-finite where reference is finite |
| dtypes | float32 and bfloat16, both checked and both timed |
| routing | precomputed; **strong zipf** (2.0–3.5, hidden per invocation); loads dramatically imbalanced |
| forbidden calls | listed under Constraints; word-boundary scan, comments stripped |
| rubric gate | every rubric must pass |

## Constraints

- One H100. **No internet.** torch, triton, einops, pytest installed.
- Do not change the semantics, signature or output dtype of `wide_expert_swiglu_moe`.
- Forbidden in `kernel.py`: `torch._grouped_mm`, `_grouped_mm`, `grouped_mm`, `fused_moe`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `cutlass`, `flashinfer`.
- Uniform numeric policy; no branching on input magnitude, load balance, or timing context.
- Weights are large (up to 256 experts × 4096 × 3072); use 64-bit index arithmetic where offsets can cross `2^31`.
- The gain must come from your own kernel work.
