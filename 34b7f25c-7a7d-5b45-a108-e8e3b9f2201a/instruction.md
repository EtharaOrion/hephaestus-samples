# Task: implement a fast fused MoE + ReLU²-gated GLU kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of mixture-of-experts expert compute with a **ReLU² (squared-ReLU) gated GLU** FFN and precomputed top-k routing. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `relu2_glu_moe(x, w1, w2, topk_idx, topk_w)` and its signature and semantics are fixed. **Autograd must work through it**: the harness runs forward plus backward on every graded call, and grades the gradients of `x`, `w1`, `w2` and `topk_w`. `topk_idx` is integer routing and has no gradient.

## The operator

Shapes: `x [T, D]`, `w1 [E, D, 2F]`, `w2 [E, F, D]`, `topk_idx [T, A]` int64 with values in `[0, E)`, `topk_w [T, A]` in the dtype of `x`, already normalized. Output `y [T, D]` in the dtype of `x`.

Per token `t`, with `h = x[t]`:

- each routed expert `e = topk_idx[t, a]` computes `f(e, h) = ((relu(h @ w1[e, :, :F]))² * (h @ w1[e, :, F:])) @ w2[e]` — columns `[:F]` of `w1` are the **gate** projection (activated), columns `[F:]` are the **up** projection (multiplied), and `relu(z) = max(z, 0)`;
- `y[t] = sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])`.

The gate nonlinearity is **exact squared-ReLU** — no smoothing, no `sigmoid` approximation, no `epsilon`. The backward is piecewise: `d/dg [relu(g)² · u] = 2 · relu(g) · (g > 0) · u`, zero on the negative half of the gate's support. Reference.py is the definition; a candidate that substitutes silu, gelu or a smoothed variant drifts out of tolerance on the hidden sweep and scores zero on G2.

Routing is **precomputed input**: you never compute the top-k, you honor it. The same expert may appear in more than one slot of a token; each slot contributes independently. Experts that no token routes to receive exactly zero weight gradients. The backward is driven by a scalar reduction (sum) of `y`; all four gradients — `dx`, `dw1`, `dw2`, `dtopk_w` — are compared against `reference.py`'s autograd.

`reference.py` computes everything in float32 and is the definition of correctness — where this text and that code disagree, the code wins.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not call any fused-MoE or grouped-GEMM library. The banned names are listed under Constraints and are checked by a word-boundary scan of your comment-stripped source — string literals and docstrings count, `#` comments do not.
- your output **and all four gradients** agree with `reference.py` under the disclosed tolerances, on a hidden shape set larger than, and different from, the shapes `bench.py` lets you measure. The hidden set includes non-power-of-two dimensions, `A = 1`, and an all-tokens-to-one-expert routing.
- three runs on one input produce **bitwise identical** output and gradients. The determinism input routes eight experts per token, so every output element carries eight racing contributions if you let them race.
- **at least 60% of graded device time runs inside Triton kernels your own module declares**, measured over the full forward+backward.

Above the gates, the score is the geometric mean across the hidden graded shapes of `(production time) / (your time)`, both timed on the same forward+backward surface, normalized against a hidden target and saturating at 1. **The denominator is a cuBLAS grouped-GEMM composition with an eager ReLU² gate** — tokens gathered in expert order, both expert GEMM stages through cuBLAS grouped matrix multiply, `relu² * up` in a separate elementwise pass, weighted scatter combine, backward through autograd. That anchor cannot fuse the ReLU² epilogue into the first GEMM without a bespoke kernel; you can, and that is the headroom.

The input distribution is hostile to shortcuts. The magnitude of `x` is drawn from a hidden range per invocation. The routing is drawn from a zipf-like expert-popularity distribution whose strength and whose assignment of popularity to expert ids are redrawn per invocation. The tensors that are timed are the same tensors that are checked.

## The correctness harness, in five stages

1. **Smoke.** One small input, forward and backward.
2. **Shape sweep.** The hidden shape set, across float32 and bfloat16, output and all four gradients per cell. Timed here.
3. **Stability.** Adversarial inputs: all tokens routed to one expert, uniform routing weights, tiny inputs (1e-4 scale), six decades of input magnitude, expert weights spanning six decades, and all-zero routing weights (output must be zero with finite gradients). **Gates on stability, not agreement.**
4. **Determinism.** Same input three times after a warmup — bitwise identical.
5. **Edge cases.** Non-power-of-two `T`, `D`, `F`, `E`, `A = 1`, and collapsed routing.

## The rubric check

An LLM from a different model family judges natural-language rubrics on your source and trajectory. Every rubric must pass. If the judge cannot be reached the run is judge-unavailable, not zero.

## Requirements

**Exact ReLU² gate.** The activation is squared-ReLU, not silu-shaped, not smoothed, not epsilon-clipped. Its backward is piecewise (`2·relu(g)·(g>0)·u`).

**Full routing.** Every `(token, slot)` pair is computed and combined with its weight. No capacity trimming, no dropping low-weight slots.

**All four gradients.** `dx`, `dw1`, `dw2` and `dtopk_w` produced by your own compute path.

**Determinism.** Fixed accumulation order; no atomic floating-point adds where multiple contributions land.

**Recomputation.** Every call recomputes; no memoization on pointer, shape signature or content hash.

**Data independence.** One numeric policy on every input. No branch on magnitude, distribution, load balance or timing-context detection.

**Kernels that launch.** Declared Triton kernels must actually execute on every call.

**Address the kernel, not the reader.** No comments arguing your case to the audit.

**Honest measurement.** Any speedup figure comes from a run on the exact bytes shipped, CUDA-event timed after warmup, covering the full forward+backward surface.

## Working loop

`./bench.py` checks correctness (including a non-power-of-two and a collapsed-routing shape, all four gradients), bitwise determinism, and your written-kernel share, and only then times you against the cuBLAS-grouped-GEMM + eager-ReLU² anchor. It does not tell you your score. `--json <path>` writes machine-readable output.

## Disclosed thresholds

| rule | value |
|---|---|
| activation | squared-ReLU gate: `(relu(g))² · u` (no smoothing, no silu/gelu substitute) |
| output tolerance | bfloat16: atol 0.16, rtol 0.10 · float32: atol 0.05, rtol 0.01 |
| gradient tolerance | bfloat16: atol 18.0, rtol 0.10 · float32: atol 5.0, rtol 0.02 |
| graded gradients | `dx`, `dw1`, `dw2`, `dtopk_w` — all four, every graded cell |
| surface | forward + backward, timed and checked together; loss is a sum over `y` |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| determinism | **3** runs after a warmup, bitwise identical output and all four gradients; A=8 probe |
| stability | no raise, no non-finite where `reference.py` is finite |
| dtypes | float32 and bfloat16, both checked and both timed |
| routing | precomputed; loads imbalanced under a hidden, per-invocation-redrawn popularity distribution |
| forbidden calls | listed under Constraints; word-boundary scan, comments stripped |
| rubric gate | every rubric must pass |

## Constraints

- One H100. **No internet.** torch, triton, einops and pytest are installed.
- Do not change the semantics, signature or output dtype of `relu2_glu_moe`.
- These names must not appear in `kernel.py`: `torch._grouped_mm`, `_grouped_mm`, `grouped_mm`, `fused_moe`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `cutlass`, `flashinfer`. Plain `torch` tensor ops for routing bookkeeping (`argsort`, `bincount`, `cumsum`, gathers) are allowed; the adoption floor keeps the arithmetic in your kernels.
- Numerical policy must be uniform: no branch on input magnitude, distribution, load balance or seed.
- Weight tensors are large (up to 128 experts × 4096 × 2048); use 64-bit index arithmetic where an offset could cross `2^31`.
- The gain must come from your own kernel work.
