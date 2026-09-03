# Task: implement a fast hierarchical group-then-expert routed fused-MoE kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of DeepSeek-V2 device-limited routing: a scoring softmax followed by a top-1 GROUP selection (group score = sum of the group's top-2 expert affinities), followed by top-K experts inside the chosen group, combined by `p_sel / p_sel.sum(-1)`. Both selection stages live INSIDE the operator.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `fused_moe(x, router_w, w1, w2, G, K)` with `x: [T, D]`, `router_w: [D, E]` (same dtype as `x`), `w1: [E, D, 2F]`, `w2: [E, F, D]`, `G: int` (number of groups; `E % G == 0`), `K: int` (top-K within chosen group, `1 <= K <= E/G`). It returns `y: [T, D]` in the input dtype. Signature is fixed. **No router bias.** The graded surface is forward AND backward: dx, drouter_w, dw1, dw2 are graded alongside the output.

## The operator

- **Scoring.** `logits = x @ router_w`, `p = softmax(logits, dim=-1)`, in float32. Reshape `p` to `[T, G, S]` with `S = E / G`.
- **Group selection.** For each token, `group_score[t, g] = top2_sum(p[t, g, :])`. `grp[t] = argmax_g group_score[t, g]` with ties broken to **lower group index first**; `±0.0` one tie group.
- **Intra-group top-K.** Inside `p[t, grp[t], :]`, pick the top-K local positions with ties broken to **lower local index first**. Global expert index `sel[t, k] = grp[t] * S + local[t, k]`.
- **Combine.** `w[t] = p[t, sel[t]] / p[t, sel[t]].sum(-1)` (p_sel/sum).
- **Experts.** `f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]`, `y[t] = sum_k w[t, k] * f(sel[t, k], x[t])`.
- **Arithmetic.** float32 accumulation throughout; only final `y` cast back.
- **Gradients.** Selection (both stages) is locally constant. Gradients flow through the scoring softmax (dense over the whole expert row, into `router_w` and `x`), through the p_sel/sum coupling of the K selected logits, and through the expert compute. Experts outside every token's chosen group receive exact-zero weight gradients.

## How you are graded

One float in `[0, 1]`. **Zero** unless every gate passes:

- No banned symbols in `kernel.py`.
- Output and all four gradients agree with `reference.py` under the disclosed tolerances on a hidden shape set (G from 4-ish to 8-ish, E from 8-ish to 160-ish, non-power-of-two `T`, `D`, `F`, and exact-tie draws).
- **At least 60%** of graded forward+backward device time in `@triton.jit` kernels your module declares. `torch.topk` / `torch.sort` / `torch.argsort` on the small `[T, E]` softmax matrix, `[T, G]` group scores, `[T*K]` routing metadata, and `[T, G, S]` group slabs are permitted. Framework grouped or batched matmuls are not.

Above the gates, score is geomean across hidden graded shapes (fp32 AND bf16) of `(production time)/(your time)`, normalized against a hidden target, saturating at 1. **Denominator:** router matmul + softmax + reshape + top-2-sum group score + int-key argmax group + intra-group int-key top-K + gather + p_sel/sum, feeding a `torch._grouped_mm` expert pipeline over tokens sorted by expert.

Disclosed:
- Decision margin `1e-4` on **both** boundaries (group top-1/top-2 gap AND intra-group K/(K+1) gap), or exactly 0 (tie).
- Backward driven by a fixed pseudorandom ±uniform cotangent per output shape.
- Hidden per-invocation magnitude randomization.

## Correctness harness (5 stages)

Smoke → shape sweep (fp32+bf16) → stability (saturated router, group-balanced router, tiny/huge x, near-uniform) → determinism (3 runs bitwise identical incl. gradients on a tie-forcing input) → edge cases (non-power-of-two + exact ties on both boundaries).

## Requirements

**Exact two-stage selection.** Group selection by (`top2_sum(p_group)` desc, group index asc). Intra-group selection by (`p` desc, local index asc). Both boundaries need the int64-key trick to make ties structural.

**p_sel/sum combine.** Weights sum to 1 per token.

**Precision.** float32 score arithmetic. Weight-gradient operand chains (dw1, dw2, drouter_w) float32-effective end to end.

**Dense scoring-softmax backward.** Couples the whole expert row.

**Full dispatch inside the chosen group.** Every (token, expert-in-chosen-group) selected pair is computed. Experts outside every token's chosen group produce exact-zero weight gradients.

**Recomputation, data independence, determinism, completeness, no address-to-judge, honest measurement** — as in the family conventions.

## Working loop

`./bench.py` gates before it times.

## Disclosed thresholds

| rule | value |
|---|---|
| output tolerance | fp32: `atol=rtol=2e-4`; bf16: `4.5e-2` |
| gradient tolerance | fp32: `atol=rtol=4e-3`; bf16: `4.5e-2` |
| selection margin | BOTH group and intra-group gap exactly `0` or `>= 1e-4` |
| tie-break | group: lower group index; intra-group: lower local index; `±0.0` one tie group |
| written-kernel gate | **60%** of fwd+bwd device time in `@triton.jit` kernels you declare |
| routing carve-out | `torch.topk`/`torch.sort`/`torch.argsort` on `[T, E]`, `[T, G]`, `[T, G, S]` or `[T*K]` only |
| dtypes | fp32 and bf16, both checked and both timed |
| determinism | **3** fwd+bwd bitwise identical incl. gradients |
| forbidden calls | see Constraints |

## Constraints

- One H100, no internet.
- Do not change signature or output dtype.
- Names disallowed in `kernel.py` (word-boundary scan): `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`.
- One numerical policy for all inputs.
- Gain from your own kernel work.
