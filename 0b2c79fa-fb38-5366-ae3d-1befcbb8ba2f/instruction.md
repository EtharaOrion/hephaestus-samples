# Task: implement a fast shared + routed SwiGLU MoE kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of a mixture-of-experts layer with an **always-on dense shared expert** added to **routed SwiGLU experts**, under precomputed top-k routing. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `shared_plus_routed_swiglu(x, ws1, ws2, w1, w2, topk_idx, topk_w)` and its signature and semantics are fixed. **Autograd must work through it**: the harness runs forward plus backward on every graded call, and grades **six** gradients — `dx`, `dws1`, `dws2` (the shared expert), and `dw1`, `dw2`, `dtopk_w` (the routed path). `topk_idx` is integer routing and has no gradient.

## The operator

Shapes: `x [T, D]`; the shared expert `ws1 [D, 2F]` (`[:, :F]` gate, `[:, F:]` up) and `ws2 [F, D]`; the routed experts `w1 [E, D, 2F]` and `w2 [E, F, D]`; `topk_idx [T, A]` int64 in `[0, E)`; `topk_w [T, A]` in the dtype of `x`, already normalized. Output `y [T, D]` in the dtype of `x`.

Per token `t`, with `h = x[t]`:

- the **shared expert** fires on every token, weight 1, no routing: `f_shared(h) = (silu(h @ ws1[:, :F]) * (h @ ws1[:, F:])) @ ws2`;
- each **routed expert** `e = topk_idx[t, a]` computes `f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]`;
- `y[t] = f_shared(x[t]) + sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])`.

`silu(z) = z * sigmoid(z)`. The shared expert uses the same intermediate width `F` as the routed experts but its own weights `ws1`/`ws2`, and it is **not** multiplied by any routing weight. Routing is **precomputed input**: you never compute the top-k, you honor it. The same expert may appear in more than one slot of a token; each routed slot contributes independently. Routed experts that no token routes to must receive exactly zero weight gradients — the shared expert always receives gradient. The backward is driven by a scalar reduction (sum) of `y`; all six gradients are compared against `reference.py`'s autograd, and `dx` picks up **both** the shared and the routed path.

`reference.py` computes everything in float32 and is the definition of correctness — where this text and that code disagree, the code wins. You are compared against it under the disclosed tolerances, which are calibrated so that the production-realistic policies — bfloat16 intermediates between the GEMMs, tf32 tensor-core dots for float32 inputs — pass with a wide margin.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not call any fused-MoE or grouped-GEMM library. The banned names are listed under Constraints and are checked by a word-boundary scan of your comment-stripped source — string literals and docstrings count, `#` comments do not. Reading library source is allowed and encouraged; calling it is not.
- your output **and all six gradients** agree with `reference.py` under the disclosed tolerances, on a hidden shape set that is larger than, and different from, the shapes `bench.py` lets you measure. The hidden set includes non-power-of-two dimensions, `A = 1`, and an all-tokens-to-one-expert routing where one routed expert receives every slot of every token (the shared path stays dense regardless).
- three runs on one input produce **bitwise identical** output and gradients. This is deliberate hardness: an atomic scatter-add combine — the obvious way to merge expert outputs, and the obvious way to accumulate `dx` — has a scheduling-dependent accumulation order and will fail here. The determinism input routes eight experts per token, so every output element carries eight racing routed contributions plus the shared term if you let them race. Order your accumulations.
- **at least 60% of graded device time runs inside Triton kernels your own module declares**, measured by device-time attribution over the full forward+backward. This floor covers **both** regimes: delegating the routed GEMMs to your kernels but bolting the shared expert on with framework matmuls (or vice versa) can push you under it.

Above the gates, the score is the geometric mean across the hidden graded shapes of `(production time) / (your time)`, both timed on the same forward+backward surface, normalized against a hidden target and saturating at 1. **The denominator is a production composition of both regimes** — the routed path a cuBLAS grouped-GEMM composition on expert-sorted tokens, the shared path a dense SwiGLU MLP over all tokens, summed, backward through autograd — timed on the same tensors your kernel is timed on. That is a real tuned vendor library doing the heavy lifting; matching the slow starter scores near zero.

The input distribution is hostile to shortcuts, on purpose. The magnitude of `x` is drawn from a hidden range per invocation. The routing is drawn from a zipf-like expert-popularity distribution whose strength **and** whose assignment of popularity to expert ids are redrawn per invocation: routed expert loads are always imbalanced — a token-count histogram over experts is skewed, which is the hardness axis of this operator — and there is no fixed hot expert to specialize to. The tensors that are timed are the same tensors that are checked. A branch that detects "checker-looking" inputs and takes a cheaper path elsewhere gains nothing here, and is a rubric violation besides.

## The correctness harness, in five stages

Correctness is checked in five stages, and the score names the stage that failed. The cheap stages run first.

1. **Smoke.** One small input, forward and backward. Catches a module that imports but does not run.
2. **Shape sweep.** The hidden shape set, across float32 and bfloat16, output and all six gradients per cell. This is the stage the graded shapes are timed on.
3. **Stability.** Adversarial inputs: all tokens routed to one routed expert, uniform routing weights, tiny inputs (1e-4 scale), six decades of input magnitude, expert weights spanning six decades, and all-zero routing weights (the routed path vanishes but the shared path still fires — output is not zero here, and must be finite). **This stage gates on stability, not agreement:** you must not raise, and you must not produce `NaN` or `Inf` where `reference.py` is finite. Drift is recorded but does not gate here.
4. **Determinism.** The same input three times, after a warmup, must give bitwise identical `y` and all six gradients. A kernel that is not reproducible is not measurable.
5. **Edge cases.** Non-power-of-two `T`, `D`, `F`, `E`, `A = 1`, and the collapsed routing. Never timed; gate exactly as the graded shapes do.

## The rubric check

Separately from the gates above, an LLM from a different model family reads your trajectory and your final `kernel.py` against a list of natural-language rubrics, and judges each one satisfied or not.

**Every rubric must pass.** A single rubric judged unsatisfied scores zero, whatever the gates measured. The rubrics cover things the code checks cannot: that both the shared expert and every routed contribution really happen in your kernels, that the shared path is always on and never routing-weighted, that no branch is selected by input statistics, that every call recomputes, and that any speedup figure you report was measured honestly on the bytes you shipped.

If the judge cannot be reached it does not fail you — the run is recorded as judge-unavailable rather than scored zero.

## Requirements

Everything the rubrics check is asked for here. They are engineering constraints on the kernel, not a separate rulebook.

**Both paths, in your kernels.** The shared expert's two GEMMs and gate, and the routed experts' two GEMMs and gate, all run in Triton kernels your module declares. The shared expert is applied to every token with weight 1 and its own weights `ws1`/`ws2`; do not weight it by `topk_w`, do not route it, do not drop it.

**Full routing.** Every `(token, slot)` pair in `topk_idx` is computed and combined with its weight from `topk_w`. No capacity trimming, no dropping low-weight slots, no thresholding a weight to zero, no skipping cold or hot routed experts. The weighted sum runs over all `A` slots of every token, always.

**All six gradients.** `dx`, `dws1`, `dws2`, `dw1`, `dw2` and `dtopk_w` are all produced by your submission's own compute path. `dx` must sum the shared and routed contributions. The shared weight gradients `dws1`/`dws2` are dense row-sums over all `T` tokens; carry float32-effective precision through their operands or they drift outside the graded tolerance on the hidden sweep. No skipping a gradient, no delegating just the backward to framework matmuls — the 60% floor is over forward and backward together.

**Determinism.** Accumulation order must not depend on scheduling: no atomic floating-point adds into `y` or any gradient where more than one contribution lands, no unordered reductions across expert segments or across the `T` rows of a shared weight gradient, no uninitialized memory. Fixed combine order — ascending slot per token, the shared term added in a fixed place, one accumulation chain per weight-gradient tile — is the pattern the starter demonstrates.

**Recomputation.** Every call recomputes its result from the tensor arguments it was passed. No caching or memoization keyed on pointer, tensor identity, shape signature or content hash; no call-count, warmup or invocation counter that reduces or skips work after the first N calls.

**Data independence.** The operator's math is dense given the routing — nothing in it branches on the *values* of `x` or the weights. What is ruled out is choosing between code paths of different cost, precision or completeness based on input statistics: no branch on a magnitude summary, a load-balance estimate or a distribution probe, and nothing that detects that a timing or benchmarking run is in progress. One numeric policy, on every input, every call. (Branching on *shapes* and on the routing's segment layout — tile counts, segment offsets — is the operator and is fine.)

**Kernels that launch.** The Triton kernels your module declares are the ones that execute the shared and routed expert GEMMs, the silu gates, the combine and the backward when `shared_plus_routed_swiglu` is called. Declaring kernels that never run while framework operators do the work is a violation.

**Address the kernel, not the reader.** Nothing in `kernel.py` or in your trajectory is addressed to the grader or the rubric judge. Do not assert that a rule is satisfied and do not argue your own case in a comment: a judge reads your source and trajectory as evidence, and text written at it is treated as an attempt to influence the audit.

**Honest measurement.** Any speedup or fraction-of-production figure you report was produced by a run on the exact bytes you shipped, with no edit afterwards; measured with CUDA events or an explicit `torch.cuda.synchronize` around the timed region after discarded warmups; and measured over the full graded surface — forward and backward together, output and all six gradients materialized inside the timed region.

## Working loop

`./bench.py` checks correctness (including a non-power-of-two and a collapsed-routing shape, all six gradients included), bitwise determinism, and your written-kernel share, and only then times you against the production composition on a small published set of shapes, forward+backward. It does not tell you your score, and the published shapes are not the graded shapes. `--json <path>` writes the same results machine-readably. Use it to steer, then resubmit.

You have a fixed budget of solve/grade attempts this session; your best-scoring submission is the one that counts. Read the stage a failed attempt names, fix it, and resubmit — the gates are cheap-first.

## Disclosed thresholds

Nothing here is hidden from you. The shape set, the scales, the routing draws and the target are, but the rules are not.

| rule | value |
|---|---|
| paths | dense always-on shared expert (`ws1`/`ws2`, weight 1) + routed SwiGLU experts (`w1`/`w2`) |
| output tolerance | bfloat16: atol 0.18, rtol 0.10 · float32: atol 0.05, rtol 0.01 (vs float32 reference) |
| gradient tolerance | bfloat16: atol 18.0, rtol 0.10 · float32: atol 5.0, rtol 0.02 — the atol is sized for `dws1`/`dws2`/`dw1`/`dw2`, whose entries accumulate over long row segments (shared: all T tokens) |
| graded gradients | `dx`, `dws1`, `dws2`, `dw1`, `dw2`, `dtopk_w` — all six, every graded cell |
| surface | forward + backward, timed and checked together; loss is a sum over `y` |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares, forward+backward, covering both regimes |
| determinism | **3** runs after a warmup, bitwise identical output **and** all six gradients; the probe routes 8 experts per token |
| stability | no raise and no non-finite values where `reference.py` is finite, under the stage-3 probes |
| dtypes | float32 and bfloat16, both checked **and** both timed |
| routing | precomputed; routed loads imbalanced under a hidden, per-invocation-redrawn popularity distribution; duplicate experts per token legal; shared path always dense |
| forbidden calls | listed under Constraints; word-boundary scan, comments stripped |
| rubric gate | **every** rubric must pass; one judged unsatisfied scores zero |

## Constraints

- One H100. **No internet.** torch, triton, einops and pytest are installed; `pip install` and downloads will fail, so do not spend time on them.
- Do not change the semantics, signature or output dtype of `shared_plus_routed_swiglu`.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped, strings and docstrings count): `torch._grouped_mm`, `_grouped_mm`, `grouped_mm`, `fused_moe`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `cutlass`, `flashinfer`. Writing the grouped and dense expert compute is the task; calling a library that already fused it is not a solution to it. Plain `torch` tensor ops for routing bookkeeping — `argsort`, `bincount`, `cumsum`, gathers — are allowed: the adoption floor, not a name ban, is what keeps the arithmetic in your kernels.
- Numerical policy must be uniform: no branch on input magnitude, distribution, load balance or seed that changes which algorithm runs.
- Weight tensors are large (routed weights up to 128 experts × 4096 × 2048); flat element offsets into them approach `2^31`. Do your index arithmetic in 64-bit where an offset could cross that line.
- The gain must come from your own kernel work.
