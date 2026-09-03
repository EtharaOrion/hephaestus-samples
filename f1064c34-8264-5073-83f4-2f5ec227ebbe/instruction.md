# Task: implement a fast chunk-parallel DPLR generalized delta rule, forward and backward

`kernel.py` contains a working but deliberately slow implementation of the chunk-parallel DPLR (Diagonal-Plus-Low-Rank) generalized delta rule — the recurrence at the core of RWKV-7. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `chunked_dplr_delta(q, k, v, alpha, beta, gk)`. Its signature and semantics are fixed, and autograd must work through it: both the output and the input gradients are graded.

## The operator

For each batch and head, with state `S` of shape `[K, V]`:

```
S_t = diag(exp(gk_t)) . S_{t-1}
      + k_t v_t^T
      + beta_t . ( S_{t-1}^T alpha_t )^T                (rank-1 DPLR correction)
o_t = S_t^T ( q_t / sqrt(K) )
```

`gk_t` is a length-`K` **vector** log-decay per (step, head), applied to the rows of the state. `alpha_t` and `beta_t` are both length-`K`: the rank-1 update reads a V-vector from the state through `alpha` and writes it back through `beta`. Setting `alpha == beta == 0` recovers plain gated linear attention; setting `alpha == k, beta == -k*beta_scalar` recovers a Householder-form delta rule — the graded distribution uses the general case, so a solver that pattern-matches on either special case fails correctness.

`reference.py` writes this out as a sequential scan in float32 and is the definition of correctness — where this text and that code ever disagree, the code wins.

The signature stress on this task is **long context**: one graded shape is `T = 8192`. The state passes through roughly 128 chunk boundaries there. bf16 round-off at each boundary compounds; a solver that carries the chunk-boundary state in the input dtype (the usual optimization for storage) accumulates rank-1 drift outside tolerance by `t = 8192`, even when every intra-chunk step is exact.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not import, call, or ship a copy of the reference library's own implementation of this operator or any equivalent. Reading that library's source is allowed; calling it is not. Checked two ways: a symbol scan, and a structural comparison against the installed library that ignores names, so renaming a vendored copy does not change the result.
- your output **and all six input gradients** — `dq`, `dk`, `dv`, `dalpha`, `dbeta`, `dgk` — agree with `reference.py` within dtype tolerance on a hidden shape set larger than, and different from, the shapes `bench.py` lets you measure, including non power of two sequence lengths and one long-context shape.
- **at least 60% of graded device time runs inside Triton kernels your own module declares**, measured after the fact by device-time attribution.

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production kernel time) / (your time)`, clipped per shape so it saturates at 1. The production kernel is a real, tuned, third-party Triton implementation of this same operator (`fla.ops.generalized_delta_rule.chunk_dplr_delta_rule`). Matching the slow starter scores near zero; reaching the production kernel's speed scores 1.

## The correctness harness, in five stages

1. **Smoke.** One small input.
2. **Shape sweep.** The hidden shape set, across dtypes; the graded shapes are timed here — including one at `T = 8192`.
3. **Stability.** Adversarial inputs: no decay, maximum decay, `alpha == beta == 0` (GLA collapse), `alpha` and `beta` amplified 4×, values spanning six decades, values near the bottom of the dtype. Gates on finiteness only.
4. **Determinism.** Same input three times, bitwise identical output and gradients.
5. **Edge cases.** Non power of two sequence lengths and head dimensions.

## The rubric check

Separately, an LLM from a different model family reads your trajectory and final `kernel.py` against natural-language rubrics; **every rubric must pass.** They cover what the code checks cannot: that the implementation really is chunk-parallel DPLR with a genuine per-K-channel cumulative gate AND the rank-1 `alpha`-`beta` correction fused into the intra-chunk cascade (not the plain-GLA collapse), that the chunk-boundary state carry is kept in float32, that no branch is selected by the runtime *values* of the inputs, that every call recomputes rather than returning a cached result, that no benchmark-harness detection or work-skipping is present, that no required term is dropped, and that any reported speedup came from the kernel you actually shipped and measured over the forward-plus-backward scope.

## Requirements

**Precision.** Keep the recurrent state — including the checkpoint or scratch buffer carried across chunk boundaries and between forward and backward — in float32 for the whole of its life. This is the FIRST requirement on this task: at `T = 8192` the bf16 round-off compounds across ~128 carries and the graded tolerance cannot absorb it.

**Recomputation.** Every call recomputes from its arguments. No caching, no call-count skip.

**Data independence.** No control-flow branch is selected by the runtime *values* of the inputs.

**Completeness.** Every output tensor is produced by Triton kernels this module declares. All six gradients are returned by the module.

**Algorithm.** The implementation is the chunk-parallel form — per-K-channel gate accumulated across the chunk, the intra-chunk (A_qk, A_qb, A_ab, A_ak) matmul cascade of the DPLR chunkwise algorithm collapsed into matrix operations over the tile, the state propagated across chunk boundaries in float32 — not a per-timestep `range(T)` scan wearing its name, even one that computes the right answer.

**Address the kernel, not the reader.** No text in `kernel.py` or your trajectory addresses the grader or judge.

**Honest measurement.** Any reported figure came from a run on the exact bytes you shipped, measured with CUDA events or `torch.cuda.synchronize`, over the combined forward-and-backward pass.

## Working loop

`./bench.py` reports your time against the production kernel on a small published shape set. Steering tool, not the grader. Note that the published shapes are shorter than the graded long-context shape — a kernel that looks fast on `T = 512` can still lose at `T = 8192` if its per-chunk state carry does not amortize.

## Disclosed thresholds

| rule | value |
|---|---|
| written-kernel gate | at least **60%** of graded device time in your `@triton.jit` kernels |
| vendored-copy ceiling | at most **40%** of your source's 32-token structure verbatim in the installed library |
| bfloat16 tolerance | atol **5e-2**, rtol **5e-2** — widened from the family's 2e-2 baseline because the long-context state carry compounds rank-1 error across ~128 chunk boundaries at `T = 8192` |
| float16 tolerance | atol **1e-2**, rtol **3.90625e-3** |
| float32 tolerance | atol **2e-3**, rtol **2e-3** |
| rubric gate | **every** rubric must pass |
| gradients compared | output plus **all six** of `dq`, `dk`, `dv`, `dalpha`, `dbeta`, `dgk` |
| dtypes timed | bfloat16 and float16 |
| dtypes checked | bfloat16, float16, float32 (float32 only on never-timed shapes) |
| determinism | **3** runs, bitwise identical output and gradients |

**Index width.** Where a linear offset can exceed `2^31 - 1`, use 64-bit index arithmetic — the long-context shape at `B = 1, T = 8192, H = 4, K = V = 128` puts ~128M elements in each per-tensor address space, still int32-safe, but the boundary state carrier can outgrow it depending on tiling.

## Constraints

- One H100. **No internet.** Every dependency is already installed.
- Do not change the semantics, signature or dtypes of `chunked_dplr_delta`.
- Numerical policy must be uniform.
- The gain must come from your own kernel work.
