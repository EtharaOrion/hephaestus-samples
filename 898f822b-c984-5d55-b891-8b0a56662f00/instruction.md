# Task: implement a fast chunk-parallel COMBA (auxiliary-key delta rule), forward and backward

`kernel.py` contains a working but deliberately slow implementation of chunk-parallel COMBA. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `chunked_comba(q, k, v, p, g, beta)`. Its signature and semantics are fixed, and autograd must work through it: both the output and the input gradients are graded.

## The operator

For each batch and head, with state `S` of shape `[K, V]`:

```
S <- diag(exp(g_t)) . S                                       # gate first
delta_t = ( v_t - S^T p_t ) * beta_t                          # erase against p
S <- S + k_t . delta_t^T                                      # write with k
o_t = S^T ( q_t / sqrt(K) )
```

`p` and `k` are BOTH length-`K` but semantically distinct: `p` is the auxiliary basis the state is regressed against (erase term uses `p`) while `k` is the write basis (the outer product added to the state uses `k`). `g` is a per (step, head) scalar log-decay and `beta` is a per (step, head) scalar write strength. `reference.py` writes this out as a sequential scan in float32 and is the definition of correctness — where this text and that code ever disagree, the code wins.

The single most tempting way to get this wrong is to treat COMBA as a plain delta rule and use `k` for both the erase and the write. That collapses the cross basis to the symmetric `k k^T` and the output and every gradient move outside tolerance on the graded distribution.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not import, call, or ship a copy of the reference library's own implementation of this operator or any equivalent. Reading that library's source is allowed; calling it is not. Checked two ways: a symbol scan, and a structural comparison against the installed library that ignores names, so renaming a vendored copy does not change the result.
- your output **and all six input gradients** — `dq`, `dk`, `dv`, `dp`, `dg`, `dbeta` — agree with `reference.py` within dtype tolerance on a hidden shape set larger than, and different from, the shapes `bench.py` lets you measure, including non power of two sequence lengths.
- **at least 60% of graded device time runs inside Triton kernels your own module declares**, measured after the fact by device-time attribution. Delegating the arithmetic to framework operators scores zero on this gate.

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production kernel time) / (your time)`, clipped per shape so it saturates at 1. The production kernel is a real, tuned, third-party Triton implementation of this same operator (`fla.ops.comba.chunk_comba`). Matching the slow starter scores near zero; reaching the production kernel's speed scores 1.

## The correctness harness, in five stages

1. **Smoke.** One small input.
2. **Shape sweep.** The hidden shape set, across dtypes; the graded shapes are timed here.
3. **Stability.** Adversarial inputs: no decay, maximum decay, `beta = 1`, `p == k` (delta-rule collapse), values spanning six decades, values near the bottom of the dtype. Gates on finiteness only.
4. **Determinism.** Same input three times, bitwise identical output and gradients.
5. **Edge cases.** Non power of two sequence lengths and head dimensions.

## The rubric check

Separately, an LLM from a different model family reads your trajectory and final `kernel.py` against natural-language rubrics; **every rubric must pass.** They cover what the code checks cannot: that the implementation really is chunk-parallel with a genuine WY-style intra-chunk lower-triangular inverse on the CROSS basis `p k^T` (not the collapsed `k k^T`), that no branch is selected by the runtime *values* of the inputs, that every call recomputes rather than returning a cached result, that no benchmark-harness detection or work-skipping is present, that no required term is dropped, and that any reported speedup came from the kernel you actually shipped and measured over the forward-plus-backward scope.

## Requirements

**Precision.** Keep the recurrent state and the transition-inverse accumulator in float32 for the whole of their life.

**Recomputation.** Every call recomputes from its arguments. No caching, no call-count skip.

**Data independence.** No control-flow branch is selected by the runtime *values* of the inputs.

**Completeness.** Every output tensor is produced by Triton kernels this module declares. All six gradients are returned by the module.

**Algorithm.** The implementation is the chunk-parallel form with the intra-chunk lower-triangular inverse built on the CROSS basis `p k^T` (not `k k^T`). The gate is applied before the write; the state is propagated across chunk boundaries so each chunk starts from the state the previous chunk produced.

**Address the kernel, not the reader.** No text in `kernel.py` or your trajectory addresses the grader or judge.

**Honest measurement.** Any reported figure came from a run on the exact bytes you shipped, measured with CUDA events or `torch.cuda.synchronize`, over the combined forward-and-backward pass.

## Working loop

`./bench.py` reports your time against the production kernel on a small published shape set. Steering tool, not the grader.

## Disclosed thresholds

| rule | value |
|---|---|
| written-kernel gate | at least **60%** of graded device time in your `@triton.jit` kernels |
| vendored-copy ceiling | at most **40%** of your source's 32-token structure verbatim in the installed library |
| bfloat16 tolerance | atol **5e-2**, rtol **5e-2** — widened for the cross-basis rank-1 error |
| float16 tolerance | atol **1e-2**, rtol **3.90625e-3** |
| float32 tolerance | atol **2e-3**, rtol **2e-3** |
| rubric gate | **every** rubric must pass |
| gradients compared | output plus **all six** of `dq`, `dk`, `dv`, `dp`, `dg`, `dbeta` |
| dtypes timed | bfloat16 and float16 |
| dtypes checked | bfloat16, float16, float32 (float32 only on never-timed shapes) |
| determinism | **3** runs, bitwise identical output and gradients |

**Index width.** Where a linear offset can exceed `2^31 - 1`, use 64-bit index arithmetic.

## Constraints

- One H100. **No internet.** Every dependency is already installed.
- Do not change the semantics, signature or dtypes of `chunked_comba`.
- Numerical policy must be uniform.
- The gain must come from your own kernel work.
