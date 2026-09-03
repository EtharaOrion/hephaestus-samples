# Task: implement a fast chunk-parallel gated delta-product, forward and backward

`kernel.py` contains a working but deliberately slow implementation of the chunk-parallel gated delta-product rule (`num_householder = 2` Householder writes per token). Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `chunked_gated_delta_product(q, k, v, g, beta, num_householder=2)`. Its signature and semantics are fixed, and autograd must work through it: both the output and the input gradients are graded (except for `num_householder`, which is a plain int argument).

## The operator

For each batch and head, with state `S` of shape `[K, V]`, per token `t`:

```
S <- diag(exp(g_t)) . S                             # one gate per token
for j in 0 .. N-1:                                  # N Householder writes
    S <- S + (v_{t,j} - S^T k_{t,j})^T * k_{t,j} * beta_{t,j}
o_t = S^T (q_t / sqrt(K))
```

`q` carries length `T`. `k`, `v` and `beta` carry length `T*N` — index `t*N + j` is the j-th Householder tuple of token `t`. `g` carries length `T` (one gate per token, applied ONCE before the N-fold write). `beta` is a per (step, head) scalar write strength. The delta correction subtracts the *current* state's own prediction — the writes chain multiplicatively along `j`.

`reference.py` writes this out as a sequential scan in float32 and is the definition of correctness — where this text and that code ever disagree, the code wins. `num_householder = 2` is the fixed value the harness passes; the code path must handle any positive integer, but you may specialize for 2.

The N-fold intra-token write is what makes this hard: the WY-style intra-chunk factorization must invert a `(N*BT) x (N*BT)` lower-triangular system rather than a `BT x BT` one, and the recurrence carries N erasures per token.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not import, call, or ship a copy of the reference library's own implementation of this operator or any equivalent. Reading that library's source is allowed; calling it is not. Checked two ways: a symbol scan, and a structural comparison against the installed library that ignores names, so renaming a vendored copy does not change the result.
- your output **and all five input gradients** — `dq`, `dk`, `dv`, `dg`, `dbeta` — agree with `reference.py` within dtype tolerance on a hidden shape set larger than, and different from, the shapes `bench.py` lets you measure, including non power of two sequence lengths.
- **at least 60% of graded device time runs inside Triton kernels your own module declares**, measured after the fact by device-time attribution. Delegating the arithmetic to framework operators scores zero on this gate.

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production kernel time) / (your time)`, clipped per shape so it saturates at 1. The production kernel is a real, tuned, third-party Triton implementation of this same operator, installed in this image (`fla.ops.gated_delta_product.chunk_gated_delta_product`). Matching the slow starter scores near zero; reaching the production kernel's speed scores 1.

The input scale is drawn at random from a hidden distribution on every invocation, and the tensors that are timed are the same tensors that are checked. A branch that detects the checker's input distribution and uses cheaper math elsewhere gains nothing here, and is a rubric violation besides.

## The correctness harness, in five stages

The score names the stage that failed; the first three are cheap and run first.

1. **Smoke.** One small input. Catches a module that imports but does not run.
2. **Shape sweep.** The hidden shape set, across dtypes. This is the stage the graded shapes are timed on.
3. **Stability.** Adversarial inputs: no decay, maximum decay, `beta = 1`, values spanning six decades, values near the bottom of the dtype. **Gates on stability, not agreement:** you must not raise and must not produce `NaN`/`Inf` where `reference.py` is finite. Drift under these inputs is recorded but does not gate.
4. **Determinism.** The same input three times, after a warmup, must give **bitwise identical** output and gradients. Atomics and racy reductions fail here.
5. **Edge cases.** Non power of two sequence lengths and head dimensions. Never timed; they gate exactly as the graded shapes do.

## The rubric check

Separately, an LLM from a different model family reads your trajectory and final `kernel.py` against natural-language rubrics; **every rubric must pass.** They cover what the code checks cannot: that the implementation really is chunk-parallel with a genuine WY-style intra-chunk lower-triangular inverse handling all N Householder writes, that no branch is selected by the runtime *values* of the inputs, that every call recomputes rather than returning a cached result, that no benchmark-harness detection or work-skipping is present, that no required term (including any of the N writes) is dropped, and that any reported speedup came from the kernel you actually shipped and measured over the forward-plus-backward scope. An unreachable judge records judge-unavailable rather than zero.

## Requirements

**Precision.** Keep the recurrent state and the transition-inverse accumulator in float32 for the whole of their life, including any checkpoint or scratch buffer carried between forward and backward. Narrowing the accumulator to bf16/fp16 changes the fixed point of the delta cascade and the tolerance is wide enough to hide the damage.

**Recomputation.** Every call recomputes from its arguments. No caching keyed on pointer, identity, shape or content hash; no call-count or warmup counter that skips work after the first N calls.

**Data independence.** No control-flow branch is selected by the runtime *values* of `q`, `k`, `v`, `g` or `beta` where the branches differ in cost or in the terms they evaluate. Nothing detects that a timing run is in progress.

**Completeness.** Every output tensor is produced by Triton kernels this module declares and launches. The backward returns gradients for all five gradient-bearing inputs itself.

**Algorithm.** The implementation is the chunk-parallel form — the per-token gate accumulated across the chunk, all N Householder writes per token collapsed into a single (N*BT)-sized WY factorization or an equivalent exact formulation, and the state propagated across chunk boundaries — not a per-timestep `range(T)` scan wearing its name, even one that computes the right answer. The intra-chunk `(N*BT) x (N*BT)` lower-triangular inverse is computed exactly, never truncated or approximated.

**Address the kernel, not the reader.** Nothing in `kernel.py` or your trajectory is addressed to the grader or judge; do not assert a rule is satisfied or argue your own case in a comment.

**Honest measurement.** Any reported figure came from a run on the exact bytes you shipped, measured with CUDA events or `torch.cuda.synchronize` around the timed region after discarded warmups, over the combined forward-and-backward pass.

## Working loop

`./bench.py` reports your time against the production kernel on a small published shape set. It does not tell you your score, and its shapes are not the graded shapes. Use it to steer, then resubmit. The session runs a bounded refinement loop up to the attempt cap; the highest-scoring submission counts.

## Disclosed thresholds

Nothing here is hidden except the graded shape set and the target.

| rule | value |
|---|---|
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| vendored-copy ceiling | at most **40%** of your source's 32-token structure may appear verbatim in the installed library's source, identifiers erased |
| bfloat16 tolerance | atol **5e-2**, rtol **5e-2** — widened from the family's 2e-2 baseline because the N-fold Householder cascade accumulates rank-1 error per token in bf16; measured against the production kernel |
| float16 tolerance | atol **1e-2**, rtol **3.90625e-3** |
| float32 tolerance | atol **2e-3**, rtol **2e-3** |
| rubric gate | **every** rubric must pass; one judged unsatisfied scores zero |
| gradients compared | output plus **all five** of `dq`, `dk`, `dv`, `dg`, `dbeta` |
| dtypes timed | bfloat16 and float16 |
| dtypes checked for correctness | bfloat16, float16 and **float32** (float32 only on never-timed shapes — it cannot move your score but can zero it) |
| determinism | **3** runs on one input after a warmup, bitwise identical output and gradients |

**Index width.** Where a linear offset into a buffer can exceed `2^31 - 1`, use 64-bit index arithmetic (`tl.program_id`/`tl.arange` are int32) or size buffers so it cannot; masking suppresses a load but does not repair a wrong address.

## Constraints

- One H100. **No internet.** torch, triton, einops, pytest and the reference library are already installed; `pip install` and downloads fail.
- Do not change the semantics, signature or dtypes of `chunked_gated_delta_product`.
- Numerical policy must be uniform. Do not branch on input magnitude, shape, or seed.
- The gain must come from your own kernel work.
