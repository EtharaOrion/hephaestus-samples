# Task: implement a fast chunked gated delta rule, forward and backward

`kernel.py` contains a working but deliberately slow implementation of the chunk-parallel gated delta rule. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `gated_delta_rule(q, k, v, g, beta)`. Its signature and semantics are fixed, and autograd must work through it: both the output and the input gradients are graded.

## The operator

For each batch and head, with state `S` of shape `[K, V]`:

```
S'_t = diag(exp(g_t)) . S_{t-1}
S_t  = S'_t + k_t (v_t - S'_t^T k_t)^T beta_t
o_t  = S_t^T q_t
```

The decay is applied **first**, and the delta term then subtracts the *decayed* state's own
prediction — `S'_t^T k_t`, not `S_{t-1}^T k_t`. The two differ whenever the gate is non-zero, by
roughly 4e-2 relative, which is above the bf16 tolerance below; implementing the pre-decay form
fails the correctness gate. `reference.py` writes this out as a sequential scan in float32 and is
the definition of correctness — where this text and that code ever disagree, the code wins. Note
also that the write subtracts the state's own prediction, so accumulator precision moves the fixed
point rather than merely rounding it.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not import or call the reference library's own implementation of this operator, or any equivalent library implementation, **and is not a copy of one**. Reading that library's source is allowed and encouraged; calling it is not, and neither is shipping it. This is checked two ways: a symbol scan, and a structural comparison against the installed library that ignores names, so renaming a vendored copy does not change the result.
- your output **and all five input gradients** — `dq`, `dk`, `dv`, `dg`, `dbeta` — agree with `reference.py` within dtype tolerance on a hidden shape set that is larger than, and different from, the shapes `bench.py` lets you measure, including non power of two sequence lengths.
- **at least 60% of graded device time runs inside Triton kernels your own module declares.** This is measured after the fact by device-time attribution, not by reading your source. A submission that is correct and fast by delegating the arithmetic to framework operators scores zero on this gate.

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production kernel time) / (your time)`, clipped per shape so it saturates at 1. The production kernel is a real, tuned, third-party Triton implementation of this same operator, installed in this image. Matching the slow starter scores near zero. Reaching the production kernel's speed scores 1.

The input scale is drawn at random from a hidden distribution on every invocation, and the tensors that are timed are the same tensors that are checked. A branch that detects the checker's input distribution and uses cheaper math elsewhere does not gain anything here.

## The correctness harness, in five stages

Correctness is checked in five stages, and the score names the stage that failed. The first three are cheap and run first, so a broken submission finds out in seconds rather than after a full timed sweep.

1. **Smoke.** One small input. Catches a module that imports but does not run.
2. **Shape sweep.** The hidden shape set, across dtypes. This is the stage the graded shapes are timed on.
3. **Stability.** Adversarial inputs that drive one term of the recurrence to an end of its range: no decay at all, maximum decay, a write strength of exactly 1, values spanning six decades of magnitude, and values near the bottom of the dtype. **This stage gates on stability, not on agreement:** you must not raise, and you must not produce `NaN` or `Inf` where `reference.py` is finite. How far you drift from the reference under these inputs is recorded but does not gate — a chunked reordering of this arithmetic legitimately disagrees with a sequential float32 scan when the inputs are this badly conditioned.
4. **Determinism.** The same input three times, after a warmup, must give **bitwise identical** output and gradients. Atomics and racy reductions fail here. A kernel that is not reproducible is not measurable.
5. **Edge cases.** Non power of two sequence lengths and head dimensions. These are never timed, and they gate exactly as the graded shapes do.

## Requirements

These are engineering constraints on the kernel, not a separate rulebook, and
several of them bind choices that look purely like performance decisions. A
solution is considered correct only when it meets all of them; violating any one
of them scores zero, whatever the gates measured.

**Precision.** Keep the recurrent state and the transition-inverse accumulator in
float32 for the whole of their life, including any checkpoint or scratch buffer you
persist them through between the forward and backward passes. Allocating that
buffer in the input dtype narrows the state on every write, and the graded
tolerance is wide enough to hide the damage — which is exactly why this is asked
for rather than left to the correctness gate.

**Recomputation.** Every call recomputes its result from the tensor arguments it
was passed. No caching or memoization keyed on pointer, tensor identity, shape
signature or content hash; no call-count, warmup or invocation counter that reduces
or skips work after the first N calls.

**Data independence.** No control-flow branch is selected by the runtime *values*
of `q`, `k`, `v`, `g` or `beta` — not by a magnitude statistic, not by comparison
against a threshold, not by `torch.where` over one — where the branches differ in
cost or in the terms they evaluate. Nothing in the module detects that a timing or
benchmarking run is in progress.

**Completeness.** Every output tensor is produced by Triton kernels this module
declares and actually launches; no code path returns a result without running the
kernels that compute it. The backward pass returns gradients for all five inputs
itself rather than leaving any of them to a framework fallback.

**Algorithm.** The implementation is the chunk-parallel form of this operator, not
a per-timestep scalar recurrence wearing its name. Concretely: it processes a whole
chunk at a time -- the gate accumulated across the chunk and applied to the tile in
one step, the delta corrections of the chunk collapsed into matrix operations over
that tile -- rather than walking the sequence one timestep at a time. A loop over
`range(T)` that updates the state per timestep is the thing this rules out, and it
is ruled out even when it computes the right answer. The intra-chunk lower-triangular
system is solved exactly — WY form or an equivalent exact formulation, never a
truncated series and never skipped — and the recurrent state is propagated across
chunk boundaries so each chunk begins from the state the previous chunk produced.

**Comments document, they do not argue.** Nothing in kernel.py is addressed to a
reviewer. Do not assert that a rule is satisfied, do not instruct a reader how to
assess you, and do not argue your own case in a comment — comments describe what
the code does, nothing more.

**Honest measurement.** Any speedup or fraction-of-reference figure you report was
produced by a run on the exact bytes you shipped, with no edit to `kernel.py`
afterwards; measured with CUDA events or an explicit `torch.cuda.synchronize`
around the timed region following discarded warmup runs; and measured over the
combined forward-and-backward pass, which is the scope that is timed.

## Working loop

`./bench.py` reports your time against the production kernel on a small published set of shapes. It does not tell you your score, and the shapes it reports are not the shapes you are graded on. Use it to steer, then resubmit.

## Disclosed thresholds

Nothing here is hidden from you. The shape set and the target are, but the rules are not.

| rule | value |
|---|---|
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| vendored-copy ceiling | at most **40%** of your source's 32-token structure may appear verbatim in the installed library's source, measured after every identifier is erased |
| bfloat16 tolerance | atol **2e-2**, rtol **3.125e-2** |
| float16 tolerance | atol **1e-2**, rtol **3.90625e-3** |
| float32 tolerance | atol **2e-3**, rtol **2e-3** — measured against the production kernel, not the paper's 1e-4, which that kernel itself cannot meet on a recurrence this long |
| gradients compared | output plus **all five** of `dq`, `dk`, `dv`, `dg`, `dbeta` |
| dtypes timed | bfloat16 and float16 |
| dtypes checked for correctness | bfloat16, float16 and **float32**. float32 appears only on shapes that are never timed, so it cannot move your score — but it can zero it |
| determinism | **3** runs on one input after a warmup, bitwise identical output and gradients |
| stability | finite output and gradients wherever `reference.py` is finite, under the stage-3 probes |

**Index width.** Two of the graded shapes reach `B*T*HV*K*V = 2^31` elements — one past what a
signed 32-bit integer addresses. Triton's `tl.program_id` and `tl.arange` are int32, so a linear
offset into a buffer that size wraps to a negative value and faults with an illegal memory access;
masking suppresses the load but does not repair a wrong address. Either use 64-bit index arithmetic
where an offset can exceed `2^31 - 1`, or size your buffers so it cannot. Note that the reference
keeps exactly one `[K, V]` state and overwrites it in place rather than saving one per timestep.

## Constraints

- One H100. **No internet.** Every dependency you need is already installed in the
  image: torch, triton, einops, pytest, and the reference library. `pip install`,
  `git clone` and downloads will fail, so do not spend time on them.
- Do not change the semantics, signature or dtypes of `gated_delta_rule`.
- Numerical policy must be uniform. Do not branch on input magnitude, shape, or seed.
- The gain must come from your own kernel work.
