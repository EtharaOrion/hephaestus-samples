# Task: implement a fast decode-regime Mamba-2 SSD selective scan, forward only

`kernel.py` contains a working but deliberately slow implementation of the Mamba-2 SSD (State-Space Duality) selective state update, run in its decode regime — one token per step from an explicit initial state. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `mamba2_ssd_decode(x, dt, A_log, B, C, D, dt_bias, state0)`. Its signature and semantics are fixed, and **both** of its outputs are graded: the per-step outputs `o` and the final state `state_out`. There is no backward pass: this operator is graded forward-only.

## The operator

For each batch and head, with state `S` of shape `[P, N]` starting from the caller's `state0`:

```
dt_t = softplus(dt_raw_t + dt_bias)     # positive time step, scalar per head
A    = -exp(A_log)                       # scalar per head, strictly negative
dA_t = exp(dt_t * A)                      # scalar per head, in (0, 1)
S_t  = dA_t . S_{t-1} + (dt_t * x_t) B_t^T
y_t  = S_t C_t + D * x_t
```

Shapes: `x` is `[B, S, H, P]` (per-head input / value), `dt` is `[B, S, H]` (raw time-step, pre-softplus, float32), `A_log` and `D` and `dt_bias` are `[H]` (float32, scalar per head), `B` and `C` are `[B, S, H, N]` (the selective input/output vectors), and `state0` is `[B, H, P, N]` float32. The call returns `(o, state_out)` with `o` of shape `[B, S, H, P]` in `x`'s dtype and `state_out` of shape `[B, H, P, N]` in **float32**. Ungrouped multi-head — one `(B, C)` pair per head.

Every part matters and is graded: the time step is `softplus(dt_raw + dt_bias)` (not the raw value); `A` is `-exp(A_log)` (strictly negative, so `dA` lands in `(0, 1)`); the write is the rank-1 outer product of the discretized input `dt_t * x_t` with `B_t`; the read is `S_t C_t` contracted over `N`; and `D * x_t` is an input skip added to every output. The recurrence starts from `state0`, not from zero — a kernel that ignores it is wrong on every graded input — and `state_out` is the state after the **complete** last step. `reference.py` writes this out as a sequential float32 scan and is the definition of correctness — where this text and that code ever disagree, the code wins. Carry the state in float32.

This is the heaviest operator in its family: the state is a full `[P, N]` matrix per head and each step consumes five selective parameter streams. The `N` axis is the reduction axis of the `y = S C` read, so a design that keeps the whole `[P, N]` state of one `(batch, head)` pair resident (with `N` un-tiled) computes each output without any cross-program reduction.

Why the starter is slow, in one sentence: it launches one kernel per timestep, paying `S` launch latencies and moving the entire `[P, N]` state through HBM twice per step, while the state of one `(batch, head)` pair fits in the registers of a single program for the whole sequence — the compulsory traffic is the `x/dt/B/C` stream, `o`, and **one** state read plus **one** state write in total.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- **G1 — no library.** Your submission does not import or call any library implementation of this operator — no selective-scan / selective-state-update kernel, no linear-attention library. Reading library source is allowed; calling or shipping it is not. Checked by a word-boundary symbol scan of your comment-stripped source (banned names under *Constraints*; string literals and docstrings count, `#` comments do not). *(There is no published production kernel of this exact operator to vendor, so unlike the sibling tasks there is no structural-copy comparison here — but the symbol scan is live.)*
- **G2 — correctness.** Your output — `o` **and** `state_out` — agrees with `reference.py` within the tolerances below on a hidden shape set larger than, and different from, the shapes `bench.py` lets you measure. The hidden set includes non-power-of-two batch/sequence/head counts, the single-step `S = 1` case, a single-sequence `B = H = 1` case, and a large-state shape; `P` and `N` are always powers of two.
- **G4 — written-kernel share.** **At least 60% of graded device time runs inside Triton kernels your own module declares.** Measured after the fact by device-time attribution. Delegating the recurrence to framework operators scores zero here.

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production baseline time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is a `torch.compile` of a fused selective step run in the decode loop — the strongest reproducible torch baseline for this operator, since no hand-tuned production kernel of it exists — timed on the same tensors your kernel is timed on.** Matching the slow starter scores near zero; reaching the compiled baseline's speed scores 1. This is the hardest member of the family: the large state and the five selective streams leave the least headroom.

The magnitude of `x` and `state0` is drawn at random from a hidden distribution on every invocation, and the tensors that are timed are the same tensors that are checked. A branch that detects the checker's input statistics and uses cheaper math elsewhere gains nothing here, and is a rubric violation besides.

## The written-kernel requirement

Both graded outputs — `o` and `state_out` — must be produced by `@triton.jit` kernels your module declares and actually launches. The 60% device-time floor is the machine-checkable half; the intent is that the recurrence itself is your kernel's work. Choosing tile sizes or launch parameters from the static **shapes** is fine; selecting a branch by the runtime **values** of any input is not.

## Constraints

- One H100. **No internet.** torch, triton and einops are installed; `pip install`, `git clone` and downloads will fail.
- Do not change the semantics, signature, or output dtypes of `mamba2_ssd_decode`. `state0` is read-only — never mutate a caller tensor.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped, strings and docstrings count): `fla.ops`, `import fla`, `from fla`, `selective_state_update`, `selective_scan`, `mamba_chunk_scan`, `mamba_ssm`, `mamba_split_conv1d_scan`. The recurrence is the task; calling a library's implementation of it is not a solution.
- Compute the time step as `softplus(dt_raw + dt_bias)` and the decay as `exp(dt * A)` with `A = -exp(A_log)`; keep the `D * x` skip. No step may be skipped, truncated, or approximated away.
- Numerical policy must be uniform: no branch on input magnitude, distribution, or seed that changes which algorithm runs. Carry the state in float32 from `state0` to `state_out`.
- Determinism: repeated calls on one input must be bitwise identical on **both** outputs — no atomics on the state or output path, no scheduling-dependent reduction order.

## What is hidden, and what is not

The rules above are complete. Hidden from you are: the exact graded shape set (larger than and disjoint from `bench.py`'s published shapes, and including non-power-of-two `B`/`S`/`H`, `S = 1`, `B = H = 1`, and a large-state shape), the per-invocation magnitude of `x`/`state0`, and the target the geometric-mean fraction is normalized against. Disclosed thresholds:

| rule | value |
|---|---|
| bfloat16 tolerance on `o` | atol **2e-2**, rtol **3.125e-2** |
| float32 tolerance on `o` | atol **2e-3**, rtol **2e-3** |
| `state_out` tolerance | float32, atol **2e-3**, rtol **2e-3**, on every input dtype |
| output dtypes | `o` in `x`'s dtype; `state_out` in float32, always |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| determinism | **3** runs on one input, bitwise identical `o` and `state_out` |
| stability | no raise, no non-finite where `reference.py` is finite (includes a 1024-step probe, a large-`dt` probe, and a slow-decay probe) |

## The refinement loop

`./bench.py` gates in the grader's order — forbidden-name scan, correctness on both outputs (including a non-power-of-two shape and `S = 1`), three-run bitwise determinism, written-kernel share — and only then times you against the compiled baseline on a small published set of shapes. It does not tell you your score: the published shapes are not the graded shapes, the graded set randomizes hidden magnitudes and grades float32, and `bench.py` does not run the 1024-step stability probe. `--json <path>` writes the same results machine-readably. Steer with it, then resubmit; you may iterate up to the session cap, and the best-scoring submission is the one that counts.
