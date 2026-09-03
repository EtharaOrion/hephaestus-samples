# Task: implement a fast SPECULATIVE-DECODE gated delta rule, forward only

`kernel.py` contains a working but deliberately slow implementation of the gated delta rule run in a speculative-decode regime: `S` candidate tokens per batch, per-batch commit horizon `commit_len[b]` in `[0, S]`. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `speculative_gated_delta_decode(q, k, v, g, beta, commit_len, state0)`. Its signature and semantics are fixed, and **both** of its outputs are graded: the per-step outputs `o` (computed for every position from the speculative state) and the per-batch commit snapshot `state_out`. No backward.

## The operator

For each batch `b` and head, per step `t = 0..S-1`:

```
S_t = exp(g_t) * S_{t-1} + k_t delta_t^T
      with delta_t = (v_t - S_{t-1}^T k_t) beta_t          (gate BEFORE the write)
o_t = S_t^T (q_t / sqrt(K))                                 (always computed)
state_out[b] = S_{commit_len[b] - 1}   if commit_len[b] > 0 else state0[b]
```

Shapes: `q`/`k` are `[B, S, H, K]`, `v` is `[B, S, H, V]`, `g` and `beta` are `[B, S, H]` (with `g` float32), `commit_len` is `[B]` int32 in `[0, S]`, `state0` is `[B, H, K, V]` float32. The call returns `(o, state_out)` with `o` of shape `[B, S, H, V]` in `v`'s dtype and `state_out` of shape `[B, H, K, V]` in **float32**.

Every part matters and is graded: `o` is always computed from the SPECULATIVE state (an acceptance check downstream reads every position); the per-batch snapshot for `state_out` is taken at `commit_len[b]` steps of advance, and MUST equal `state0[b]` when `commit_len[b] == 0`. `reference.py` writes this out as a sequential float32 scan and is the definition of correctness — where this text and that code ever disagree, the code wins. Carry the state in float32.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- **G1 — no library.** Your submission does not import or call any delta-rule library implementation. Checked by a word-boundary symbol scan of your comment-stripped source (banned names under *Constraints*).
- **G2 — correctness.** Your output — `o` **and** `state_out` — agrees with `reference.py` within the tolerances below on a hidden shape set that also randomizes `commit_len` per batch (including at least one batch with `commit_len == 0` and one with `commit_len == S`). Includes non-power-of-two `B`/`S`/`H`, `S = 1`, single-sequence, and large-state shapes; `K` and `V` are always powers of two.
- **G4 — written-kernel share.** **At least 60% of graded device time runs inside Triton kernels your own module declares.** The commit-snapshot logic counts too: doing it on the host by a Python loop over batches will pull your adoption share down.

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production baseline time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is a `torch.compile` of a fused gated-delta step + a masked commit snapshot in a decode loop — the strongest reproducible torch baseline for this operator (no hand-tuned production kernel of speculative-decode-with-per-batch-commit exists).**

## The written-kernel requirement

Both graded outputs — `o` and `state_out` — must be produced by `@triton.jit` kernels your module declares and actually launches. The 60% device-time floor is the machine-checkable half; the commit snapshot must be inside the kernel path.

## Constraints

- One H100. **No internet.** torch, triton and einops are installed.
- Do not change the semantics, signature, or output dtypes. `state0` is read-only.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped): `fla.ops`, `import fla`, `from fla`, `fused_recurrent_gated_delta_rule`, `chunk_gated_delta_rule`, `fused_recurrent_delta_rule`, `chunk_delta_rule`, `fused_recurrent`.
- Apply the scalar gate BEFORE the delta write; compute the pre-update prediction before every delta.
- Numerical policy must be uniform: no branch on input magnitude, distribution, or seed. Carry the state in float32.
- Determinism: repeated calls on one input must be bitwise identical on **both** outputs.

## What is hidden, and what is not

Disclosed thresholds:

| rule | value |
|---|---|
| bfloat16 tolerance on `o` | atol **2e-2**, rtol **3.125e-2** |
| float32 tolerance on `o` | atol **2e-3**, rtol **2e-3** |
| `state_out` tolerance | float32, atol **2e-3**, rtol **2e-3**, on every input dtype |
| output dtypes | `o` in `v`'s dtype; `state_out` in float32, always |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels |
| determinism | **3** runs on one input, bitwise identical `o` and `state_out` |
| stability | no raise, no non-finite where `reference.py` is finite (includes a 1024-step probe, an all-zero-commit probe, and an all-full-commit probe) |

## The refinement loop

`./bench.py` gates in the grader's order and then times you against the compiled baseline on a small published set of shapes. It does not tell you your score. `--json <path>` writes results machine-readably.
