# Task: implement a fast PER-CHANNEL-DECAY + BETA-DELTA linear-attention decode step, forward only

`kernel.py` contains a working but deliberately slow implementation of a hybrid decode-regime linear-attention step that fuses two per-step levers no single existing library kernel exposes together: a PER-CHANNEL log-forget gate on the key axis of the `[K, V]` state AND a per-step scalar write-strength `beta` on a delta-corrected write. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `rwkv7_channel_delta_decode(q, k, v, g, beta, state0)`. Its signature and semantics are fixed, and **both** of its outputs are graded: the per-step outputs `o` and the final state `state_out`. No backward.

## The operator

For each batch and head, per step `t`, with state `S` of shape `[K, V]`:

```
S = diag(exp(g_t)) . S                     # per-channel decay on the key rows
pred = S^T k_t                              # pre-write read
delta = (v_t - pred) * beta_t               # scalar write strength
S = S + k_t delta^T                         # delta write
o_t = S^T (q_t / sqrt(K))                   # post-write read
```

Shapes: `q`/`k` are `[B, S, H, K]`, `v` is `[B, S, H, V]`, `g` is `[B, S, H, K]` float32 (per-channel log-gate, `g <= 0`), `beta` is `[B, S, H]`, `state0` is `[B, H, K, V]` float32. The call returns `(o, state_out)` with `o` of shape `[B, S, H, V]` in `v`'s dtype and `state_out` of shape `[B, H, K, V]` in **float32**.

Every part matters and is graded: the per-channel decay is applied to each **row** of the state **before** the delta prediction; the delta correction subtracts the pre-write prediction; the read is against the state AFTER the delta write. `reference.py` writes this out as a sequential float32 scan and is the definition of correctness — where this text and that code ever disagree, the code wins. Carry the state in float32.

`beta = 0` degenerates the operator to plain gated linear attention (GLA); `g = 0` degenerates it to the plain delta rule. Neither of those library kernels handles both levers at once, and that is where the headroom is — the pre-write prediction wants to share reduction work with the post-write output, and both want to consume the same state tile that is only decayed once.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- **G1 — no library.** Your submission does not import or call any linear-attention library implementation. Checked by a word-boundary symbol scan of your comment-stripped source (banned names under *Constraints*).
- **G2 — correctness.** Your output — `o` **and** `state_out` — agrees with `reference.py` within the tolerances below on a hidden shape set larger than, and different from, the shapes `bench.py` lets you measure. Includes non-power-of-two `B`/`S`/`H`, `S = 1`, single-sequence, and large-state shapes; `K` and `V` are always powers of two.
- **G4 — written-kernel share.** **At least 60% of graded device time runs inside Triton kernels your own module declares.**

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production baseline time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is a `torch.compile` of a fused hybrid step (per-channel decay + delta write + post-write read) in a decode loop — the strongest reproducible torch baseline for this operator (no hand-tuned production kernel of the per-channel-gate + beta-delta combination exists).**

## The written-kernel requirement

Both graded outputs — `o` and `state_out` — must be produced by `@triton.jit` kernels your module declares and actually launches.

## Constraints

- One H100. **No internet.** torch, triton and einops are installed.
- Do not change the semantics, signature, or output dtypes. `state0` is read-only.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped): `fla.ops`, `import fla`, `from fla`, `fused_recurrent_gla`, `chunk_gla`, `fused_recurrent_gated_delta_rule`, `chunk_gated_delta_rule`, `fused_recurrent_delta_rule`, `chunk_delta_rule`, `fused_recurrent_rwkv7`, `chunk_rwkv7`, `fused_recurrent`.
- Apply the per-channel decay to each row of the state BEFORE the pre-write prediction; compute the read AFTER the delta write. No step may be skipped, truncated, or approximated away.
- Numerical policy must be uniform. Carry the state in float32.
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
| stability | no raise, no non-finite where `reference.py` is finite (includes a 1024-step probe, a beta-zero probe, and a split-gate probe) |

## The refinement loop

`./bench.py` gates in the grader's order and then times you against the compiled baseline on a small published set of shapes. It does not tell you your score. `--json <path>` writes results machine-readably.
