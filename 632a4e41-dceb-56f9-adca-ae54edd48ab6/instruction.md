# Task: implement a fast GROUPED-QUERY DELTA RULE decode step, forward only

`kernel.py` contains a working but deliberately slow implementation of the delta rule run in its decode regime under a grouped-query topology: `Hq` query heads share `Hkv` state heads with a group ratio `G = Hq / Hkv`. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `gqa_delta_decode(q, k, v, beta, state0)`. Its signature and semantics are fixed, and **both** of its outputs are graded: the per-step outputs `o` and the final state `state_out`. There is no backward pass: this operator is graded forward-only.

## The operator

For each batch and kv-head `hk` with state `S[hk]` of shape `[K, V]` starting from `state0[b, hk]`, per step `t`:

```
pred_t  = S[hk]_{t-1}^T k_t
delta_t = (v_t - pred_t) * beta_t
S[hk]_t = S[hk]_{t-1} + k_t delta_t^T
o_t[hq] = S[hk]_t^T (q_t[hq] / sqrt(K))       for every hq in the group of hk
```

Shapes: `q` is `[B, S, Hq, K]`, `k` is `[B, S, Hkv, K]`, `v` is `[B, S, Hkv, V]`, `beta` is `[B, S, Hkv]`, `state0` is `[B, Hkv, K, V]` float32. The call returns `(o, state_out)` with `o` of shape `[B, S, Hq, V]` in `v`'s dtype and `state_out` of shape `[B, Hkv, K, V]` in **float32**. `Hq` is a multiple of `Hkv`; the group ratio is `G = Hq / Hkv`.

Every part matters and is graded: the state is per-kv-head; every query head in the group of `hk` reads the SAME state after the delta write; the pre-update prediction is the state before this step's delta; `q` is scaled by `K^{-0.5}`. `reference.py` writes this out as a sequential float32 scan and is the definition of correctness — where this text and that code ever disagree, the code wins. Carry the state in float32.

The state SHARING across `G` query heads is the source of the extra structure. An ordinary per-query-head kernel either wastes `G x Hkv` state slots (memory-quadratic in `G`) or races `G` query heads writing to the same state without a designed accumulation order. A fast solution performs ONE delta write per (batch, kv-head, step) and lets all `G` query-head reads consume the same state tile from registers/shared memory before the next step.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- **G1 — no library.** Your submission does not import or call any delta-rule library implementation (fla or otherwise). Checked by a word-boundary symbol scan of your comment-stripped source (banned names under *Constraints*; string literals and docstrings count, `#` comments do not).
- **G2 — correctness.** Your output — `o` **and** `state_out` — agrees with `reference.py` within the tolerances below on a hidden shape set larger than, and different from, the shapes `bench.py` lets you measure. The hidden set includes non-power-of-two `B`/`S`/`Hq`/`Hkv` (keeping `Hq % Hkv == 0`), the single-step `S = 1` case, a single-sequence case, and a large-`G` shape; `K` and `V` are always powers of two.
- **G4 — written-kernel share.** **At least 60% of graded device time runs inside Triton kernels your own module declares.**

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production baseline time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is a `torch.compile` of a fused delta step plus a broadcast query read across the group — the strongest reproducible torch baseline for this operator (no hand-tuned production kernel of the GQA-topology delta-rule decode exists).** Matching the slow starter scores near zero; reaching the compiled baseline's speed scores 1.

The magnitude of `v` and `state0` is drawn at random from a hidden distribution on every invocation, and the tensors that are timed are the same tensors that are checked.

## The written-kernel requirement

Both graded outputs — `o` and `state_out` — must be produced by `@triton.jit` kernels your module declares and actually launches. The 60% device-time floor is the machine-checkable half; the intent is that the recurrence AND the grouped read are your kernel's work.

## Constraints

- One H100. **No internet.** torch, triton and einops are installed.
- Do not change the semantics, signature, or output dtypes of `gqa_delta_decode`. `state0` is read-only.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped): `fla.ops`, `import fla`, `from fla`, `fused_recurrent_delta_rule`, `chunk_delta_rule`, `fused_recurrent_gated_delta_rule`, `fused_recurrent`.
- Compute the pre-update prediction before every delta write. No step may be skipped, truncated, or approximated away.
- Numerical policy must be uniform: no branch on input magnitude, distribution, or seed that changes which algorithm runs. Carry the state in float32.
- Determinism: repeated calls on one input must be bitwise identical on **both** outputs — no atomics on the state or output path, no scheduling-dependent reduction order across the `G` query heads.

## What is hidden, and what is not

The rules above are complete. Hidden from you are: the exact graded shape set (larger than and disjoint from `bench.py`'s published shapes, and including non-power-of-two `B`/`S`/`Hq`/`Hkv`, `S = 1`, `B = Hkv = 1`, and a large-`G` shape), the per-invocation magnitude of `v`/`state0`, and the target the geometric-mean fraction is normalized against. Disclosed thresholds:

| rule | value |
|---|---|
| bfloat16 tolerance on `o` | atol **2e-2**, rtol **3.125e-2** |
| float32 tolerance on `o` | atol **2e-3**, rtol **2e-3** |
| `state_out` tolerance | float32, atol **2e-3**, rtol **2e-3**, on every input dtype |
| output dtypes | `o` in `v`'s dtype; `state_out` in float32, always |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| determinism | **3** runs on one input, bitwise identical `o` and `state_out` |
| stability | no raise, no non-finite where `reference.py` is finite (includes a 1024-step probe, a large-group probe, and a beta-full probe) |

## The refinement loop

`./bench.py` gates in the grader's order — forbidden-name scan, correctness on both outputs, three-run bitwise determinism, written-kernel share — and only then times you against the compiled baseline on a small published set of shapes. It does not tell you your score. `--json <path>` writes the same results machine-readably.
