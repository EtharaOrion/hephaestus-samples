# Task: implement a fast PAGED GATED LINEAR ATTENTION decode step, forward only

`kernel.py` contains a working but deliberately slow implementation of gated linear attention (GLA) run in its decode regime — one token per step from a per-batch initial state that lives in a shared paged pool. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only. Do not edit or bypass anything under the verifier mount.

The entry point is `paged_gla_decode(q, k, v, g, page_table, paged_state)`. Its signature and semantics are fixed, and **both** of its outputs are graded: the per-step outputs `o` and the dense per-batch final state `state_out`. There is no backward pass: this operator is graded forward-only.

## The operator

For each batch `b` and head, with state `S` of shape `[K, V]` starting from `paged_state[page_table[b]]`:

```
S_t = diag(exp(g_t)) . S_{t-1} + k_t v_t^T
o_t = S_t^T (q_t / sqrt(K))
```

Shapes: `q` and `k` are `[B, S, H, K]` (queries and keys), `v` is `[B, S, H, V]`, `g` is `[B, S, H, K]` float32 (per-channel log-forget gate, `g <= 0`), `page_table` is `[B]` int32 (the physical page id each batch owns for this call), `paged_state` is `[P, H, K, V]` float32 (the shared pool, `P >= B`, **read-only** — you must never mutate it). The call returns `(o, state_out)` with `o` of shape `[B, S, H, V]` in `v`'s dtype and `state_out` of shape `[B, H, K, V]` in **float32** — a dense per-batch tensor of the states after the complete last step.

Every part matters: `q` is scaled by `K^{-0.5}` in the read; the per-channel gate is applied to each **row** of the state **before** the write; the write is the plain outer product `k v^T` (no delta correction); the recurrence starts from the page each batch owns, not from zero, and `state_out` is the state after the **complete** last step. `reference.py` writes this out as a sequential float32 scan and is the definition of correctness — where this text and that code ever disagree, the code wins. Carry the state in float32.

The `page_table` is drawn as a random permutation of distinct page ids on every graded call, so the per-batch state gather is non-contiguous by design. A kernel that assumes a stride-1 per-batch state address will be wrong on every graded input.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- **G1 — no library.** Your submission does not import or call any linear-attention library implementation of this operator, and does not delegate to `paged_attention` / `vllm.attention` / `flash_attn_with_kvcache` either. Reading library source is allowed; calling or shipping it is not. Checked by a word-boundary symbol scan of your comment-stripped source (banned names under *Constraints*; string literals and docstrings count, `#` comments do not).
- **G2 — correctness.** Your output — `o` **and** `state_out` — agrees with `reference.py` within the tolerances below on a hidden shape set larger than, and different from, the shapes `bench.py` lets you measure. The hidden set includes non-power-of-two batch/sequence/head counts, the single-step `S = 1` case, a single-sequence case, and a large-state shape; `K` and `V` are always powers of two.
- **G4 — written-kernel share.** **At least 60% of graded device time runs inside Triton kernels your own module declares.** Measured after the fact by device-time attribution. Delegating the recurrence to framework operators scores zero here.

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production baseline time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is a `torch.compile` of a fused GLA step run in a dense decode loop after materializing the initial state from the paged pool via fancy-index — the strongest reproducible torch baseline for this operator (no hand-tuned production kernel of the paged-recurrent variant exists).** Matching the slow starter scores near zero; reaching the compiled baseline's speed scores 1.

The magnitude of `v` and `paged_state` is drawn at random from a hidden distribution on every invocation, and the tensors that are timed are the same tensors that are checked. A branch that detects the checker's input statistics and uses cheaper math elsewhere gains nothing here, and is a rubric violation besides.

## The written-kernel requirement

Both graded outputs — `o` and `state_out` — must be produced by `@triton.jit` kernels your module declares and actually launches. The 60% device-time floor is the machine-checkable half; the intent is that the recurrence itself is your kernel's work, INCLUDING the page-table gather (a fast solution gathers one (batch, head) state tile per program directly from the pool without ever materializing a dense scratch state).

## Constraints

- One H100. **No internet.** torch, triton and einops are installed; `pip install`, `git clone` and downloads will fail.
- Do not change the semantics, signature, or output dtypes of `paged_gla_decode`. `paged_state` is read-only — never mutate a caller tensor.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped, strings and docstrings count): `fla.ops`, `import fla`, `from fla`, `fused_recurrent_gla`, `chunk_gla`, `fused_recurrent`, `paged_attention`, `vllm.attention`, `flash_attn_with_kvcache`. The recurrence is the task; calling a library's implementation of it is not a solution.
- Compute the per-channel decay as `exp(g)` applied to each **row** of the state before the outer-product write. No step may be skipped, truncated, or approximated away.
- Numerical policy must be uniform: no branch on input magnitude, distribution, or seed that changes which algorithm runs. Carry the state in float32 from `paged_state[page_table[b]]` to `state_out`.
- Determinism: repeated calls on one input must be bitwise identical on **both** outputs — no atomics on the state or output path, no scheduling-dependent reduction order.

## What is hidden, and what is not

The rules above are complete. Hidden from you are: the exact graded shape set (larger than and disjoint from `bench.py`'s published shapes, and including non-power-of-two `B`/`S`/`H`, `S = 1`, `B = H = 1`, and a large-state shape), the per-invocation magnitude of `v`/`paged_state`, the graded `page_table` permutation, and the target the geometric-mean fraction is normalized against. Disclosed thresholds:

| rule | value |
|---|---|
| bfloat16 tolerance on `o` | atol **2e-2**, rtol **3.125e-2** |
| float32 tolerance on `o` | atol **2e-3**, rtol **2e-3** |
| `state_out` tolerance | float32, atol **2e-3**, rtol **2e-3**, on every input dtype |
| output dtypes | `o` in `v`'s dtype; `state_out` in float32, always |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels your module declares |
| determinism | **3** runs on one input, bitwise identical `o` and `state_out` |
| stability | no raise, no non-finite where `reference.py` is finite (includes a 1024-step probe, a reversed-page-table probe, and a saturating-gate probe) |

## The refinement loop

`./bench.py` gates in the grader's order — forbidden-name scan, correctness on both outputs (including a non-power-of-two shape and `S = 1`), three-run bitwise determinism, written-kernel share — and only then times you against the compiled baseline on a small published set of shapes. It does not tell you your score: the published shapes are not the graded shapes, the graded set randomizes hidden magnitudes and page permutations, and `bench.py` does not run the 1024-step stability probe. `--json <path>` writes the same results machine-readably. Steer with it, then resubmit; you may iterate up to the session cap, and the best-scoring submission is the one that counts.
