# Task: implement a fast INT8-STATE delta rule decode step, forward only

`kernel.py` contains a working but deliberately slow implementation of the delta rule run in its decode regime with the recurrent state stored in **int8** at the operator boundary (per-`(batch, head)` symmetric scale). Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `int8_state_delta_decode(q, k, v, beta, state0_q, state0_scale)`. Its signature and semantics are fixed, and **all three** of its outputs are graded: the per-step outputs `o`, the int8 quantized final state `state_out_q`, and its per-`(batch, head)` float32 scale `state_out_scale`. No backward.

## The operator

For each batch and head, per step `t`:

```
S_0    = state0_q * state0_scale                              (dequantize on entry)
S_t    = S_{t-1} + k_t delta_t^T   with delta_t = (v_t - S_{t-1}^T k_t) * beta_t
o_t    = S_t^T (q_t / sqrt(K))
scale_out[b,h]  = max(|S_S[b,h]|) / 127                       (symmetric per-(b,h))
state_out_q[b,h] = round(S_S[b,h] / scale_out[b,h])           (clamped to [-127, 127])
```

Shapes: `q`/`k` are `[B, S, H, K]`, `v` is `[B, S, H, V]`, `beta` is `[B, S, H]`, `state0_q` is `[B, H, K, V]` int8, `state0_scale` is `[B, H]` float32. The call returns `(o, state_out_q, state_out_scale)` with `o` of shape `[B, S, H, V]` in `v`'s dtype, `state_out_q` of shape `[B, H, K, V]` int8, and `state_out_scale` of shape `[B, H]` float32.

Every part matters and is graded: the state is dequantized on entry with the per-`(b, h)` scale (a batch/head with `state0_scale == 0` starts from a true zero state, both `state0_q` and `state0_scale` are zero in that case — do not divide by zero); the recurrence runs in **float32**; the final state is symmetrically quantized per-`(b, h)` with `scale = max(|S|) / 127` and `q = round(S / scale)` clamped to `[-127, 127]`, and the zero-state case must round-trip to `(q = 0, scale = 0)`. `reference.py` is the definition of correctness — where this text and that code ever disagree, the code wins. Carry the state in float32 inside the operator.

The int8 boundary is roughly 4x lighter than the fp32 boundary a plain delta rule kernel pays, and it lets the caller keep a paged KV cache in 4x less HBM. That is where the headroom is: fusing the dequantize into the first step and the requantize into the last step of a persistent per-`(batch, head)` scan avoids the S-time fp32 scratch state entirely.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- **G1 — no library.** Your submission does not import or call any delta-rule library implementation, and does not delegate the quantization to `torch.quantization`, `quantize_per_tensor`, `quantize_per_channel`, `bitsandbytes`, `tensorrt`, or `torchao.quantization`. Checked by a word-boundary symbol scan.
- **G2 — correctness.** Your outputs agree with `reference.py` on a hidden shape set. `o` is compared at the input dtype's tolerance; the state is compared on the **dequantized** value (`state_q * state_scale`) at the float32 state tolerance PLUS a half-LSB slack derived from the reference's per-`(b, h)` scale; `state_scale` is compared at the same float32 tolerance. Includes non-power-of-two `B`/`S`/`H`, `S = 1`, single-sequence, and large-state shapes; `K` and `V` are always powers of two.
- **G4 — written-kernel share.** **At least 60% of graded device time runs inside Triton kernels your own module declares.** The dequantize on entry and the requantize on exit count too.

Above that gate, the score is the geometric mean across the hidden graded shapes of `(production baseline time) / (your time)`, normalized against a hidden target and saturating at 1. **The denominator is a `torch.compile` of a fused fp32 delta step with a torch-side dequantize/requantize — the strongest reproducible torch baseline for this operator (no hand-tuned production kernel of int8-boundary linear attention exists).**

## The written-kernel requirement

All three graded outputs — `o`, `state_out_q`, `state_out_scale` — must be produced by `@triton.jit` kernels your module declares and actually launches. The 60% device-time floor is the machine-checkable half; the intent is that the recurrence AND the quantization are your kernel's work.

## Constraints

- One H100. **No internet.** torch, triton and einops are installed.
- Do not change the semantics, signature, or output dtypes. `state0_q` and `state0_scale` are read-only.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped): `fla.ops`, `import fla`, `from fla`, `fused_recurrent_delta_rule`, `chunk_delta_rule`, `fused_recurrent_gated_delta_rule`, `chunk_gated_delta_rule`, `fused_recurrent`, `torch.quantization`, `quantize_per_tensor`, `quantize_per_channel`, `bitsandbytes`, `tensorrt`, `torchao.quantization`.
- The interior of the recurrence must run in **float32**. A bfloat16 or float16 scratch state accumulates ~1e-3 relative error per step and crosses the state tolerance at long horizons.
- The final quantization must be per-`(batch, head)` symmetric with `scale = max(|S|) / 127`; zero-state must round-trip to `(q = 0, scale = 0)` without a divide-by-zero.
- Numerical policy must be uniform. Determinism: repeated calls on one input must be bitwise identical on **all three** outputs.

## What is hidden, and what is not

Disclosed thresholds:

| rule | value |
|---|---|
| bfloat16 tolerance on `o` | atol **2e-2**, rtol **3.125e-2** |
| float32 tolerance on `o` | atol **2e-3**, rtol **2e-3** |
| dequantized-state tolerance | atol **2e-3** + `state0_scale[b,h] / 2`, rtol **2e-3**, per-`(b, h)`, on every input dtype |
| `state_scale` tolerance | float32, atol **2e-3**, rtol **2e-3** |
| output dtypes | `o` in `v`'s dtype; `state_out_q` int8; `state_out_scale` float32, always |
| written-kernel gate | at least **60%** of graded device time in `@triton.jit` kernels |
| determinism | **3** runs on one input, bitwise identical `o`, `state_out_q`, `state_out_scale` |
| stability | no raise, no non-finite where `reference.py` is finite (includes a 1024-step probe, a beta-zero probe, and a state-zero probe) |

## The refinement loop

`./bench.py` gates in the grader's order and then times you against the compiled baseline on a small published set of shapes. It does not tell you your score. `--json <path>` writes results machine-readably.
