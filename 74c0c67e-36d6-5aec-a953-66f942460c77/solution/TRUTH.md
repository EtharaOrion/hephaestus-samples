# truth.md - Golden Solve Path

## Task: hephaestus `moe_fused_experts_fp8_e4m3_swiglu_moe_fwdbwd` (anchor `torch._grouped_mm`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `fp8_e4m3_swiglu_moe` | **Editable:** `kernel.py`
- **Objective:** implement fp8_e4m3_swiglu_moe in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
- **Graded outcome:** a float in [0, 1] written to the verifier's score.json

## Section 1: Scope

The agent sees exactly 4 files: `instruction.md`, `kernel.py`, `reference.py`, `bench.py`.
`kernel.py` is the only editable path; `reference.py` is the correctness
oracle and is read-only. The graded shapes, scales and value distributions are hidden.

The solver gets 50 attempts within 6.0 hours on nvidia/cuda:12.8.0-devel-ubuntu24.04@sha256:813c3b834aaafe2df2891f9a840c412a4df3c073dd9e0ac75aab810ed82a5b48 with network `no-network`.

## Section 2: Canonical Solve Path

The operator this path must reproduce, as `reference.py` states it:

```text
Reference implementation -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY.

Family: moe_fused_experts (precomputed top-k routing, fwd+bwd). This member is
the fp8-e4m3 weight-only quantized SwiGLU expert:

    fp8_e4m3_swiglu_moe(x, w1_q, w1_s, w2_q, w2_s, topk_idx, topk_w) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    w1_q:     [E, D, 2F] "quantized" gate/up weights, stored in `x.dtype`
                         but with values snapped to the fp8-e4m3fn grid
    w1_s:     [E, 2F]    per-output-channel scale, `x.dtype`
    w2_q:     [E, F, D]  quantized down weights, `x.dtype`, on fp8-e4m3 grid
    w2_s:     [E, D]     per-output-channel scale, `x.dtype`
    topk_idx: [T, A]     int64, no gradient
    topk_w:   [T, A]     routing weights, `x.dtype`, normalized
    y:        [T, D]     `x.dtype`

Semantics, exactly:

    w1_dq[e] = w1_q[e] * w1_s[e][None, :]        (elementwise per output channel)
    w2_dq[e] = w2_q[e] * w2_s[e][None, :]
    y[t] = sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])          (a ascending)
    f(e, h) = (silu(h @ w1_dq[e][:, :F]) * (h @ w1_dq[e][:, F:])) @ w2_dq[e]
    silu(z) = z * sigmoid(z)

Arithmetic is accumulated in float32; only the final y is cast back. Graded
gradients: `dx` and `dtopk_w` only -- quantized weight buffers `w1_q`, `w2_q`
and their scales `w1_s`, `w2_s` are inference-shaped inputs and carry no
gradient. `topk_idx` is integer and has no gradient. The hardness lever is
that a fast kernel fuses the per-out-channel dequant into the MMA prologue
of each expert GEMM (equivalently: uses fp8-e4m3 tensor cores directly on
the pre-snapped weights). A correctness-preserving bf16 GEMM on the same
values is legal but slow.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes fp8_e4m3_swiglu_moe
  - establishes: fp8_e4m3_swiglu_moe is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: fp8_e4m3_swiglu_moe agrees with reference.py and is the baseline to beat
  - survives: the published shapes in bench.py are smaller than and different from the graded shapes

**Step 3 (G1).** read the production anchor to understand the fast formulation, without importing or calling it
  - establishes: the decomposition that makes the operator fast is understood
  - survives: a forbidden-symbol scan over the submitted source with comments stripped, so citing a symbol is legal and calling it is not

**Step 4 (G2).** write the kernel and iterate against bench.py, holding the reference semantics fixed
  - establishes: the output and every input gradient still agrees with reference.py on the hidden configurations
  - survives: correctness on the graded draw, which is the same draw that is timed

**Step 5 (G4).** confirm the timed work runs in the candidate's own kernels before claiming the result
  - establishes: device time is dominated by kernels the candidate wrote
  - survives: the written-kernel adoption floor

**Step 6 (G3).** time the candidate against the anchor on the graded shapes under a locked clock
  - establishes: the geometric mean of the per-shape ratios is the score
  - survives: a candidate that is fast only on the shapes it could see

## Section 3: Value Lock

| Quantity | Value |
| --- | --- |
| metric | fraction_of_sota |
| target fraction of anchor | 0.68 |
| starter fraction of anchor (measured) | 0.077565 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| bfloat16 tolerance | atol 6.0, rtol 0.1 |
| float32 tolerance | atol 2.0, rtol 0.02 |
| comparison | within dtype tolerance |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes fp8_e4m3_swiglu_moe. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
| `G1` forbidden_symbols | the submission neither imports nor calls the production kernel or any equivalent library implementation, scanned over the source with comments stripped so that citing a symbol stays legal and calling it does not. |
| `G2` correctness | the output and all input gradients agrees with reference.py on every hidden configuration, on the same randomly scaled draw that is timed, so no branch can tell the checked distribution from the timed one. |
| `G3` speed | the candidate is timed against the anchor on the graded shapes under a locked clock, and the geometric mean of the per-shape ratios becomes the score. |
| `G4` written_kernel | the share of device time spent in the candidate's OWN kernels clears the adoption floor, so a submission that delegates the work elsewhere cannot claim the result. |
| rubric judge | a cross-family LLM reads `tests/rubrics.jsonl` against the trajectory. All 12 gate. |

### Checker reconciliation

Every checker the bundle commits is named by at least one step above:
  - `G0`, `G1`, `G2`, `G3`, `G4`

## Section 5: Known Failure Modes

- **NC1 unmodified_starter** - rejected by the metric; expect score well below target; the starter is the deliberately sub-optimal implementation
- **NC2 library_call** - submission imports or calls the production kernel - rejected by G1; expect score exactly 0.0
- **NC3 fast_and_wrong** - accumulation precision dropped to buy speed - rejected by G2; expect score exactly 0.0
- **NC4 delegated_work** - the timed region runs in someone else's kernels - rejected by G4; expect score exactly 0.0

Design rationale carried from `taskdef.py`:

```text
Taskdef: fp8-e4m3 weight-only quantized SwiGLU MoE, forward + backward.

HARD member of the moe_fused_experts family (precomputed integer routing). The
hardness lever is LOW-PRECISION EXPERT WEIGHTS: the routed expert weights
w1/w2 are stored on the fp8-e4m3fn grid with per-out-channel bf16 scales, and
a fast kernel MUST fuse the dequant (or, equivalently, use fp8-e4m3 tensor
cores directly on the pre-snapped weights) into the MMA prologue. A candidate
that dequantizes eagerly to bf16 and runs a standard bf16 GEMM is correct but
pays for the dequant traffic; the winning kernel keeps the weights in fp8 and
consumes them from tensor cores.

Only x and topk_w are differentiable (`grad_names = ("x", "topk_w")`); the
quantized weight buffers are inference-shaped inputs and carry no gradient.

STATUS: authored-draft. TOL/OUT_TOL/TARGET_FRACTION_OF_SOTA are AUTHORED
PLACEHOLDERS; MUST be re-measured on the delivery GPU (fp8 quant noise
dominates the output tolerance floor, calibrate against the oracle's own
dequant-then-bf16 GEMM).

Operator (reference.py is the definition):
    fp8_e4m3_swiglu_moe(x[T,D], w1_q[E,D,2F], w1_s[E,2F], w2_q[E,F,D],
        w2_s[E,D], topk_idx[T,A] i64, topk_w[T,A]) -> y[T,D]
    w1_dq[e] = w1_q[e] * w1_s[e][None,:];  w2_dq[e] = w2_q[e] * w2_s[e][None,:]
    y[t] = sum_a topk_w[t,a] * ((silu(g)*u) @ w2_dq[e]),
    g,u = h @ w1_dq[e]  split.
    SURFACE fwdbwd: gradients of x AND topk_w graded (two).

Anchor: cuBLAS grouped-GEMM composition on expert-sorted tokens with an
eager per-expert dequant pass -- torch._grouped_mm consuming bf16-dequantized
weights, silu*up in eager, weighted index_add combine, fwd via autograd (the
dequant pass and the GEMM run as two device calls, exactly what the fp8
prologue-fusion kernel elides).
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `115ba5c3d2f45ae0` |
| `kernel.py` | `7e5ecb6073c9ecea` |
| `bench.py` | `b5748229fce52d96` |
| `instruction.md` | `3fc1fa73a0a48d19` |
| `hidden_shapes.json` | `796433cf1bdad2df` |
| `solution/oracle_kernel.py` | `b19414e56b4483ea` |

- **seed_base:** `118973`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | T=8192 D=2048 F=1408 E=64 A=4 | bfloat16, float32 |
| `g2` | T=4096 D=2048 F=1024 E=128 A=8 | bfloat16, float32 |
| `g3` | T=16384 D=1024 F=2816 E=32 A=2 | bfloat16, float32 |

- **forbidden symbols (10):** `torch._grouped_mm`, `_grouped_mm`, `grouped_mm`, `fused_moe`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `cutlass`, `flashinfer`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 ecf5206580e37be68bc54a69b3ad246bad51767f4cc933c8777403fa864ef951
FORGE-CANARY-SLOT-1 b6e9982ed2ead637d94fb4c2a78ffdc6d891542b2ebfaeeaf60aabe932ddb3c6
FORGE-CANARY-SLOT-2 5211f21413770f59bf72e5dbfb4cd79bc7d06b47f1ab9d1ca507ed3dc4e73d62
FORGE-CANARY-SLOT-3 671305a7b95a9d8f290e21eefb64b67ad48046e4587d802513cc93e2d3cc4b15
FORGE-CANARY-END
