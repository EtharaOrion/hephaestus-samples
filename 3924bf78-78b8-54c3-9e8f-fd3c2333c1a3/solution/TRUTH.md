# truth.md - Golden Solve Path

## Task: hephaestus `moe_fused_experts_int4_wq_swiglu_moe_fwdbwd` (anchor `torch._grouped_mm`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `int4_wq_swiglu_moe` | **Editable:** `kernel.py`
- **Objective:** implement int4_wq_swiglu_moe in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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
the int4 weight-only quantized (W4A16) SwiGLU expert with per-output-channel
ASYMMETRIC (scale + zero-point) dequant:

    int4_wq_swiglu_moe(x, w1_pack, w1_s, w1_zp,
                          w2_pack, w2_s, w2_zp, topk_idx, topk_w) -> y

    x:        [T, D]        tokens (float32 or bfloat16)
    w1_pack:  [E, D, F]     uint8; two int4 nibbles per byte along the LAST dim,
                            unpacked width is 2F (low nibble = column 2j,
                            high nibble = column 2j+1); nibble values in [0, 15]
    w1_s:     [E, 2F]       per-output-channel scale, `x.dtype`
    w1_zp:    [E, 2F]       per-output-channel zero-point, int8 in [0, 15]
    w2_pack:  [E, F, D/2]   uint8, same nibble convention, unpacked width D
    w2_s:     [E, D]        per-output-channel scale, `x.dtype`
    w2_zp:    [E, D]        per-output-channel zero-point, int8 in [0, 15]
    topk_idx: [T, A]        int64, no gradient
    topk_w:   [T, A]        `x.dtype`, normalized
    y:        [T, D]        `x.dtype`

`2F` (i.e. `w1_pack.shape[-1] * 2`) and `D` (i.e. `w2_pack.shape[-1] * 2`) must
both be even. Semantics, exactly:

    w1_int[e] = unpack_nibbles(w1_pack[e])                      # [D, 2F], int in [0, 15]
    w1_dq[e]  = (w1_int[e] - w1_zp[e][None, :]) * w1_s[e][None, :]
    w2_int[e] = unpack_nibbles(w2_pack[e])                      # [F, D]
    w2_dq[e]  = (w2_int[e] - w2_zp[e][None, :]) * w2_s[e][None, :]
    y[t] = sum_a topk_w[t, a] * ((silu(g)*u) @ w2_dq[e]);   e = topk_idx[t, a]
    g, u  = (h @ w1_dq[e]) split into first F and last F columns

Arithmetic in float32; final cast back. Graded gradients: dx and dtopk_w
only -- packed weight buffers, scales and zero-points carry no gradient.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes int4_wq_swiglu_moe
  - establishes: int4_wq_swiglu_moe is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: int4_wq_swiglu_moe agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.66 |
| starter fraction of anchor (measured) | 0.101543 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| bfloat16 tolerance | atol 6.0, rtol 0.1 |
| float32 tolerance | atol 2.0, rtol 0.02 |
| comparison | within dtype tolerance |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes int4_wq_swiglu_moe. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: int4 weight-only quantized (W4A16) SwiGLU MoE, forward + backward.

HARD member of the moe_fused_experts family. Hardness lever: LOW-PRECISION
EXPERT WEIGHTS via int4 W4A16 with per-output-channel ASYMMETRIC (scale +
zero-point) dequant. Distinct from the fp8_e4m3_swiglu_moe sibling: int4 has
NO native tensor-core representation, so a fast kernel must UNPACK the packed
nibbles (2 int4 per uint8) and SUBTRACT the zero-point before it can feed a
tensor-core dtype (bf16 or fp16). Fusing unpack + zero-point subtraction +
scale multiply into the MMA prologue is the win over the anchor's eager
dequant-then-bf16-GEMM path.

Only x and topk_w are differentiable (`grad_names = ("x", "topk_w")`); packed
weights, scales and zero-points are inference-shaped inputs.

STATUS: authored-draft. TOL/OUT_TOL/TARGET_FRACTION_OF_SOTA are AUTHORED
PLACEHOLDERS; MUST be re-measured on the delivery GPU (int4 asymmetric quant
noise dominates the output tolerance floor, calibrate against the oracle's
own dequant-then-bf16 GEMM).
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `f4a5d65b32239300` |
| `kernel.py` | `a4331f77a96da009` |
| `bench.py` | `cc78ee8504b44f8c` |
| `instruction.md` | `33e3e6936d0c8ff0` |
| `hidden_shapes.json` | `0946d29dd72b9a8f` |
| `solution/oracle_kernel.py` | `b27a666642a02410` |

- **seed_base:** `132241`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | T=8192 D=2048 F=1408 E=64 A=4 | bfloat16, float32 |
| `g2` | T=4096 D=2048 F=1024 E=128 A=8 | bfloat16, float32 |
| `g3` | T=16384 D=1024 F=2816 E=32 A=2 | bfloat16, float32 |

- **forbidden symbols (13):** `torch._grouped_mm`, `_grouped_mm`, `grouped_mm`, `fused_moe`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `cutlass`, `flashinfer`, `bitsandbytes`, `auto_gptq`, `awq`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 044eda056a577a65c7220299a73d3149fe07aa5f3b3fcf7a93b8490da37d53ba
FORGE-CANARY-SLOT-1 48fe8ab47273a9b68530219458d2dabc65ac7bbd7bdb6fff6c47caec9cb76d56
FORGE-CANARY-SLOT-2 1444827bb5ef43fcd0c4d39153007a9805bb7482d970cce6982b74b42823a7c7
FORGE-CANARY-SLOT-3 db7c789d5a08da4d80b752723905259a7ff80f7168556ba2cda874a620216e39
FORGE-CANARY-END
