# truth.md - Golden Solve Path

## Task: hephaestus `moe_fused_experts_wide_expert_swiglu_moe_grouped_fwdbwd` (anchor `torch._grouped_mm`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `wide_expert_swiglu_moe` | **Editable:** `kernel.py`
- **Objective:** implement wide_expert_swiglu_moe in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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
the SwiGLU expert operator (silu-gated GLU) with the same math as swiglu_moe
but a shape distribution that pushes two hardness levers at once: (a) variable-
length grouped GEMM over an expert-major permutation with E up to 256 experts
and strongly imbalanced token counts per expert (many short/empty segments,
few long ones), and (b) wide-vs-tall aspect where the FFN intermediate F is
substantially larger than the model dim D (F ~= 2*D..3*D). The operator math
is identical to the family baseline; the reference below is the same
sequential (slot, expert)-loop oracle.

    wide_expert_swiglu_moe(x, w1, w2, topk_idx, topk_w) -> y

    x:        [T, D]
    w1:       [E, D, 2F]  gate = [:, :, :F], up = [:, :, F:]
    w2:       [E, F, D]
    topk_idx: [T, A] int64, values in [0, E); no gradient
    topk_w:   [T, A] same dtype as x, already normalized
    y:        [T, D] same dtype as x

    y[t] = sum_a topk_w[t, a] * f(topk_idx[t, a], x[t]),  a ascending;
    f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e];
    silu(z) = z * sigmoid(z).

Graded gradients: dx, dw1, dw2, dtopk_w (four). Experts no token routes to
receive exactly zero weight gradients. Combine order fixed (ascending slot).
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes wide_expert_swiglu_moe
  - establishes: wide_expert_swiglu_moe is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: wide_expert_swiglu_moe agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.72 |
| starter fraction of anchor (measured) | 0.129384 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| bfloat16 tolerance | atol 20.0, rtol 0.1 |
| float32 tolerance | atol 6.0, rtol 0.02 |
| comparison | within dtype tolerance |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes wide_expert_swiglu_moe. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: SwiGLU MoE for wide-FFN, high-E shape distributions.

HARD member of the moe_fused_experts family (precomputed integer routing).
Operator math is identical to the family baseline (silu-gated GLU experts);
the hardness lever is a shape distribution that pushes TWO KernelBench-hard
axes at once: (a) variable-length grouped GEMM with E up to 256 experts and
strongly imbalanced token counts per expert (strong zipf, many
short/empty segments, few long ones), and (b) wide-vs-tall aspect where the
FFN intermediate F is 2--3x the model dim D. A candidate that specializes to
a small, balanced expert set (as swiglu_moe or geglu_moe permit) fails here:
segment-tile schedulers have to handle 200+ segments with sub-tile counts,
and the wide FFN inflates the intermediate act[T*A_eff, F] to the point that
a naive materialization is memory-bound.

STATUS: authored-draft. TOL/OUT_TOL/TARGET_FRACTION_OF_SOTA are AUTHORED
PLACEHOLDERS carried from moe_swiglu, WIDENED for the longer per-expert
segments possible at E=256 with strong zipf (one expert can hold the
majority of T*A slots), and MUST be re-measured on the delivery GPU.

Operator (reference.py is the definition):
    wide_expert_swiglu_moe(x[T,D], w1[E,D,2F], w2[E,F,D], topk_idx[T,A] i64,
        topk_w[T,A]) -> y[T,D]
    y[t] = sum_a topk_w[t,a] * f(topk_idx[t,a], x[t]);
    f(e,h) = (silu(h@w1[e,:,:F]) * (h@w1[e,:,F:])) @ w2[e].
    SURFACE fwdbwd: gradients of x, w1, w2 AND topk_w (four).

Anchor: cuBLAS grouped-GEMM composition on expert-sorted tokens (identical
to swiglu_moe's anchor); the wide-vs-tall aspect and E=256 stress the
grouped GEMM scheduler and the intermediate-materialization traffic.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `0a68f4e8a9bd2b26` |
| `kernel.py` | `53bbd8609685b19c` |
| `bench.py` | `92a6e7d5cea36ab6` |
| `instruction.md` | `d1e2fdc435f93245` |
| `hidden_shapes.json` | `f4f9c74c6832f34f` |
| `solution/oracle_kernel.py` | `18d757ab70429956` |

- **seed_base:** `105397`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | T=8192 D=1024 F=2816 E=128 A=4 | bfloat16, float32 |
| `g2` | T=4096 D=1024 F=3072 E=256 A=4 | bfloat16, float32 |
| `g3` | T=16384 D=768 F=2048 E=64 A=2 | bfloat16, float32 |

- **forbidden symbols (10):** `torch._grouped_mm`, `_grouped_mm`, `grouped_mm`, `fused_moe`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `cutlass`, `flashinfer`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 ee63dc6e01c821ba229fe54a1b17b536b4e12f77393482fb462da30ada6c7758
FORGE-CANARY-SLOT-1 6cc28039e2c7ac13f582035bcb25468f68dca60367d553d735b2c4eb3d6f5954
FORGE-CANARY-SLOT-2 2f82a766307e31a9f5ea6337a11f15c1db255764d7ede76ca7b45f9b5e461f75
FORGE-CANARY-SLOT-3 4c8cd138e01315ef2dec00edbeedc16e701d4e642b5d364680b922189959c5dc
FORGE-CANARY-END
