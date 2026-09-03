# truth.md - Golden Solve Path

## Task: hephaestus `chunked_linear_attn_chunked_gated_delta_fwdbwd` (anchor `fla.ops.gated_delta_rule.chunk_gated_delta_rule`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `chunked_gated_delta` | **Editable:** `kernel.py`
- **Objective:** implement chunked_gated_delta in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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
DO NOT MODIFY. This is the oracle the benchmark harness checks every candidate against.

The operator is the chunk-parallel GATED delta rule, the recurrence at the core of
Gated DeltaNet and Kimi Delta Attention. For a single head, with state S in R^{K x V}:

    S'_t = diag(exp(g_t)) . S_{t-1}
    S_t  = S'_t + k_t (v_t - S'_t^T k_t)^T beta_t
    o_t  = S_t^T (q_t / sqrt(K))

The decay is applied BEFORE the prediction: the delta term subtracts the DECAYED
state's own prediction (`S'_t^T k_t`), not the pre-decay one (`S_{t-1}^T k_t`). The
two forms differ whenever the gate is non-zero, by roughly 4e-2 relative -- above
the bf16 tolerance -- so a solver implementing the pre-decay form fails the
correctness gate. The code below is the definition; where any prose disagrees with
it, the code wins.

`g` is a scalar (per (step, head)) log-space decay applied to the whole state, and
`beta` is a per (step, head) write strength. Grouped value attention is supported:
`HV` value heads may share `H` key heads (`HV // H` value heads per key head), so a
kernel that assumes a one-to-one head mapping breaks -- which is the point of
grading it.

This is written as a readable sequential scan in float32 and is deliberately slow.
The surface is forward+backward: autograd differentiates this scan, and the input
gradients dq, dk, dv, dg, dbeta ARE the graded gradient oracle. Correctness is
defined as agreement with this function, never with any Triton or CUDA
implementation of it.

Ported verbatim (semantics, scale, grouping, decay-before-prediction) from the
project's calibrated KDA exemplar at seed/forge/categories/kda/bundle/reference.py.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes chunked_gated_delta
  - establishes: chunked_gated_delta is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: chunked_gated_delta agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.7 |
| starter fraction of anchor (measured) | 0.006466 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| bfloat16 tolerance | atol 0.02, rtol 0.03125 |
| float16 tolerance | atol 0.01, rtol 0.00390625 |
| float32 tolerance | atol 0.002, rtol 0.002 |
| comparison | within dtype tolerance |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes chunked_gated_delta. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: chunk-parallel gated delta rule, forward+backward (SURFACE fwdbwd).

FLAGSHIP / HARDEST task of the chunked_linear_attn family (spec.yaml hard:true):
it carries BOTH the per-head gated decay AND the delta-rule WY inverse (the exact
intra-chunk lower-triangular solve) AND grouped value attention (HV > H) -- the
most moving parts, so the widest gap between a naive chunk starter and the fused
production kernel, and the lowest reachable fraction of the anchor. It mirrors the
project's calibrated KDA exemplar (seed/forge/categories/kda) exactly.

Operator (reference.py is the definition; this is a summary):
    chunked_gated_delta(q[B,T,H,K], k[B,T,H,K], v[B,T,HV,V], g[B,T,HV], beta[B,T,HV])
        -> o[B,T,HV,V]
    S'_t = diag(exp(g_t)) S_{t-1};  S_t = S'_t + k_t (v_t - S'_t^T k_t)^T beta_t;
    o_t = S_t^T (q_t / sqrt(K)). Decay BEFORE prediction. Grouped value attention:
    HV value heads share H key heads (q, k expanded HV//H to one). SURFACE fwdbwd:
    gradients of q, k, v, g, beta are graded, timed and determinism-checked.

Anchor: fla.ops.gated_delta_rule.chunk_gated_delta_rule -- the production chunk
kernel of the pinned library (editable install /home/hephaestus-gpu/opt-fla).
sota() calls it with KEYWORDS and use_qk_l2norm_in_kernel=False; scale=None
defaults to K^-0.5 inside the library, matching the reference.

STATUS: authored-draft -- NEEDS GPU CALIBRATION. reference.py is CPU-verified
against the library's own naive ground truth to 6e-8 (fla naive_recurrent_gated_
delta_rule). TOL, TARGET_FRACTION_OF_SOTA, the vendor-shingle ceiling, the anchor
semantic-divergence check and every negative control's bound-gate behaviour are
INHERITED placeholders that a GPU calibration pass (measure/golden/controls) must
re-derive and confirm; see spec.yaml.

Distribution invariance (frontier-defeat-analysis §7.1): q, k are L2-normalized
along K (the operator's own contraction convention, FLA carries
use_qk_l2norm_in_kernel for exactly this), so the recurrence stays bounded at any
hidden scale; the per-invocation scale grade_core draws multiplies v (the one
magnitude carrier), and the tensors that are timed ARE the tensors that are
checked. NC-DP plants the §5.4 magnitude-branch exploit on the oracle under
FORGE_ORACLE=1 and must zero at exactly G2.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `cd33144cd268753f` |
| `kernel.py` | `d94638ccdb00881a` |
| `bench.py` | `4d16e43d5d9f5045` |
| `instruction.md` | `1c9deaa692d381f1` |
| `hidden_shapes.json` | `d7892c80c91044cd` |
| `solution/oracle_kernel.py` | `02569641bfa0803b` |

- **seed_base:** `89841`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=1 T=4096 H=8 HV=8 K=128 V=128 | bfloat16 |
| `g2` | B=4 T=1024 H=8 HV=8 K=128 V=128 | bfloat16, float16 |
| `g3` | B=2 T=2048 H=4 HV=16 K=128 V=128 | bfloat16 |
| `g4` | B=2 T=2048 H=8 HV=8 K=256 V=64 | bfloat16 |
| `g5` | B=16 T=512 H=8 HV=8 K=128 V=128 | bfloat16 |

- **forbidden symbols (21):** `fla.ops`, `import fla`, `from fla`, `chunk_gated_delta_rule`, `chunk_gdn`, `chunk_kda`, `chunk_delta_rule`, `chunk_gla`, `chunk_simple_gla`, `chunk_retention`, `chunk_delta_h`, `fused_recurrent_gated_delta_rule`, `fused_recurrent_gdn`, `fused_recurrent_kda`, `fused_recurrent_delta_rule`, `fused_recurrent_gla`, `fused_recurrent_simple_gla`, `fused_recurrent_retention`, `fused_recurrent`, `naive_recurrent_gated_delta_rule`, `naive_chunk_gated_delta_rule`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 e1babe41f81202dabfe8359afd8b5de29f822b2471ef5b83e1a6ec13bfadd1c8
FORGE-CANARY-SLOT-1 0226c24712e845c431dad141943e1c24d4e96ff960427d36ebcae525b45da5f5
FORGE-CANARY-SLOT-2 f66c96128e13464a51db92505b855c2d3e45b9331e8d30843e7c68fa0fe97f49
FORGE-CANARY-SLOT-3 3143df2c938f203c96533fb9165a5c0ac6d8607bda73a72c3ecaaf9af4f11605
FORGE-CANARY-END
