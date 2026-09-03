# truth.md - Golden Solve Path

## Task: hephaestus `moe_fused_experts_shared_plus_routed_swiglu_fwdbwd` (anchor `torch._grouped_mm`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `shared_plus_routed_swiglu` | **Editable:** `kernel.py`
- **Objective:** implement shared_plus_routed_swiglu in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

Family: moe_fused_experts (fused mixture-of-experts expert compute with
PRECOMPUTED top-k routing, forward + backward). This member is the HARDEST:
an ALWAYS-ON dense SHARED expert applied to EVERY token, ADDED to the routed
SwiGLU experts. Two structurally different GEMM regimes coexist in one op --
a dense [T,D]x[D,2F] shared path over all tokens, and an irregular
expert-grouped routed path -- and both are differentiated.

    shared_plus_routed_swiglu(x, ws1, ws2, w1, w2, topk_idx, topk_w) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    ws1:      [D, 2F]    SHARED expert input projection; [:, :F] gate, [:, F:] up
    ws2:      [F, D]     SHARED expert output projection
    w1:       [E, D, 2F] routed expert input projections; [:, :, :F] gate, [:, :, F:] up
    w2:       [E, F, D]  routed expert output projections
    topk_idx: [T, A]     int64 expert ids in [0, E); PRECOMPUTED routing, carries
                         no gradient. The same expert MAY appear in more than one
                         slot of a token (each slot contributes independently).
    topk_w:   [T, A]     routing weights, same dtype as x, already normalized
    y:        [T, D]     same dtype as x

Semantics, exactly:

    y[t] = f_shared(x[t]) + sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])   (a ascending)
    f_shared(h) = (silu(h @ ws1[:, :F]) * (h @ ws1[:, F:])) @ ws2
    f(e, h)     = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]
    silu(z)     = z * sigmoid(z)

  * The shared expert is applied to ALL tokens and is NOT weighted by any
    routing weight (it is always on, weight 1). It uses the SAME intermediate
    width F as the routed experts but its OWN weights ws1/ws2.
  * Arithmetic is accumulated in float32; only the final y is cast back to the
    input dtype. Combine order per token is fixed (ascending slot); the shared
    contribution is added last as written above (associativity within tol).
  * Forward AND backward. Graded gradients are dx, dws1, dws2, dw1, dw2 and
    topk_w -- the family's four PLUS the two shared-expert weight gradients,
    because the always-on shared path's backward is part of this operator.
    topk_idx is integer and has no gradient. Experts no token routes to receive
    exactly zero weight gradients (the shared expert always receives gradient).

The implementation is deliberately simple and slow (dense shared path + a
python loop over (slot, expert) pairs for the routed path), autograd-capable
end to end.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes shared_plus_routed_swiglu
  - establishes: shared_plus_routed_swiglu is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: shared_plus_routed_swiglu agrees with reference.py and is the baseline to beat
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
| starter fraction of anchor (measured) | 0.082568 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| bfloat16 tolerance | atol 18.0, rtol 0.1 |
| float32 tolerance | atol 5.0, rtol 0.02 |
| comparison | within dtype tolerance |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes shared_plus_routed_swiglu. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: shared expert + routed SwiGLU experts, forward + backward (fwdbwd).

The HARD member of the moe_fused_experts family (precomputed integer routing).
Two structurally different GEMM regimes coexist in ONE operator: a DENSE
always-on SHARED expert applied to every token (its own weights ws1/ws2, weight
1, no routing) ADDED to the irregular routed SwiGLU path. SIX gradients are
graded: dx, dws1, dws2 (shared) and dw1, dw2, dtopk_w (routed). Everything
category-specific the generic verifier consumes lives here (protocol:
core/taskdef_api.py).

STATUS: authored-draft. TOL/OUT_TOL/TARGET_FRACTION_OF_SOTA below are AUTHORED
PLACEHOLDERS carried from the proven moe_swiglu category; they MUST be
re-measured on the delivery GPU. CALIBRATION NOTE: dws1/dws2 accumulate over
ALL T tokens (up to 16384 rows -- longer than any routed segment), so their
honest atol floor is likely the LARGEST in the family; and the reachable
fraction of the anchor is the LOWEST (two regimes to fuse), so
TARGET_FRACTION_OF_SOTA should calibrate below the other members'.

Operator (reference.py is the definition; this is a summary):
    shared_plus_routed_swiglu(x[T,D], ws1[D,2F], ws2[F,D], w1[E,D,2F],
        w2[E,F,D], topk_idx[T,A] i64, topk_w[T,A]) -> y[T,D]
    y[t] = f_shared(x[t]) + sum_a topk_w[t,a] * f(topk_idx[t,a], x[t]);
    f_shared(h) = (silu(h@ws1[:, :F]) * (h@ws1[:, F:])) @ ws2   (weight 1, all t);
    f(e,h)      = (silu(h@w1[e,:,:F]) * (h@w1[e,:,F:])) @ w2[e].
    SURFACE fwdbwd: gradients of x, ws1, ws2, w1, w2 AND topk_w are graded,
    timed and determinism-checked. topk_idx is precomputed routing, no grad.

Anchor: a production composition of the SAME two regimes -- the routed path is a
cuBLAS grouped-GEMM composition on expert-sorted tokens (torch._grouped_mm,
eager silu, weighted index_add combine) and the shared path is a dense torch
SwiGLU MLP over all tokens; fwd+bwd through native autograd (_anchor below).
The GEMMs are cuBLAS and dominate device time. Anchor choice must be MEASURED
on the delivery image (see moe_swiglu provenance).

Distribution invariance: every draw hides BOTH the input scale (SCALE_RANGE)
and the ROUTING distribution (zipf-like popularity, hidden strength + hidden
assignment, redrawn per invocation). The always-on shared path means expert
loads are imbalanced but the shared GEMM is dense every time. The tensors that
are timed are the tensors that are checked; NC-DP proves the closure with the
exemption env ON. Determinism uses A=8 so an atomic scatter-add combine cannot
pass.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `276f334d5e06fddb` |
| `kernel.py` | `94f5552eceb0ca99` |
| `bench.py` | `45fdcd432edc3ff3` |
| `instruction.md` | `ba90478839d15efc` |
| `hidden_shapes.json` | `aeb1d40da7c8dfbe` |
| `solution/oracle_kernel.py` | `1700f8bfa2a6cf60` |

- **seed_base:** `76259`

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
FORGE-CANARY-SLOT-0 d538446f85bc39875bb628f00cf17a7a97a5d159517f50bd43ce3c06b2d8acb8
FORGE-CANARY-SLOT-1 7df7803065da4f96c1800b8cd6e504d2305e6814fc02e8d6ebff6c2f12d828c5
FORGE-CANARY-SLOT-2 6f74b1b3d0e00698544d032260b10ba32d241718cc4a431f33cd2d390e5e46d7
FORGE-CANARY-SLOT-3 b3898928d8c8063037f30ba821403dbc0213ca27a8b3aec5485b3f40508e9da9
FORGE-CANARY-END
