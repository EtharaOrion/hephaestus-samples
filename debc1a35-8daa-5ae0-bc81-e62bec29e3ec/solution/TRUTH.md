# truth.md - Golden Solve Path

## Task: hephaestus `moe_fused_experts_fused_downproj_swiglu_moe_fwdbwd` (anchor `torch._grouped_mm`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `fused_downproj_swiglu_moe` | **Editable:** `kernel.py`
- **Objective:** implement fused_downproj_swiglu_moe in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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
PRECOMPUTED top-k routing, forward + backward). This member is SwiGLU-MoE
augmented with a **per-expert output bias** `b2[E, D]` that is added INTO the
down-projection output BEFORE the routing weight multiply. The bias is why the
hardness lever of this variant is "fused DOWN-projection": a fast kernel must
fuse the bias-add AND the routing-weight scalar multiply into the epilogue of
the second (down) GEMM without materializing `(act @ w2[e])` as an intermediate.

    fused_downproj_swiglu_moe(x, w1, w2, b2, topk_idx, topk_w) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    w1:       [E, D, 2F] expert input projections; [:, :, :F] gate, [:, :, F:] up
    w2:       [E, F, D]  expert output (down) projections
    b2:       [E, D]     per-expert output bias, same dtype as x
    topk_idx: [T, A]     int64 expert ids in [0, E); no gradient
    topk_w:   [T, A]     routing weights, same dtype as x, already normalized
    y:        [T, D]     same dtype as x

Semantics, exactly:

    y[t] = sum_a topk_w[t, a] * ( f(topk_idx[t, a], x[t]) )        (a ascending)
    f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]  +  b2[e]
    silu(z) = z * sigmoid(z)

Every routed slot adds the routed expert's output bias BEFORE its topk_w scalar
multiply. The bias is INSIDE the routing sum: with A slots active per token
and the same expert e appearing k times in that token's routing, b2[e] is
added k times, each scaled by its own topk_w[t, a]. Experts no token routes
to receive exactly zero weight gradients AND exactly zero bias gradients.
Graded grads: dx, dw1, dw2, db2, dtopk_w (FIVE -- the family's four plus db2).
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes fused_downproj_swiglu_moe
  - establishes: fused_downproj_swiglu_moe is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: fused_downproj_swiglu_moe agrees with reference.py and is the baseline to beat
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
| starter fraction of anchor (measured) | 0.082537 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| bfloat16 tolerance | atol 18.0, rtol 0.1 |
| float32 tolerance | atol 5.0, rtol 0.02 |
| comparison | within dtype tolerance |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes fused_downproj_swiglu_moe. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: fused MoE + SwiGLU with per-expert output bias, forward + backward.

HARD member of the moe_fused_experts family (precomputed integer routing). The
hardness lever is FUSED DOWN-PROJECTION: a per-expert output bias `b2[E, D]`
is added into the down-projection output BEFORE the routing weight scalar,
and a fast kernel MUST fuse the bias-add + routing-weight scale into the
epilogue of the second (down) GEMM. No torch/vendor builtin exposes a grouped
GEMM with per-group additive bias + per-token scalar in one epilogue -- the
cuBLAS `torch._grouped_mm` epilogue is C = A @ B, and the anchor is forced
into three passes (grouped_mm -> +b2 -> * w_slot -> index_add). Adding db2
(five graded gradients rather than four) tightens the trap: a solution that
delegates the down GEMM to a vendor grouped_mm and adds the bias eagerly on
the outside cannot fuse without a custom kernel and cannot reach adoption.

STATUS: authored-draft. TOL/OUT_TOL/TARGET_FRACTION_OF_SOTA are AUTHORED
PLACEHOLDERS carried from moe_swiglu, WIDENED because db2 accumulates the
routing-weighted downstream signal over ALL routed slots of ALL tokens (long
row-sum, tightest atol floor of the five). MUST be re-measured on the
delivery GPU.

Operator (reference.py is the definition; this is a summary):
    fused_downproj_swiglu_moe(x[T,D], w1[E,D,2F], w2[E,F,D], b2[E,D],
        topk_idx[T,A] i64, topk_w[T,A]) -> y[T,D]
    y[t] = sum_a topk_w[t,a] * ((silu(g)*u) @ w2[e] + b2[e]),  a ascending;
    g = h @ w1[e,:,:F]; u = h @ w1[e,:,F:];   e = topk_idx[t,a].
    SURFACE fwdbwd: gradients of x, w1, w2, b2 AND topk_w are graded (FIVE).

Anchor: cuBLAS grouped-GEMM on expert-sorted tokens, eager silu*up, eager
`+ b2[e]` add on the segment, weighted index_add combine, fwd+bwd through
native autograd (_anchor_grouped). The bias-add and the topk_w multiply are
two separate elementwise passes the anchor cannot fuse.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `c3226d4618786664` |
| `kernel.py` | `223e1a6c9b65202d` |
| `bench.py` | `a5a1359eb8b45903` |
| `instruction.md` | `6f48d0f907dc8065` |
| `hidden_shapes.json` | `e66162c80122c5a7` |
| `solution/oracle_kernel.py` | `942a6387fdda6ce0` |

- **seed_base:** `91847`

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
FORGE-CANARY-SLOT-0 2a1deafa70f6b6810bf015b22a201f9107dd5267ca2a152d02cc11af9d42e3a6
FORGE-CANARY-SLOT-1 e21c30338eee80f964bf3276724a33ee4dfffac9413e43f84a415ab03f47a93d
FORGE-CANARY-SLOT-2 0ed09920d356ba04ce8c4b4b4e818ce9825f5b39759088fd264df7dcc1ab9d94
FORGE-CANARY-SLOT-3 8bf6068c93eaa8460bef77ff72b6a8f9b34cfc2382549e56f2d3755680ce8653
FORGE-CANARY-END
