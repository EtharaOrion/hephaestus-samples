# truth.md - Golden Solve Path

## Task: hephaestus `moe_fused_experts_relu2_glu_moe_fwdbwd` (anchor `torch._grouped_mm`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `relu2_glu_moe` | **Editable:** `kernel.py`
- **Objective:** implement relu2_glu_moe in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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
PRECOMPUTED top-k routing, forward + backward). This member is the ReGLU-squared
gated expert: the SwiGLU shape (gate + up projection sharing one w1, output
projection w2) with the gate nonlinearity switched from silu to squared-ReLU
(a.k.a. ReLU^2, the Primer/Squared-ReLU gate). The activation is nonstandard,
NOT covered by torch.nn.functional as a single fused op, and its epilogue is
piecewise: relu(g) is zero on half of g's support, so the fused MMA epilogue
that a fast kernel must implement is a masked square-and-multiply.

    relu2_glu_moe(x, w1, w2, topk_idx, topk_w) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    w1:       [E, D, 2F] expert input projections; columns [:, :, :F] are the
                         GATE projection, columns [:, :, F:] are the UP projection
    w2:       [E, F, D]  expert output projections
    topk_idx: [T, A]     int64 expert ids in [0, E); PRECOMPUTED routing, carries
                         no gradient. The same expert MAY appear in more than one
                         slot of a token (each slot contributes independently).
    topk_w:   [T, A]     routing weights, same dtype as x, already normalized
    y:        [T, D]     same dtype as x

Semantics, exactly:

    y[t] = sum_a topk_w[t, a] * f(topk_idx[t, a], x[t])        (a ascending)
    f(e, h) = ((relu(h @ w1[e, :, :F]))^2 * (h @ w1[e, :, F:])) @ w2[e]
    relu(z) = max(z, 0)

  * Arithmetic is accumulated in float32; only the final y is cast back to the
    input dtype. Combine order per token is fixed (ascending slot).
  * Forward AND backward. Graded gradients: dx, dw1, dw2, dtopk_w. topk_idx is
    integer and has no gradient. Experts no token routes to receive exactly
    zero weight gradients. The squared-relu backward has a piecewise derivative
    d/dg [relu(g)^2] = 2 * relu(g) * (g > 0) that a candidate must implement
    without smoothing.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes relu2_glu_moe
  - establishes: relu2_glu_moe is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: relu2_glu_moe agrees with reference.py and is the baseline to beat
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
| starter fraction of anchor (measured) | 0.092158 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| bfloat16 tolerance | atol 18.0, rtol 0.1 |
| float32 tolerance | atol 5.0, rtol 0.02 |
| comparison | within dtype tolerance |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes relu2_glu_moe. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: fused MoE + ReLU^2-gated GLU expert compute, forward + backward.

HARD member of the moe_fused_experts family (precomputed integer routing). The
hardness lever is a NONSTANDARD ACTIVATION EPILOGUE: squared-ReLU (ReLU^2, the
Primer/Squared-ReLU gate). No fused-MoE library exposes this activation as a
primitive -- vLLM / SGLang / Transformer Engine / fused_moe / grouped_gemm all
ship silu- and gelu-gated variants but not ReLU^2 -- and torch has no builtin
that composes it with a grouped GEMM epilogue. A candidate must implement the
piecewise mask-square-and-multiply epilogue itself and fuse it into the MMA
epilogue of the first expert GEMM, otherwise the two extra global-memory
round-trips (write gu, read gu, write act, read act) dominate the routed path.
The backward is likewise piecewise: d/dg[relu(g)^2] = 2*relu(g)*(g>0), and a
candidate that smooths it with an epsilon or with silu-shaped math drifts out
of tolerance on the hidden sweep.

STATUS: authored-draft. TOL/OUT_TOL/TARGET_FRACTION_OF_SOTA below are AUTHORED
PLACEHOLDERS carried from the proven moe_swiglu category, WIDENED for the
piecewise activation (its backward has zero measure at g=0 but a much larger
Lipschitz constant on the g>0 half than silu) and MUST be re-measured on the
delivery GPU.

Operator (reference.py is the definition; this is a summary):
    relu2_glu_moe(x[T,D], w1[E,D,2F], w2[E,F,D], topk_idx[T,A] i64, topk_w[T,A])
        -> y[T,D]
    y[t] = sum_a topk_w[t,a] * f(topk_idx[t,a], x[t]),  a ascending;
    f(e,h) = ((relu(h @ w1[e,:,:F]))^2 * (h @ w1[e,:,F:])) @ w2[e].
    SURFACE fwdbwd: gradients of x, w1, w2 AND topk_w are graded.

Anchor: a cuBLAS grouped-GEMM composition on expert-sorted tokens with the
ReLU^2 gate product in eager -- torch._grouped_mm for the two expert GEMM
stages, plain torch elementwise for relu-squared * up, weighted index_add
combine, fwd+bwd through native autograd (_anchor_grouped). The GEMMs are
cuBLAS and dominate device time; the activation epilogue is a second-pass
elementwise the anchor cannot fuse without a bespoke kernel. Anchor choice
mirrors moe_swiglu provenance.

Distribution invariance: every draw hides BOTH the input scale (SCALE_RANGE)
and the ROUTING distribution (zipf-like popularity, hidden strength + hidden
assignment, redrawn per invocation). Determinism uses A=8 so an atomic
scatter-add combine cannot pass.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `ca5f950ff15bf694` |
| `kernel.py` | `be8493f2aefe74c2` |
| `bench.py` | `1e1c606fc6e039e3` |
| `instruction.md` | `4e7a26b3e8ceda50` |
| `hidden_shapes.json` | `279600c3f0f05cb7` |
| `solution/oracle_kernel.py` | `8a427420d02ce01a` |

- **seed_base:** `82103`

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
FORGE-CANARY-SLOT-0 d496256d2ec9a1098ab38268bc44b8f2e34ff00736d440a3adbd8786124693dc
FORGE-CANARY-SLOT-1 fc4911e36fce37ba42c05c98fcb3404e2f65cf4995a1f71c6ac7edf2e32f5e02
FORGE-CANARY-SLOT-2 de521e9094845359d40b828a7fecff28f64b18a5be4d71f60cce414d011ba557
FORGE-CANARY-SLOT-3 48cd51a7a3604374d50cf1faa7856d02176588925907c8730f704754addab66e
FORGE-CANARY-END
