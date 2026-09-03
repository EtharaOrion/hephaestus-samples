# truth.md - Golden Solve Path

## Task: hephaestus `moe_routed_fine_grained_softmax_topk_fwdbwd` (anchor `torch._grouped_mm`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `fused_moe` | **Editable:** `kernel.py`
- **Objective:** implement fused_moe in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

Family: moe_routed (routing computed INSIDE the graded op; forward + backward).
This member is FINE-GRAINED softmax top-K routing with a re-softmax combine
rule (DeepSeek-V3-adjacent, distinct from the mixtral p_sel/sum rule): very
large expert counts (E in the hundreds) with a small K (e.g. K=8) per token,
and the combine weights are softmax over the SELECTED logits, not the
softmax weights divided by their sum.

    fused_moe(x, router_w, w1, w2, K) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    router_w: [D, E]     router projection, same dtype as x
    w1:       [E, D, 2F] expert input projections; [:, :, :F] gate, [:, :, F:] up
    w2:       [E, F, D]  expert output projections
    K:        int        experts selected per token, 1 <= K <= E
    y:        [T, D]     same dtype as x

Semantics, exactly (all arithmetic in float32; inputs are cast up, only the
final y is cast back to the input dtype):

    logits = x @ router_w                              # [T, E]
    p      = softmax(logits, dim=-1)                   # [T, E] scoring softmax
    sel    = top-K experts per token by p, ties broken by LOWER expert index
             first; -0.0 and +0.0 compare equal
    w      = softmax(logits[t, sel], dim=-1)           # RE-SOFTMAX over the
                                                       # SELECTED LOGITS
                                                       # (distinct from
                                                       # p_sel/sum: order-
                                                       # preserving but not
                                                       # linearly equal to the
                                                       # renormalized weights)
    y[t]   = sum_k w[t, k] * f(sel[t, k], x[t])
    f(e,h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]
    silu(z) = z * sigmoid(z)

  * softmax is monotone in logits, so top-K by p and top-K by logit pick the
    same set with the same tie structure. Selection is by (p desc, index asc)
    via the int64-key trick used in the sibling variants.
  * The re-softmax combine rule is what makes gradients flow through EVERY
    selected column (through softmax over K terms) AND through the scoring
    softmax (through the top-1 selection ..). A candidate that combines by
    p_sel/sum(p_sel) will fail every graded gradient tolerance.
  * The graded surface is forward AND backward: gradients for x, router_w,
    w1 and w2 are graded (there is no router_b). Selection indices are
    almost-everywhere locally constant. Gradients flow through the scoring
    softmax over the whole expert row (into router_w and x) AND through the
    re-softmax over the K selected logits AND through the expert compute.
  * Experts no token selects receive exactly zero weight gradients.
  * The same expert never appears twice for one token (top-K of E distinct).

Correctness is defined as agreement with this function under the disclosed
per-dtype tolerances -- and where prose and this code disagree, the code wins.

The implementation below is deliberately simple and deliberately slow: a
python loop over experts with boolean-mask gathers. It is autograd-capable
end to end, which is how the harness obtains the reference gradients.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes fused_moe
  - establishes: fused_moe is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: fused_moe agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.22 |
| starter fraction of anchor (measured) | 0.083705 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| float32 tolerance | atol 0.004, rtol 0.004 |
| bfloat16 tolerance | atol 0.045, rtol 0.045 |
| comparison | within dtype tolerance |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes fused_moe. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: fine-grained softmax top-K routed fused-MoE with re-softmax combine.

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). Family
moe_routed: the routing decision -- scores, selection, combine weights -- is
INSIDE the graded surface. This member is marked hard:true.

    fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D], K) -> y[T,D]
    logits = x @ router_w; p = softmax(logits); sel = top-K by p (ties LOWER
    expert index); w = softmax(logits[sel]) -- RE-softmax over the SELECTED
    LOGITS, distinct from the mixtral p_sel/sum rule; y[t] = sum_k w[t,k] *
    f(sel[t,k], x[t]). SwiGLU experts, fp32 arithmetic, one final cast.
    Gradients graded: dx, drouter_w, dw1, dw2 -- through the scoring softmax
    (whole expert row), the re-softmax over K, and the expert compute.

WHY HARD (KernelBench-hard lever): very large expert counts (E in the
hundreds, K small, e.g. E=256/K=8 -- the DeepSeek-V3 fine-grained regime)
combined with a re-softmax combine rule. No torch or vendor builtin covers
"softmax + top-K selection + re-softmax-over-selected-logits + gated
grouped-GEMM + dense-router-backward" in a single primitive. The scoring
softmax is HUGE ([T, 256]) and dominates the router matmul; the combine
softmax is short but couples the K selected logits to every expert path in
the backward. bf16 router drift flips top-K boundaries; a p_sel/sum-shaped
combine looks locally right but fails every graded gradient.

UNCALIBRATED SEAMS (re-measure before freeze): TOL, OUT_TOL,
TARGET_FRACTION_OF_SOTA, _MARGIN, hidden shape sweep, _SOTA_ROUTE, negative-
control anchor strings.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `0eb5778e08cbe9b3` |
| `kernel.py` | `91b07e78380e2455` |
| `bench.py` | `fae2ac176a401342` |
| `instruction.md` | `1fe7c69757631043` |
| `hidden_shapes.json` | `6c23e4750a072abd` |
| `solution/oracle_kernel.py` | `5f6fb6987e7e1a3a` |

- **seed_base:** `91013`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | T=8192 D=2048 F=704 E=256 K=8 | float32, bfloat16 |
| `g2` | T=4096 D=2048 F=1408 E=128 K=4 | float32, bfloat16 |
| `g3` | T=16384 D=1024 F=704 E=160 K=6 | float32, bfloat16 |

- **forbidden symbols (7):** `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 291cde5eb2f646bdd8b7a65b2d9e020499aa1f717aca1e853a0fc60c70f0b3ad
FORGE-CANARY-SLOT-1 1f880b229b7aba1f0d26116887035113fd68d9b50e9638140bfa96b898206960
FORGE-CANARY-SLOT-2 4dc73cace413b93e20fd114987e457a196f0a4ad4b43222c1ca235edcfc5e602
FORGE-CANARY-SLOT-3 7d17190aceccf602a2b3f4a1af66e4c5c86a0ae61f854c04e87b211b036eb791
FORGE-CANARY-END
