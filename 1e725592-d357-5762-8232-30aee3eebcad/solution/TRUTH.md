# truth.md - Golden Solve Path

## Task: hephaestus `moe_routed_hierarchical_group_topk_fwdbwd` (anchor `torch._grouped_mm`)

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
This member is HIERARCHICAL group-then-expert top-K routing (DeepSeek-V2
device-limited routing): experts are partitioned into G equal-size groups,
each token selects the single top-1 group by its GROUP SCORE (sum of the
group's top-2 expert softmax weights), then selects top-K experts within
that chosen group by expert p; combine by p_sel/p_sel.sum(-1). This is a
TWO-STAGE selection with two independent discontinuity boundaries.

    fused_moe(x, router_w, w1, w2, G, K) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    router_w: [D, E]     router projection; E must be divisible by G
    w1:       [E, D, 2F] expert input projections; :F gate, F: up
    w2:       [E, F, D]  expert output projections
    G:        int        number of groups (E % G == 0)
    K:        int        top-K experts within selected group, 1 <= K <= E/G
    y:        [T, D]     same dtype as x

Semantics, exactly (all arithmetic in float32; inputs cast up, only final y cast back):

    logits = x @ router_w                              # [T, E]
    p      = softmax(logits, dim=-1)                   # [T, E]
    reshape p to [T, G, S] where S = E / G             # per-group [T, S] slabs
    # stage 1: group score = sum of top-2 p within each group
    gs[t, g] = top2_sum(p[t, g, :])                    # [T, G]
    grp[t]   = argmax_g gs[t, g]  ties -> LOWER GROUP index; -0.0 == +0.0
    # stage 2: top-K experts within the chosen group by p
    sel_local[t] = top-K positions in p[t, grp[t], :]  ties LOWER LOCAL index
    sel[t, k]    = grp[t] * S + sel_local[t, k]        # global expert index
    # combine
    w[t] = p[t, sel[t]] / p[t, sel[t]].sum(-1)         # p_sel/sum
    y[t] = sum_k w[t, k] * f(sel[t, k], x[t])
    f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]

  * Both boundaries decided by the int64-key trick: the group score is a
    small [G] softmax-of-sums, and its tie-break rule is LOWER group index;
    the intra-group top-K is over [S] entries and its tie-break rule is
    LOWER LOCAL expert index (which is also the LOWER GLOBAL index within
    the chosen group).
  * Only experts inside the chosen group receive gradients. Every logit
    inside that group's slab receives a scoring-softmax gradient AND a
    group-score gradient (top-2 of that group's p sums into gs[t, grp]);
    logits outside the chosen group receive only the scoring-softmax
    gradient through gs[t, g!=grp[t]] flowing back into p, but since the
    group selection is argmax (locally constant), those grads land only
    on the scoring softmax's Jacobian into the WHOLE row (as always).
  * The graded surface is fwd AND bwd; 4 grads (dx, drouter_w, dw1, dw2).
    Experts outside the chosen group of every token receive exact-zero
    weight gradients (both w1 and w2).

Correctness is defined as agreement with this function under the disclosed
per-dtype tolerances -- and where prose and this code disagree, the code wins.
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
| target fraction of anchor | 0.24 |
| starter fraction of anchor (measured) | 0.08389 |
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
Taskdef: hierarchical group-then-expert routed fused-MoE (SURFACE fwdbwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). Family
moe_routed: the routing decision -- scoring softmax, group selection,
intra-group top-K, combine -- is INSIDE the graded surface. This member is
marked hard:true. DeepSeek-V2 device-limited routing pattern:

    fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D], G, K) -> y[T,D]
    logits = x @ router_w; p = softmax(logits); pg = reshape [T, G, E/G];
    group score gs[t,g] = top-2 sum of pg[t,g]; grp[t] = argmax_g gs (ties
    LOWER group); intra-group top-K by pg[t,grp] (ties LOWER local index);
    global sel = grp*S + local; w = p_sel / p_sel.sum(-1); y[t] = sum_k
    w[t,k] * f(sel[t,k], x[t]). SwiGLU experts; fp32 arithmetic, one final
    cast. Gradients graded: dx, drouter_w, dw1, dw2.

WHY HARD: TWO independent discontinuity boundaries stack inside one op --
the group-selection argmax AND the intra-group top-K -- so the fp64 shadow
softmax must enforce TWO margins simultaneously, and bf16 router-arithmetic
drift can flip EITHER boundary independently. Only experts inside the
chosen group of a token contribute to that token's output; grouping is
what lets serving stacks localize expert traffic per-device, and no torch
or vendor builtin covers "scoring-softmax + top-2-sum group scoring +
argmax group + intra-group top-K + p_sel/sum combine + grouped-GEMM"
as a single primitive. The candidate must rebuild every stage in
@triton.jit.

UNCALIBRATED SEAMS (re-measure before freeze): TOL, OUT_TOL,
TARGET_FRACTION_OF_SOTA, _MARGIN, hidden shape sweep, _SOTA_ROUTE, negative-
control anchor strings.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `51a77e309832763a` |
| `kernel.py` | `7163d93cc09ab9c2` |
| `bench.py` | `95fd2f024a67799b` |
| `instruction.md` | `d8cf3facbe3b7abd` |
| `hidden_shapes.json` | `75e4c2f6d5f662cd` |
| `solution/oracle_kernel.py` | `2bd220f3916e28e7` |

- **seed_base:** `91697`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | T=8192 D=2048 F=1408 E=64 G=8 K=4 | float32, bfloat16 |
| `g2` | T=4096 D=2048 F=704 E=160 G=8 K=4 | float32, bfloat16 |
| `g3` | T=16384 D=1024 F=1408 E=32 G=4 K=2 | float32, bfloat16 |

- **forbidden symbols (7):** `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 19386f5093ddc3f25489ad1e573171ada161012e470c3b9d8625cb5eac9c2606
FORGE-CANARY-SLOT-1 3b085cfcafe8fff741c881194a38510b32d3e1bf8cd1dcea5c82181ef75922b8
FORGE-CANARY-SLOT-2 b7d6aeace93d74b5bf64cb41bada31abd24dacd9928922865071163705d181cf
FORGE-CANARY-SLOT-3 17b3553eebd0b23f27d50867357b7ddcc0cb34454ce062932fbc72fe31303d1f
FORGE-CANARY-END
