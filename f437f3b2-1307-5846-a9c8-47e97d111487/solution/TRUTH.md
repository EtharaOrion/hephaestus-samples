# truth.md - Golden Solve Path

## Task: hephaestus `moe_routed_expert_choice_fwdbwd` (anchor `torch._grouped_mm`)

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
This member is EXPERT-CHOICE routing (Zhou et al. 2022): instead of each token
picking its experts, each EXPERT picks its top-C tokens by affinity. Routing is
irregular in the token dimension -- a token may be chosen by any number of
experts from 0 to E -- which is what makes this the hardest member of the family.

    fused_moe(x, router_w, w1, w2, C) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    router_w: [D, E]     router projection, same dtype as x
    w1:       [E, D, 2F] expert input projections; [:, :, :F] is the GATE
                         projection, [:, :, F:] is the UP projection
    w2:       [E, F, D]  expert output projections
    C:        int        capacity: tokens each expert selects, 1 <= C <= T
    y:        [T, D]     same dtype as x

Semantics, exactly (all arithmetic in float32; inputs are cast up, only the
final y is cast back to the input dtype):

    logits = x @ router_w                             # [T, E]
    S      = softmax(logits, dim=-1)                  # [T, E] affinities;
                                                      # softmax is over EXPERTS,
                                                      # so each token's row sums
                                                      # to 1 (token-to-expert
                                                      # affinity)
    for each expert e:
        sel[e] = the C tokens with the largest S[:, e], ties broken by LOWER
                 TOKEN index first; -0.0 and +0.0 compare equal
        g[e,c] = S[sel[e,c], e]                        # the affinity gate, used
                                                        # WITHOUT renormalization
        f_e    = (silu(x[sel[e]] @ w1[e,:,:F]) * (x[sel[e]] @ w1[e,:,F:])) @ w2[e]
    y[t]   = sum over every (e, c) with sel[e,c] == t of  g[e,c] * f_e[c]
    silu(z) = z * sigmoid(z)

  * The selection is over TOKENS for each fixed expert, so the deterministic
    tie-break is by LOWER TOKEN index (the structural analog of the
    lower-expert-index rule used by the token-choice siblings). It is realised
    exactly by packing each affinity's order-preserving float32 bits with the
    inverted token index into one int64 key per (expert, token) entry -- equal
    affinities never compete, key order IS selection order. -0.0 is
    canonicalised to +0.0 first.
  * The gate g[e,c] = S[sel[e,c], e] is the RAW softmax affinity and is used
    without renormalization: a token chosen by k experts sums k contributions,
    each scaled by that expert's affinity, so its total scale is data-dependent
    (unlike the token-choice gates, which renormalize to 1 per token).
  * The combine is irregular: each token gathers a variable number of expert
    contributions (0 .. E). They are summed in ascending expert order, so the
    accumulation order is fixed and the result is deterministic.
  * The graded surface is forward AND backward: gradients for x, router_w, w1
    and w2 are graded (there is no router_b). Selection indices are almost-
    everywhere locally constant; gradients flow through the affinity gates
    S[sel[e], e] (the softmax Jacobian over the whole expert row, into router_w
    and x) and through the expert computation. This function's autograd graph is
    the definition of those gradients.
  * A token selected by NO expert receives y[t] = 0 and, since its row of S is
    then unused, exactly zero input gradient. Experts are always full (exactly
    C tokens each), so no expert has empty weight gradients unless C == 0
    (excluded: C >= 1).

Correctness is defined as agreement with this function under the disclosed
per-dtype tolerances -- and where prose and this code disagree, the code wins.

The implementation below is deliberately simple and deliberately slow: a python
loop over experts with per-expert gathers. It is autograd-capable end to end,
which is how the harness obtains the reference gradients.
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
| target fraction of anchor | 0.25 |
| starter fraction of anchor (measured) | 0.094353 |
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
Taskdef: expert-choice routed fused-MoE (SURFACE fwdbwd). THE HARD MEMBER.

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). Family
moe_routed: the routing decision -- affinities, selection, gate -- is INSIDE the
graded surface, and gradients flow through the router. This member is
expert-choice routing (Zhou et al. 2022) and is marked hard:true: each EXPERT
picks its top-C tokens, so a token is chosen by a data-dependent number of
experts (0..E) and the combine is irregular in the token dimension.

    fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D], C) -> y[T,D]
    logits = x @ router_w; S = softmax(logits, dim=-1) (softmax over EXPERTS, so
    each token's affinity row sums to 1); for each expert e, sel[e] = the top-C
    tokens by S[:,e] (ties to the LOWER TOKEN index); the gate g[e,c] =
    S[sel[e,c], e] is the RAW affinity (no renormalization); y[t] = sum over
    every (e,c) with sel[e,c]==t of g[e,c] * f(e, x[t]), summed in ascending
    expert order. SwiGLU experts; fp32 arithmetic, one final cast. Gradients
    graded: dx, drouter_w, dw1, dw2 -- all four flow through the softmax
    affinities (a DENSE Jacobian over the expert row) and the expert compute.
    reference.py is the definition.

WHY HARD (most headroom / least reachable): the selection is over the TOKEN axis
(a top-C over T, transposed from the token-choice siblings' top-A over E), the
per-token combine has variable fan-in, and the softmax-over-experts gate coupled
with a token-axis selection resists the token-sorted grouped-GEMM dispatch the
other members reuse. The dispatch itself is perfectly load-balanced (exactly C
tokens per expert, E*C pairs), but a correct deterministic combine must reduce a
data-dependent number of contributions per output row.

UNCALIBRATED SEAMS (must be re-measured by a golden/controls run in this folder
before freeze): TOL, OUT_TOL, TARGET_FRACTION_OF_SOTA, _MARGIN (the affinity
scale is ~1/E, unlike the [0,1] sigmoid siblings, so the margin floor may need
per-E adjustment), the hidden shape sweep, _SOTA_ROUTE, and the negative-control
anchor strings (which target this folder's kernel.py starter). Values below are
informed placeholders carried from the sibling calibrated moe_routed task and
flagged where set.

Selection-stability design (load-bearing): expert-choice selection is
discontinuous, so make_inputs enforces a decision margin PER EXPERT -- experts
whose C-th/(C+1)-th affinity gap is positive but below _MARGIN have their two
boundary tokens redrawn (fp64 shadow softmax over the dtype-cast tensors,
iterated). Exact ties (gap == 0) are legal and manufactured by the tie-forcing
draw, which duplicates TOKEN rows (bit-equal x -> bit-equal affinities in both
dtypes): a boundary that falls inside a duplicate group is split by the lower-
token-index rule, and because the two tied tokens are DIFFERENT output rows, a
wrong tie winner moves the contribution to the wrong row and changes y.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `29145742042ec51b` |
| `kernel.py` | `12a552eb610362e6` |
| `bench.py` | `89fa2807f28210a3` |
| `instruction.md` | `d25ef29480844480` |
| `hidden_shapes.json` | `f00110bad01b87ad` |
| `solution/oracle_kernel.py` | `27d6891949f5dd2a` |

- **seed_base:** `63709`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | T=8192 D=2048 F=1408 E=64 C=128 | float32, bfloat16 |
| `g2` | T=4096 D=2048 F=704 E=160 C=32 | float32, bfloat16 |
| `g3` | T=16384 D=1024 F=1408 E=32 C=512 | float32, bfloat16 |

- **forbidden symbols (7):** `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 4e7e762d8b3dad1dcfb246d2edc3390b8a553be5b596e14b1c00221d229e5cae
FORGE-CANARY-SLOT-1 f08bb6ec1361f90cd08417c20f95f8ec81e31dc6bcf31a3558867cdecfbd918b
FORGE-CANARY-SLOT-2 8c5423ffa0358f6242ad22232d8bca7a153d783182ad9b217f7d2e58727be8fc
FORGE-CANARY-SLOT-3 7fa891e696b67473713e10a3920721b855ce5dff2494d7bc9dff7fbe0536f295
FORGE-CANARY-END
