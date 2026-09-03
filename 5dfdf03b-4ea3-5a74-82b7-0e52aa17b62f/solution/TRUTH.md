# truth.md - Golden Solve Path

## Task: hephaestus `moe_routed_sigmoid_topk_sumrenorm_fwdbwd` (anchor `torch._grouped_mm`)

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
This member is SIGMOID top-K routing with SUM-RENORMALIZED combine weights
(DeepSeek-V3-adjacent, distinct from the sibling sigmoid_topk_glm which uses
a raw sigmoid combine and a selection bias): sigmoid affinity, top-K by
sigmoid, weights = s_sel / s_sel.sum() combine. NO selection bias.

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
    s      = sigmoid(logits)                           # [T, E] independent
                                                       # per-expert affinities
    sel    = top-K experts per token by s, ties broken by LOWER expert index
             first; -0.0 and +0.0 compare equal
    w      = s[t, sel] / s[t, sel].sum(-1)             # sum-renormalized
                                                       # combine (distinct from
                                                       # sigmoid_topk_glm's raw
                                                       # sigmoid combine and
                                                       # from softmax_topk_
                                                       # mixtral's p_sel/sum
                                                       # over softmax scores)
    y[t]   = sum_k w[t, k] * f(sel[t, k], x[t])
    f(e,h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]

  * Sigmoid is monotone in the logit, so top-K by s and top-K by logit pick
    the same set with the same tie structure. Selection is by (s desc,
    index asc) via the int64-key trick.
  * The sum-renorm combine keeps the router gradient alive on every selected
    logit (each selected s enters both the numerator and the denominator of
    a token's weight vector). A candidate that uses raw sigmoid (no renorm)
    or the softmax_topk_mixtral rule (p_sel/sum on softmax scores) will fail
    every graded gradient tolerance.
  * The graded surface is forward AND backward: gradients for x, router_w,
    w1 and w2 are graded (there is no router_b). Selection indices are
    almost-everywhere locally constant. Gradients flow through the sigmoid
    of the selected experts (NOT dense over the whole row -- sigmoid is
    elementwise, so unselected experts get zero from the router-side path
    of the routed contribution) and through the expert compute AND through
    the sum-renorm coupling of the K selected weights.
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
| target fraction of anchor | 0.3 |
| starter fraction of anchor (measured) | 0.076665 |
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
Taskdef: sigmoid top-K routed fused-MoE with sum-renormalized combine.

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). Family
moe_routed: the routing decision -- scores, selection, combine weights -- is
INSIDE the graded surface. This member is marked hard:true.

    fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D], K) -> y[T,D]
    logits = x @ router_w; s = sigmoid(logits); sel = top-K by s (ties LOWER
    expert index); w = s_sel / s_sel.sum(-1); y[t] = sum_k w[t,k] *
    f(sel[t,k], x[t]). NO selection bias. SwiGLU experts, fp32 arithmetic,
    one final cast. Gradients graded: dx, drouter_w, dw1, dw2.

WHY HARD: the combine rule is a third distinct primitive within the family
(sigmoid_topk_glm ships raw sigmoid + bias, softmax_topk_mixtral ships
p_sel/sum on softmax scores, this variant ships s_sel/sum on sigmoid scores).
The sum-renormalization COUPLES the K selected sigmoid gradients: every
selected s enters both the numerator and the denominator of every weight in
its token's combine vector. The router backward through sigmoid is
elementwise (not dense over the whole row like softmax), so the coupling
concentrates numerical risk on the K selected columns -- the fp32-sensitive
lever from KernelBench-hard. No torch or vendor builtin covers "sigmoid +
top-K + sum-renorm + gated grouped-GEMM + coupled-sigmoid-backward" in one
primitive.

UNCALIBRATED SEAMS (re-measure before freeze): TOL, OUT_TOL,
TARGET_FRACTION_OF_SOTA, _MARGIN, hidden shape sweep, _SOTA_ROUTE, negative-
control anchor strings.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `c4564f3fef31cdf6` |
| `kernel.py` | `d7e54bd2a091a11e` |
| `bench.py` | `33275143b61fbc64` |
| `instruction.md` | `96ce6cdfad196870` |
| `hidden_shapes.json` | `99e967346d792106` |
| `solution/oracle_kernel.py` | `b2b6cf888b845c80` |

- **seed_base:** `91231`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | T=8192 D=2048 F=1408 E=64 K=8 | float32, bfloat16 |
| `g2` | T=4096 D=2048 F=704 E=160 K=6 | float32, bfloat16 |
| `g3` | T=16384 D=1024 F=1408 E=32 K=4 | float32, bfloat16 |

- **forbidden symbols (7):** `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 de3338688d6986c00757312e50fa2a89281204810a3d37ec23107c88f14465e0
FORGE-CANARY-SLOT-1 372ff620111834e8dac9c9d39937f65f2dad23ef677ee5e6eb30c1c9485a6b1a
FORGE-CANARY-SLOT-2 3efd1b54554e6b7636893e2e72469eb8719b5697473a877c24c17aa30d6993d1
FORGE-CANARY-SLOT-3 bae59c1d9ba44461565342f496a7573d1dfa27bdd303c745b47f816275797915
FORGE-CANARY-END
