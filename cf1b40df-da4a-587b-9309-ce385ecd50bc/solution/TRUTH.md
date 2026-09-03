# truth.md - Golden Solve Path

## Task: hephaestus `moe_routed_capacity_dropped_switch_fwdbwd` (anchor `torch._grouped_mm`)

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
This member is CAPACITY-LIMITED Switch top-1 routing (GShard / Switch-Transformer
overflow discipline): each token picks its single expert by argmax softmax
probability, but each expert accepts at most `capacity` tokens; overflow tokens
are DROPPED (their output row is zero and they contribute no weight gradient).
The irregularity comes from a data-dependent per-expert drop mask.

    fused_moe(x, router_w, w1, w2, capacity) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    router_w: [D, E]     router projection, same dtype as x
    w1:       [E, D, 2F] expert input projections; [:, :, :F] gate, [:, :, F:] up
    w2:       [E, F, D]  expert output projections
    capacity: int        max tokens each expert accepts, 1 <= capacity <= T
    y:        [T, D]     same dtype as x; dropped tokens carry an exact-zero row

There is NO router bias and NO A: top-1 is intrinsic; overflow discipline is
the whole point of this member.

Semantics, exactly (all arithmetic in float32; inputs are cast up, only the
final y is cast back to the input dtype):

    logits = x @ router_w                              # [T, E]
    p      = softmax(logits, dim=-1)                   # [T, E]
    e*[t]  = argmax_e p[t, e]  ties -> LOWER expert index; -0.0 == +0.0
    for each expert e:
        candidates = {t : e*[t] == e}
        keep the top-`capacity` candidates by p[t, e], ties broken to LOWER
        token index; if |candidates| <= capacity, keep them all; excess
        candidates are DROPPED.
    g[t]   = p[t, e*[t]]                               # raw Switch gate
    y[t]   = g[t] * f(e*[t], x[t])  if t is kept, else 0
    f(e,h) = (silu(h @ w1[e,:,:F]) * (h @ w1[e,:,F:])) @ w2[e]

  * The token-per-expert selection uses the same order-preserving int64 key
    trick as the family siblings: pack the softmax weight's float32 bits with
    the inverted token index into one int64, so within an expert's candidate
    set the top-`capacity` are picked with a stable, structural tie-break.
  * The gate is the RAW softmax probability p[t, e*], NOT renormalized. Kept
    tokens receive the full gate scale; dropped tokens receive zero output.
  * The graded surface is forward AND backward. Gradients flow through the
    Switch gate p[t, e*] (softmax Jacobian over the whole expert row, into
    router_w and x) and through the expert compute -- ONLY for kept tokens.
    Dropped tokens have y[t] == 0, so they carry zero gradient into x[t],
    w1[e*], w2[e*] AND into router_w through their softmax row. This is
    load-bearing: a candidate that silently keeps a dropped token in the
    backward path will fail dw1/dw2/drouter_w on the graded tolerance.
  * An expert that no token picks contributes exact-zero weight gradients.

Correctness is defined as agreement with this function under the disclosed
per-dtype tolerances -- and where prose and this code disagree, the code wins.

The implementation below is deliberately simple and deliberately slow: a python
loop over experts with per-expert candidate masking and per-expert top-`capacity`
gathers. It is autograd-capable end to end, which is how the harness obtains
the reference gradients.
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
| target fraction of anchor | 0.28 |
| starter fraction of anchor (measured) | 0.111625 |
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
Taskdef: capacity-limited Switch-style top-1 routed fused-MoE (SURFACE fwdbwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). Family
moe_routed: the routing decision -- scores, selection, per-expert capacity
clipping, gate -- is INSIDE the graded surface, and gradients flow through
the router. This member is marked hard:true.

    fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D], capacity) -> y[T,D]
    logits = x @ router_w; p = softmax(logits); e* = argmax_e p (ties to the
    LOWER expert index); for each expert e keep the top-`capacity` candidates
    of {t: e*[t]==e} by p[t,e] (ties LOWER token index); excess DROPPED.
    y[t] = p[t, e*] * f(e*, x[t]) if kept else 0. SwiGLU experts; fp32
    arithmetic, one final cast. Gradients graded: dx, drouter_w, dw1, dw2.
    reference.py is the definition.

WHY HARD (KernelBench-hard lever, adapted to a ROUTED family): the correct
kernel must implement per-expert capacity clipping structurally -- a
data-dependent per-token drop mask that decouples selection (top-1 by softmax)
from delivery (top-`capacity` within each expert bucket). NO torch or vendor
builtin covers "top-1-then-per-bucket-top-C-with-index-tie-break-and-zero-out-
overflow-with-router-gradient-only-through-kept-tokens" as a single primitive;
the token-level DROP forces the backward path to be selective in a way plain
grouped-GEMM MoE code paths never are. bf16 router-arithmetic drift moves the
selection boundary AND the capacity boundary; a capacity-boundary flip changes
which of two ties is kept vs dropped, silently corrupting dw1[e*] and drouter_w
on the graded tolerance.

UNCALIBRATED SEAMS (must be re-measured by a golden/controls run in this
folder before freeze): TOL, OUT_TOL, TARGET_FRACTION_OF_SOTA, _MARGIN, the
hidden shape sweep, _SOTA_ROUTE, and the negative-control anchor strings.
Values below are informed placeholders carried from the sibling calibrated
moe_routed task.

Selection-stability design (load-bearing): TWO discontinuities must survive
selection changes -- the top-1 argmax AND the per-expert top-`capacity` cut.
make_inputs enforces a decision margin on BOTH boundaries via an fp64 shadow
softmax over the dtype-cast tensors: rows whose top-1/top-2 gap is positive
but below _MARGIN are redrawn, and per-expert candidate lists whose C/(C+1)
p-gap is positive but below _MARGIN are redrawn on their two boundary tokens.
Exact ties (gap == 0) are legal and manufactured by tie-forcing draws.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `9dbb31dbe52fbb00` |
| `kernel.py` | `f2da2a546d2749b1` |
| `bench.py` | `5ff1ea5048ee9e10` |
| `instruction.md` | `db58d3fe0d598b19` |
| `hidden_shapes.json` | `d06e6ace4d6a4fba` |
| `solution/oracle_kernel.py` | `54bad3f21017463f` |

- **seed_base:** `90741`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | T=8192 D=2048 F=1408 E=64 capacity=160 | float32, bfloat16 |
| `g2` | T=4096 D=2048 F=704 E=128 capacity=40 | float32, bfloat16 |
| `g3` | T=16384 D=1024 F=1408 E=32 capacity=600 | float32, bfloat16 |

- **forbidden symbols (7):** `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 b173dc269c4b528e50f1815d7b6ee405819dac3b0022ede31c6e35a20ff21149
FORGE-CANARY-SLOT-1 df6844fd145ff8951e4f4e42b79e0fad590894f957d98730c58f0d3a6be2b8bd
FORGE-CANARY-SLOT-2 e0cae78f54d8feaa8bfa13a59ac9973cfdc0a2ff21bbdde6670a57e046a4911b
FORGE-CANARY-SLOT-3 edc38a4f5fb6503f8d211c7cd378c974325417c475aa6aa7329c0e106497bf0a
FORGE-CANARY-END
