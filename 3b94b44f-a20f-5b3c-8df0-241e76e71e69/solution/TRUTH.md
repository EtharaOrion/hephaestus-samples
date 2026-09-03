# truth.md - Golden Solve Path

## Task: hephaestus `moe_routed_shared_expert_routed_topk_fwdbwd` (anchor `torch._grouped_mm`)

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
This member is the SHARED-plus-ROUTED composition (DeepSeek/Qwen-MoE style):
an always-on DENSE shared expert plus a softmax-routed top-K sparse expert
pool, summed together. The routing decision (softmax + top-K + p_sel/sum
combine, mixtral-style) lives INSIDE the operator; the shared path is dense
and unconditional. SIX gradients are graded.

    fused_moe(x, router_w, w1, w2, w1_s, w2_s, K) -> y

    x:        [T, D]     tokens (float32 or bfloat16)
    router_w: [D, E]     router projection, same dtype as x
    w1:       [E, D, 2F] routed expert input projections; :F gate, F: up
    w2:       [E, F, D]  routed expert output projections
    w1_s:     [D, 2Fs]   shared expert input projection; :Fs gate, Fs: up
    w2_s:     [Fs, D]    shared expert output projection
    K:        int        experts selected per token from the routed pool, 1..E
    y:        [T, D]     same dtype as x

Semantics, exactly (all arithmetic in float32; inputs are cast up, only the
final y is cast back to the input dtype):

    # routed contribution
    logits = x @ router_w                              # [T, E]
    p      = softmax(logits, dim=-1)                   # [T, E]
    sel    = top-K experts per token by p, ties broken by LOWER expert index
             first; -0.0 and +0.0 compare equal
    w      = p[t, sel] / p[t, sel].sum(-1)             # p_sel/sum combine
    y_r[t] = sum_k w[t, k] * f_r(sel[t, k], x[t])
    f_r(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]

    # shared contribution (always-on, dense, unconditional)
    y_s[t] = (silu(x[t] @ w1_s[:, :Fs]) * (x[t] @ w1_s[:, Fs:])) @ w2_s

    # composed output
    y[t]   = y_r[t] + y_s[t]

  * Selection is by (p desc, index asc) via the int64-key trick; ties broken
    to the LOWER expert index; -0.0 canonicalised to +0.0 first.
  * The combine over routed experts uses the mixtral rule (p_sel/sum). A
    variant that drops the shared path or moves the shared path into the
    combine gate is a different operator and will fail G2.
  * SIX gradients are graded: dx (both paths), drouter_w (routed only),
    dw1, dw2 (routed), dw1_s, dw2_s (shared). Shared-path gradients into x
    are added to routed-path gradients into x; both contribute.
  * Experts no token selects receive exactly zero weight gradients.
  * The same expert never appears twice for one token (top-K of E distinct).

Correctness is defined as agreement with this function under the disclosed
per-dtype tolerances -- and where prose and this code disagree, the code wins.

The implementation below is deliberately simple and deliberately slow: a
python loop over routed experts with boolean-mask gathers, plus a dense
shared-expert path. It is autograd-capable end to end, which is how the
harness obtains the reference gradients.
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
| target fraction of anchor | 0.26 |
| starter fraction of anchor (measured) | 0.072562 |
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
| rubric judge | a cross-family LLM reads `tests/rubrics.jsonl` against the trajectory. All 11 gate. |

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
Taskdef: shared-expert routed top-K SwiGLU fused-MoE (SURFACE fwdbwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). Family
moe_routed: the routing decision is INSIDE the graded surface, and gradients
flow through the router. This member is marked hard:true. The routed path is
softmax + top-K + p_sel/sum (mixtral rule); the shared path is a dense,
always-on SwiGLU expert; y = y_routed + y_shared.

    fused_moe(x[T,D], router_w[D,E], w1[E,D,2F], w2[E,F,D],
              w1_s[D,2Fs], w2_s[Fs,D], K) -> y[T,D]

SIX graded gradients: dx, drouter_w, dw1, dw2, dw1_s, dw2_s. Shared-path
gradients into x are added to routed-path gradients into x.

WHY HARD: two arithmetic regimes coexist inside one op -- a DENSE shared
path that never batches with the sparse routed path (their tiles are
different shapes and the shared path always touches every token), plus the
routed grouped-GEMM with a p_sel/sum combine backward. A candidate that
kernel-fuses only the routed side leaves the shared matmuls in framework
operators and blows past the 60% adoption floor; a candidate that treats
both paths as one grouped-GEMM misses the shared path's dense character.
The composition also doubles the weight-gradient reduction surface (routed
per-expert reductions PLUS full-T reductions on w1_s, w2_s), which stacks
fp32-router-gradient sensitivity across two independent operand chains.

UNCALIBRATED SEAMS (re-measure before freeze): TOL, OUT_TOL,
TARGET_FRACTION_OF_SOTA, _MARGIN, hidden shape sweep, _SOTA_ROUTE, negative-
control anchor strings.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `d9d41ddde2655a54` |
| `kernel.py` | `0ecd91a2113fd7e5` |
| `bench.py` | `92ca5a70423d4563` |
| `instruction.md` | `7f54d6d76f116284` |
| `hidden_shapes.json` | `6a93aa688102a3cb` |
| `solution/oracle_kernel.py` | `b053ccc656176714` |

- **seed_base:** `91453`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | T=8192 D=2048 F=1408 Fs=1408 E=64 K=8 | float32, bfloat16 |
| `g2` | T=4096 D=2048 F=704 Fs=2816 E=160 K=6 | float32, bfloat16 |
| `g3` | T=16384 D=1024 F=1408 Fs=704 E=32 K=4 | float32, bfloat16 |

- **forbidden symbols (7):** `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 3ba11a0060fdbdbd191746ee52a4bef4e373ae78ef266af3192272c554d43c70
FORGE-CANARY-SLOT-1 dad2583c5cdffe95684adf9ca848bd9c00fe687400e563d18eebf627d6a77325
FORGE-CANARY-SLOT-2 2b174defda600ed37f94172c8a2ea1a97e1059c6b89f84081fbd0587c9593690
FORGE-CANARY-SLOT-3 799409199f4e0d6101cc6bd0efc16a073d0d4b5a9e4ac512184e3ff35e067310
FORGE-CANARY-END
