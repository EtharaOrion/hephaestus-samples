# truth.md - Golden Solve Path

## Task: hephaestus `linear_attn_decode_gqa_delta_decode_fwd` (anchor `torch.compile(delta_step + grouped_query_read) decode loop`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward` | **Entry:** `gqa_delta_decode` | **Editable:** `kernel.py`
- **Objective:** implement gqa_delta_decode in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is the DECODE regime of the DELTA RULE run in the GROUPED-QUERY
regime: `Hq` query heads share `Hkv` state heads with a group ratio `G = Hq /
Hkv` (Hq is a multiple of Hkv). The recurrent state lives on the SMALLER (kv)
head axis; every query head reads and delta-writes the state of its group's kv
head. This is the standard grouped-query attention topology carried into a
recurrent (linear-attention) operator, where the state SHARING across query
heads is the source of extra structure -- an ordinary per-query-head kernel
either wastes GxHkv state slots (memory-quadratic in G) or races G query heads
writing to the same state without a designed accumulation order.

For each batch, kv-head hk and its G query heads {hq = hk*G, ..., hk*G+G-1},
with state S[hk] of shape [K, V] starting from state0[b, hk]:

    S[hk]_t = S[hk]_{t-1} + k_t v_t^T                              (write, no gate)
    pred_t  = (S[hk]_{t-1})^T k_t                                  (pre-update pred)
    delta_t = (v_t - pred_t) * beta_t                              (per-step scalar)
    S[hk]_t = S[hk]_{t-1} + k_t delta_t^T                          (delta write)
    o_t[hq] = (S[hk]_t)^T (q_t[hq] / sqrt(K))         for every hq in the group

The delta rule (with the pre-update prediction) is the OPERATOR's definition; the
state is per-kv-head and every query head in the group reads AFTER the delta
write. q is [B, S, Hq, K]; k, v are [B, S, Hkv, {K, V}]; beta is [B, S, Hkv].
state0 is [B, Hkv, K, V] float32. BOTH outputs are graded: o [B, S, Hq, V] in
v's dtype and state_out [B, Hkv, K, V] float32.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes gqa_delta_decode
  - establishes: gqa_delta_decode is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: gqa_delta_decode agrees with reference.py and is the baseline to beat
  - survives: the published shapes in bench.py are smaller than and different from the graded shapes

**Step 3 (G1).** read the production anchor to understand the fast formulation, without importing or calling it
  - establishes: the decomposition that makes the operator fast is understood
  - survives: a forbidden-symbol scan over the submitted source with comments stripped, so citing a symbol is legal and calling it is not

**Step 4 (G2).** write the kernel and iterate against bench.py, holding the reference semantics fixed
  - establishes: the output still agrees with reference.py on the hidden configurations
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
| starter fraction of anchor (measured) | 1.0 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| bfloat16 tolerance | atol 0.02, rtol 0.03125 |
| float32 tolerance | atol 0.002, rtol 0.002 |
| comparison | within dtype tolerance |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes gqa_delta_decode. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
| `G1` forbidden_symbols | the submission neither imports nor calls the production kernel or any equivalent library implementation, scanned over the source with comments stripped so that citing a symbol stays legal and calling it does not. |
| `G2` correctness | the output agrees with reference.py on every hidden configuration, on the same randomly scaled draw that is timed, so no branch can tell the checked distribution from the timed one. |
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
Taskdef: decode-regime GROUPED-QUERY DELTA RULE (SURFACE fwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). HARD LEVER:
`Hq` query heads share `Hkv` state heads with a group ratio G = Hq / Hkv > 1
(the standard GQA topology, carried into a recurrent operator where the state
is per-kv-head and every query head in a group reads AFTER a single delta write
per step). No fla / mamba_ssm fused_recurrent kernel accepts Hq != Hkv on a
delta-rule operator; the shared-state accumulation order across a group is the
new correctness constraint no library primitive is designed for.

reference.py is the definition of correctness and is CPU-verified (includes a
head-swap invariance check for shared-state and a G=1 degenerate check against
the plain delta rule). TOL and STATE_TOL are INHERITED from the calibrated
gdn_decode/KDA family and MUST be re-confirmed by a measure/golden/controls run
in THIS folder before freeze.

Operator (reference.py is the definition; summary):
    gqa_delta_decode(q, k, v, beta, state0) -> (o, state_out)
    Per (batch, kv-head hk) with state S[hk] starting from state0[b, hk]:
      pred_t  = S[hk]_{t-1}^T k_t
      delta_t = (v_t - pred_t) * beta_t
      S[hk]_t = S[hk]_{t-1} + k_t delta_t^T
      o_t[hq] = S[hk]_t^T (q_t[hq] / sqrt(K))    for every hq in the group
    q [B, S, Hq, K]; k, v [B, S, Hkv, {K, V}]; beta [B, S, Hkv].
    BOTH outputs graded: o [B, S, Hq, V] input dtype, state_out [B, Hkv, K, V]
    float32.

Anchor: there is NO production kernel for a GQA-topology delta-rule decode
(fla's fused_recurrent_delta_rule requires Hq == Hkv and would broadcast query
heads onto per-query-head states, which is a DIFFERENT operator). Timing
denominator is a torch.compile of a fused delta step + a broadcast query read
(sota() below). VENDOR_LIB_PACKAGE empty: structural shingle scan disabled.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `4131d76a24bbdedf` |
| `kernel.py` | `be66f758f20a1b8c` |
| `bench.py` | `8c1af2013708537f` |
| `instruction.md` | `2246714da13e646d` |
| `hidden_shapes.json` | `8765881c902a7a01` |
| `solution/oracle_kernel.py` | `db678550fecfd7c9` |

- **seed_base:** `61217`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=128 S=128 Hq=32 Hkv=8 K=128 V=128 | bfloat16 |
| `g2` | B=256 S=64 Hq=32 Hkv=4 K=64 V=128 | bfloat16, float32 |
| `g3` | B=64 S=256 Hq=16 Hkv=2 K=64 V=64 | bfloat16, float32 |

- **forbidden symbols (7):** `fla.ops`, `import fla`, `from fla`, `fused_recurrent_delta_rule`, `chunk_delta_rule`, `fused_recurrent_gated_delta_rule`, `fused_recurrent`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 2020c1c5f3297c1c30599fd51a06c953764929922a2bd474bf0a5e627cbf8ca8
FORGE-CANARY-SLOT-1 bfc12e636519eb56dfe1afba9612b77ddb42bdf99604dfc205fd53ba8b20a812
FORGE-CANARY-SLOT-2 d08683cffec923750b75f3b4e1f88b74cce99be4bc0907474dee958915faf3c2
FORGE-CANARY-SLOT-3 3fa9e5ac9e26368e5572621167a3d49453ab0411713300a41fdc249337c296ee
FORGE-CANARY-END
