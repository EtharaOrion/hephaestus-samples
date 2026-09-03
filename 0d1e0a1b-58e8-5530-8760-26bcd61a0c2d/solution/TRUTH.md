# truth.md - Golden Solve Path

## Task: hephaestus `linear_attn_decode_speculative_gated_delta_decode_fwd` (anchor `torch.compile(gated_delta_step + masked commit snapshot) decode loop`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward` | **Entry:** `speculative_gated_delta_decode` | **Editable:** `kernel.py`
- **Objective:** implement speculative_gated_delta_decode in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is the SPECULATIVE-DECODE regime of the GATED DELTA RULE: on every
call the model proposes S candidate tokens per batch, the verifier accepts a
per-batch prefix of length `commit_len[b]` in [0, S], and the state advances
through those committed steps only. Outputs `o` are computed for ALL S positions
per batch (they are the speculative reads the acceptance check needs), but
`state_out[b]` is the state after step `commit_len[b] - 1` (or `state0[b]` when
commit_len[b] == 0). This is the standard tree-free linear speculative-decode
contract lifted onto a recurrent (linear-attention) operator, and no library
primitive implements it: fla's fused_recurrent_gated_delta_rule always advances
the state through every one of the S input steps.

For each batch b and head h, per step t:

    S_t = exp(g_t) * S_{t-1} + k_t delta_t^T   (with delta_t = (v_t - S_{t-1}^T k_t) beta_t)
    o_t = S_t^T (q_t / sqrt(K))                       (always computed for reads)
    state_out[b] = S_{commit_len[b] - 1}              (advance stops at commit)

The scalar gate `g_t` is applied BEFORE the delta write (family convention).
q is [B, S, H, K]; k, v are [B, S, H, {K, V}]; beta and g are [B, S, H];
commit_len is [B] int32 in [0, S]; state0 is [B, H, K, V] float32. BOTH outputs
graded: o [B, S, H, V] in v's dtype and state_out [B, H, K, V] float32.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes speculative_gated_delta_decode
  - establishes: speculative_gated_delta_decode is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: speculative_gated_delta_decode agrees with reference.py and is the baseline to beat
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
| `G0` trusted_entrypoint | the submitted module imports and exposes speculative_gated_delta_decode. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: SPECULATIVE-DECODE gated delta rule (SURFACE fwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). HARD LEVER:
multi-token speculative decode. On every call the model proposes S candidate
tokens per batch, the verifier accepts a per-batch prefix of length
`commit_len[b]` in [0, S], and the state advances through those committed steps
only -- BUT `o` is computed for every position (the speculative reads the
acceptance check needs). No library primitive implements this: fla's
fused_recurrent_gated_delta_rule always advances the state through every one
of its S input steps and has no per-batch commit horizon.

reference.py is the definition of correctness and is CPU-verified (includes
commit_len == 0, commit_len == S, and an `o` invariance-under-commit_len check
that proves the read/commit split is honored). TOL and STATE_TOL are INHERITED
from the calibrated gdn_decode/KDA family and must be re-confirmed at
calibration.

Operator (reference.py is the definition; summary):
    speculative_gated_delta_decode(q, k, v, g, beta, commit_len, state0)
      -> (o, state_out)
    S steps of the gated delta rule (decay BEFORE the delta write); o computed
    for every position from the SPECULATIVE state, state_out[b] snapshotted at
    step commit_len[b] - 1 (or state0[b] when commit_len[b] == 0). BOTH outputs
    graded: o at input dtype's tolerance, state_out at float32.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `c36235931ea8fb2c` |
| `kernel.py` | `ede878fe3444d221` |
| `bench.py` | `6e6c694f8506a2e5` |
| `instruction.md` | `d95613c3304d633b` |
| `hidden_shapes.json` | `9c1b78bbb132436a` |
| `solution/oracle_kernel.py` | `8746f5e893fc273d` |

- **seed_base:** `61331`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=128 S=128 H=8 K=128 V=128 | bfloat16 |
| `g2` | B=256 S=64 H=8 K=64 V=128 | bfloat16, float32 |
| `g3` | B=64 S=256 H=16 K=64 V=64 | bfloat16, float32 |

- **forbidden symbols (8):** `fla.ops`, `import fla`, `from fla`, `fused_recurrent_gated_delta_rule`, `chunk_gated_delta_rule`, `fused_recurrent_delta_rule`, `chunk_delta_rule`, `fused_recurrent`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 a8050f31c24844e711ed303ba9424c9974246d7a21b5f270201c1f0c1287336c
FORGE-CANARY-SLOT-1 e92e77de27b10fa804627db485af273d2d754dae99a2f31e9676c731431db03d
FORGE-CANARY-SLOT-2 45f8c3dfa575e54e1ac967d3e94cc9965dbc1086023f4f3c310a15a51d08a6b7
FORGE-CANARY-SLOT-3 e3613ebedb13d998f3af92885ac6b98ce6692de3cd3fc916d31117a71df07768
FORGE-CANARY-END
