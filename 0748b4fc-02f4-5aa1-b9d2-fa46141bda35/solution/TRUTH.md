# truth.md - Golden Solve Path

## Task: hephaestus `linear_attn_decode_rwkv7_channel_delta_decode_fwd` (anchor `torch.compile(per-channel decay + delta write + post-read) decode loop`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward` | **Entry:** `rwkv7_channel_delta_decode` | **Editable:** `kernel.py`
- **Objective:** implement rwkv7_channel_delta_decode in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is a hybrid decode-regime linear-attention step that fuses two
per-step levers no single existing library kernel exposes together: a PER-CHANNEL
log-forget gate on the key axis of the [K, V] state (GLA-style) AND a per-step
scalar write-strength `beta` on a delta-corrected write (delta-rule-style).
Neither fla.ops.gla nor fla.ops.gated_delta_rule handles both at once: GLA has no
delta correction and no beta; the gated delta rule has no per-channel gate.

Per batch, head and step, with state `S` of shape `[K, V]` starting from state0:

    S_t         = diag(exp(g_t)) . S_{t-1}                (per-channel decay)
    pred_t      = S_t^T k_t                                (pre-write read)
    delta_t     = (v_t - pred_t) * beta_t                  (scalar write strength)
    S_t         = S_t + k_t delta_t^T                      (delta write)
    o_t         = S_t^T (q_t / sqrt(K))                    (read AFTER delta write)

q, k are [B, S, H, K]; v is [B, S, H, V]; g is [B, S, H, K] float32; beta is
[B, S, H]; state0 is [B, H, K, V] float32. BOTH outputs are graded: o
[B, S, H, V] in v's dtype and state_out [B, H, K, V] float32.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes rwkv7_channel_delta_decode
  - establishes: rwkv7_channel_delta_decode is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: rwkv7_channel_delta_decode agrees with reference.py and is the baseline to beat
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
| `G0` trusted_entrypoint | the submitted module imports and exposes rwkv7_channel_delta_decode. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: PER-CHANNEL-DECAY + BETA-DELTA linear-attention decode (SURFACE fwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). HARD LEVER:
fuse a PER-CHANNEL log-forget gate on the key axis of the [K, V] state
(GLA-style) AND a per-step scalar write-strength `beta` on a delta-corrected
write (delta-rule-style) in the SAME step. No single library kernel exposes
both at once: fla.ops.gla has no delta correction and no beta; fla.ops.
gated_delta_rule has no per-channel gate. The RWKV-7 layer family combines them
in the model, but not as a single fused_recurrent kernel.

reference.py is the definition of correctness and is CPU-verified (includes a
triple-loop scan agreement check, a beta==0 degeneracy to GLA, and a g==0
degeneracy to the plain delta rule). TOL and STATE_TOL are INHERITED from the
calibrated gdn_decode/KDA family and must be re-confirmed at calibration.

Operator (reference.py is the definition; summary):
    rwkv7_channel_delta_decode(q, k, v, g, beta, state0) -> (o, state_out)
    Per step: state = diag(exp(g_t)) . state ; pred = state^T k_t ;
    delta = (v_t - pred) * beta_t ; state = state + k_t delta^T ;
    o_t = state^T (q_t / sqrt(K)).
    q, k [B, S, H, K]; v [B, S, H, V]; g [B, S, H, K] fp32; beta [B, S, H];
    state0 [B, H, K, V] fp32. BOTH outputs graded.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `cddd0492d6169557` |
| `kernel.py` | `8734bf60b8ce0f92` |
| `bench.py` | `f4e0eaace76ec622` |
| `instruction.md` | `79d47539d11bd894` |
| `hidden_shapes.json` | `7ed1e1706317ba99` |
| `solution/oracle_kernel.py` | `dd92313c30f997af` |

- **seed_base:** `61447`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=128 S=128 H=8 K=128 V=128 | bfloat16 |
| `g2` | B=256 S=64 H=8 K=64 V=128 | bfloat16, float32 |
| `g3` | B=64 S=256 H=16 K=64 V=64 | bfloat16, float32 |

- **forbidden symbols (12):** `fla.ops`, `import fla`, `from fla`, `fused_recurrent_gla`, `chunk_gla`, `fused_recurrent_gated_delta_rule`, `chunk_gated_delta_rule`, `fused_recurrent_delta_rule`, `chunk_delta_rule`, `fused_recurrent_rwkv7`, `chunk_rwkv7`, `fused_recurrent`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 6a93dd99f533b5c6800885bdc91abae3577c184ec6ac7d5d1538ba393cb1586b
FORGE-CANARY-SLOT-1 51e9e8fc2683454d7f16402bbe50e1981b07ea084445591440ab51fa1846cdb2
FORGE-CANARY-SLOT-2 8f204779bb9fc59e65b62fc9125a4a4eeec5d12fbc9215ea326b216233640f7c
FORGE-CANARY-SLOT-3 cb4d5b3ad630abb8290b0578178e274afdcff3fc29cb5a83ec2322d6cf4fb6df
FORGE-CANARY-END
