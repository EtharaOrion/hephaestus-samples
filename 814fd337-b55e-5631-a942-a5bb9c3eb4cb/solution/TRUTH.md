# truth.md - Golden Solve Path

## Task: hephaestus `linear_attn_decode_int8_state_delta_decode_fwd` (anchor `torch.compile(fp32 delta step) + torch-side dequant/requant`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward` | **Entry:** `int8_state_delta_decode` | **Editable:** `kernel.py`
- **Objective:** implement int8_state_delta_decode in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is the DECODE regime of the DELTA RULE with the recurrent state
STORED in int8 (per-(batch, head) symmetric scale) at the boundary, computed in
float32 internally. The initial state is (state0_q [B, H, K, V] int8,
state0_scale [B, H] float32); the reference dequantizes state0 at entry, runs
the sequential fp32 scan for S steps, requantizes the final state to (state_out_q
int8, state_out_scale float32) at exit. The RECURRENCE runs in float32 so its
math is unchanged from the plain delta rule; the only observable change is the
QUANTIZATION at the output boundary, which is why the state comparison happens
on the DEQUANTIZED state (state_out_q * state_out_scale) at a looser tolerance
that accounts for the int8 round-trip (max relative error ~1/127).

    S_0    = state0_q * state0_scale                       (dequantize on entry)
    S_t    = S_{t-1} + k_t delta_t^T                       (delta = (v - S^T k)*beta)
    o_t    = S_t^T (q_t / sqrt(K))
    scale_out[b,h] = max(|S_S[b,h]|) / 127                 (per-(b,h) symmetric)
    state_out_q[b,h] = round(S_S[b,h] / scale_out[b,h])    (clamped to [-127,127])

BOTH tensor outputs graded: o [B, S, H, V] in v's dtype; state_out_q int8 and
state_out_scale float32 compared jointly via the dequantized state.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes int8_state_delta_decode
  - establishes: int8_state_delta_decode is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: int8_state_delta_decode agrees with reference.py and is the baseline to beat
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
| `G0` trusted_entrypoint | the submitted module imports and exposes int8_state_delta_decode. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: INT8-STATE delta rule decode (SURFACE fwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). HARD LEVER:
low-precision (int8) state storage with fp32 accumulation. The recurrent state
crosses the operator boundary as (state_q int8 [B, H, K, V], state_scale float32
[B, H]) -- per-(batch, head) symmetric quantization; the recurrence itself runs
in float32 inside the operator. No fla / mamba_ssm fused_recurrent kernel
accepts an int8 initial state or emits an int8 final state; every published
kernel round-trips a full fp32 state, which is roughly 4x the boundary traffic
of the int8 form.

reference.py is the definition of correctness and is CPU-verified (includes a
symmetric-quantization round-trip check, a zero-state edge case, and an
fp32-scan agreement check that proves the fp32 interior is unchanged).

Operator (reference.py is the definition; summary):
    int8_state_delta_decode(q, k, v, beta, state0_q, state0_scale)
      -> (o, state_out_q, state_out_scale)
    S_0 = state0_q * state0_scale ; S_t = S_{t-1} + k_t delta_t^T
    (delta = (v - S^T k) beta) ; o_t = S_t^T (q_t / sqrt(K)) ; final state
    per-(batch, head) symmetric int8 with scale = max(|S|) / 127.

Comparison override: state is compared on the DEQUANTIZED state
(state_q * state_scale) at the family's float32 state tolerance PLUS a
quantization slack of state0_scale / 2 (per-(batch, head)), because the int8
round-trip is worth up to 1/2 LSB of the per-(batch, head) scale.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `03325aa85605543f` |
| `kernel.py` | `2d988168a18c2ae6` |
| `bench.py` | `95e42e77cfd88340` |
| `instruction.md` | `83f4f44283f41015` |
| `hidden_shapes.json` | `ada84969c456ff5c` |
| `solution/oracle_kernel.py` | `39d339c0fa374803` |

- **seed_base:** `61559`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=128 S=128 H=8 K=128 V=128 | bfloat16 |
| `g2` | B=256 S=64 H=8 K=64 V=128 | bfloat16, float32 |
| `g3` | B=64 S=256 H=16 K=64 V=64 | bfloat16, float32 |

- **forbidden symbols (14):** `fla.ops`, `import fla`, `from fla`, `fused_recurrent_delta_rule`, `chunk_delta_rule`, `fused_recurrent_gated_delta_rule`, `chunk_gated_delta_rule`, `fused_recurrent`, `torch.quantization`, `quantize_per_tensor`, `quantize_per_channel`, `bitsandbytes`, `tensorrt`, `torchao.quantization`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 613d1dd8b0c1d7e4655d28fa89c2ca836cfd992893236cc41d5ff73ed1dd6276
FORGE-CANARY-SLOT-1 ddedab146e135cbfe7af9106f7c07ab4ade9ddcf30e449923b69e83b565b3b64
FORGE-CANARY-SLOT-2 7bc132e9a66829b35dafd53d4e676a93033fe764f867322eeba98aca31219906
FORGE-CANARY-SLOT-3 f35bb1bcf447de1931c459f34599ea1c809dc379dda80dc035dae5c3a86de873
FORGE-CANARY-END
