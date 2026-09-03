# truth.md - Golden Solve Path

## Task: hephaestus `linear_attn_decode_mamba2_ssd_decode_fwd` (anchor `torch.compile(selective_step) decode loop`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward` | **Entry:** `mamba2_ssd_decode` | **Editable:** `kernel.py`
- **Objective:** implement mamba2_ssd_decode in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is the DECODE regime of the MAMBA-2 SSD (State-Space Duality) selective
scan -- one selective state-space step per token, run from an explicit initial state.
It is the HARDEST task in the linear_attn_decode family: the state is a full
[P (head dim) x N (state dim)] matrix per head AND every step consumes five selective
parameter streams (dt, A, B, C, D), so it carries the most per-step traffic and
arithmetic of any operator here. For a single head, with state S in R^{P x N}:

    dt_t = softplus(dt_raw_t + dt_bias)            # positive time step, scalar per head
    A    = -exp(A_log)                             # scalar per head, strictly negative
    dA_t = exp(dt_t * A)                           # scalar per head, in (0, 1)
    S_t  = dA_t . S_{t-1} + (dt_t * x_t) (B_t)^T   # outer product [P x N]
    y_t  = S_t C_t + D * x_t                       # contract over N, plus skip
    o_t  = y_t

This is the canonical Mamba-2 selective_state_update recurrence (see FLA's
fla/layers/mamba2.py decode path): A and dt and D are SCALAR PER HEAD, while B and C
are the per-head, per-step selective vectors of length N. There is no key/value delta
and no per-channel gate -- the whole state decays by the single data-dependent scalar
dA_t, the input x_t is discretized by dt_t before the rank-1 write against B_t, the
read is against C_t, and D provides an input skip. The code is the definition; where
any prose and this code disagree, the code wins.

Design choices, on purpose:

  * The recurrence starts from a caller-supplied `state0` ([B, H, P, N] float32),
    not from zero. A kernel that ignores it is wrong on every graded input.
  * BOTH outputs are graded: `o` ([B, S, H, P], x's dtype) at the input dtype's
    tolerance, and `state_out` ([B, H, P, N] float32) at a float32 tolerance.
  * Ungrouped multi-head: one (B, C) pair per head (n_groups == num_heads). The
    default Mamba-2 time-step clamp is (0, inf), i.e. a no-op on softplus output, so
    it is omitted here.
  * Forward only. No gradient of any input is ever requested.

Everything is accumulated in float32 regardless of the input dtype. There is no
production FLA fused_recurrent kernel for this operator; the timing denominator is a
torch.compile of a dense sequential scan (see taskdef.sota).
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes mamba2_ssd_decode
  - establishes: mamba2_ssd_decode is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: mamba2_ssd_decode agrees with reference.py and is the baseline to beat
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
| `G0` trusted_entrypoint | the submitted module imports and exposes mamba2_ssd_decode. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: decode-regime Mamba-2 SSD selective state update (SURFACE fwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). The HARDEST
member of the linear_attn_decode family: the state is a full [P (head dim) x
N (state dim)] matrix per head AND every step consumes five selective parameter
streams (dt, A, B, C, D), so it carries the most per-step traffic, arithmetic
and register pressure of any operator here. reference.py is the definition of
correctness and is CPU-verified (2026-08-17, incl. an S=1 closed form, a
triple-loop scan and a dA-in-(0,1) check). TOL and the shape sweep are INHERITED
from the calibrated gdn_decode/KDA family (GPU-measured there); the softplus /
selective numerics may need a looser bf16 knee and MUST be re-confirmed by a
measure/golden/controls run in THIS folder before freeze. oracle_kernel.py is an
UNVERIFIED from-scratch megakernel; the negative-control anchor strings target
its bytes and kernel.py's bytes as authored here.

Operator (reference.py is the definition; summary):
    mamba2_ssd_decode(x, dt, A_log, B, C, D, dt_bias, state0) -> (o, state_out)
    Canonical Mamba-2 selective_state_update, one token per step:
      dt_t = softplus(dt_raw_t + dt_bias) ; A = -exp(A_log) ; dA = exp(dt_t*A)
      S_t  = dA . S_{t-1} + (dt_t*x_t) B_t^T ; y_t = S_t C_t + D*x_t
    A, dt, D scalar per head; B, C length-N per head-step; P = head dim.
    starting from state0 [B, H, P, N] float32; ungrouped (n_groups == num_heads).
    BOTH outputs graded: o [B,S,H,P] in x's dtype, state_out [B,H,P,N] float32.

Anchor: there is NO production FLA fused_recurrent kernel for this operator, so
the TIMING DENOMINATOR is a torch.compile of a fused selective-step run in a
decode loop (sota() below) -- the strongest reproducible torch baseline for the
Mamba-2 decode step. reference.py (a dense sequential scan) defines correctness;
the compiled step and reference.py are the same math (verified equal in eager at
authoring on tiny CPU shapes via reference.py's __main__). VENDOR_LIB_PACKAGE is
empty: with no library kernel to compare against, the structural shingle scan is
skipped (recorded available:false) and only the FORBIDDEN symbol scan runs.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `4ad80a3f1dedb959` |
| `kernel.py` | `35123120e0fd1269` |
| `bench.py` | `82300da8f71a7d90` |
| `instruction.md` | `1d9b2de421a72e7a` |
| `hidden_shapes.json` | `6dd8cf198d301107` |
| `solution/oracle_kernel.py` | `11bfa791f60f3f18` |

- **seed_base:** `62771`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=128 S=128 H=32 P=64 N=128 | bfloat16, float32 |
| `g2` | B=256 S=64 H=16 P=64 N=64 | bfloat16, float32 |
| `g3` | B=64 S=256 H=8 P=64 N=128 | bfloat16, float32 |

- **forbidden symbols (8):** `fla.ops`, `import fla`, `from fla`, `selective_state_update`, `selective_scan`, `mamba_chunk_scan`, `mamba_ssm`, `mamba_split_conv1d_scan`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 e91c89c5b933e54ead0235cd801ea84738f4c76ff3e70d93ad31d8ec4d4da3b0
FORGE-CANARY-SLOT-1 7850bfa701dbab19a3499ffa9296d72b49359e91fb8cdc0eed9bed7f701ad10d
FORGE-CANARY-SLOT-2 541f7cad29753300209c4f5d955e762fa54c4f331a1e999d1bb17073d8ffcfe9
FORGE-CANARY-SLOT-3 d47c7437729060ca2301d0ea7a60dbc4ad9e0c8c02aafad5654d677d0f6d3194
FORGE-CANARY-END
