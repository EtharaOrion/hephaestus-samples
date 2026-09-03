# truth.md - Golden Solve Path

## Task: hephaestus `linear_attn_decode_paged_gla_decode_fwd` (anchor `torch.compile(gla_step) decode loop with dense fancy-index gather`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward` | **Entry:** `paged_gla_decode` | **Editable:** `kernel.py`
- **Objective:** implement paged_gla_decode in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is the DECODE regime of GATED LINEAR ATTENTION with a PAGED recurrent
state pool. Instead of a contiguous per-batch state tensor, the recurrent state of
every sequence lives in a shared pool `paged_state[P, H, K, V]` (P >= B), and a
`page_table[B]` int32 vector tells which physical page each batch owns for this
call. The gather is non-contiguous by design (page_table is a random permutation of
distinct page ids on the graded distribution), so a kernel that assumes a stride-1
per-batch state address is wrong on every graded input.

Every step of every batch is the plain GATED LINEAR ATTENTION recurrence, per-channel
forget gate on the key-row axis of the state, applied BEFORE the outer product write:

    S_t = diag(exp(g_t)) . S_{t-1} + k_t v_t^T
    o_t = S_t^T (q_t / sqrt(K))

The initial state for batch b is the page `paged_state[page_table[b]]` (float32).
The returned final state is a DENSE [B, H, K, V] float32 tensor holding the per-batch
final state (NOT the mutated pool -- the pool is read-only in this operator). Both
outputs are graded.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes paged_gla_decode
  - establishes: paged_gla_decode is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: paged_gla_decode agrees with reference.py and is the baseline to beat
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
| `G0` trusted_entrypoint | the submitted module imports and exposes paged_gla_decode. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: decode-regime PAGED GATED LINEAR ATTENTION (SURFACE fwd).

AUTHORED DRAFT (status 'authored-draft: needs GPU calibration'). HARD LEVER:
the recurrent state lives in a shared pool [P, H, K, V] and a per-batch
`page_table[B]` int32 vector maps logical->physical page. The page table is a
random permutation of distinct page ids on the graded distribution, so the
per-batch state gather is NON-CONTIGUOUS and a kernel that assumes a stride-1
per-batch state address is wrong on every graded input. This lever mirrors the
core primitive of KernelBench-hard's `03_paged_attention` -- gathered
state/KV -- adapted to a recurrent decode operator.

reference.py is the definition of correctness and is CPU-verified. TOL and
STATE_TOL are INHERITED from the calibrated gdn_decode/KDA family; they must be
re-confirmed by a measure/golden/controls run in THIS folder before freeze.

Operator (reference.py is the definition; summary):
    paged_gla_decode(q, k, v, g, page_table, paged_state) -> (o, state_out)
    S sequential steps of GLA per batch b, starting from paged_state[page_table[b]]
    (per-channel forget gate on the key-row axis, plain outer-product write):
      S_t = diag(exp(g_t)) . S_{t-1} + k_t v_t^T ; o_t = S_t^T (q_t / sqrt(K))
    paged_state [P, H, K, V] float32 is READ-ONLY (never mutated). state_out is a
    DENSE [B, H, K, V] float32 tensor of per-batch final states.

Anchor: there is NO production kernel for a paged-state linear-attention decode
(fla has fused_recurrent_gla but no page-table variant), so the TIMING
DENOMINATOR is a torch.compile of a fused selective step reused across the S-step
decode loop (sota() below), with the initial state materialized dense by fancy
indexing -- the strongest reproducible torch baseline. VENDOR_LIB_PACKAGE empty:
the structural shingle scan is skipped (recorded available:false) and only the
FORBIDDEN symbol scan runs.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `8bbed019b14b6689` |
| `kernel.py` | `6dd93dd4a16a05db` |
| `bench.py` | `56c9650ce6feaf24` |
| `instruction.md` | `f9a96db9bc7e11d3` |
| `hidden_shapes.json` | `a2ecc990a601f52b` |
| `solution/oracle_kernel.py` | `994bef8e7497c3b7` |

- **seed_base:** `61103`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=128 S=128 H=8 K=128 V=128 P=512 | bfloat16 |
| `g2` | B=256 S=64 H=8 K=64 V=128 P=1024 | bfloat16, float32 |
| `g3` | B=64 S=256 H=16 K=64 V=64 P=256 | bfloat16, float32 |

- **forbidden symbols (9):** `fla.ops`, `import fla`, `from fla`, `fused_recurrent_gla`, `chunk_gla`, `fused_recurrent`, `paged_attention`, `vllm.attention`, `flash_attn_with_kvcache`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 835fa0061c214984a01a76f3a06dbedd4d863b282964d54b8dc3e00f4aeee3f1
FORGE-CANARY-SLOT-1 b2e9295f205f11f4cbf7fb5808ed375d1bdfd797becab32e76a87104eb1f1dee
FORGE-CANARY-SLOT-2 77935e574cff55c4ecc99d63886b3fccb324d9daa7b48c6454fb1f7d2d8b9903
FORGE-CANARY-SLOT-3 cf6170bfc4ad891d6d0fb6a82a5b5669a0887be24c088ab861e09de947293fab
FORGE-CANARY-END
