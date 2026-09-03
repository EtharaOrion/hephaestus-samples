# truth.md - Golden Solve Path

## Task: hephaestus `selection_topk_ragged_fwd` (anchor `pad-to-max(-inf) + torch.topk`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward` | **Entry:** `topk_ragged` | **Editable:** `kernel.py`
- **Objective:** implement topk_ragged in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is RAGGED top-k: a descending top-k taken inside each row of a
flat tensor whose row boundaries are given by an int64 prefix-sum:

    topk_ragged(x[NNZ], seg_offsets[R+1] int64, k)
        -> (values[R, k], indices[R, k])

Semantics, exactly:

  * `seg_offsets` is a monotone non-decreasing int64 vector of length R+1
    with `seg_offsets[0] == 0` and `seg_offsets[R] == NNZ`. Row r consists
    of the flat positions in `[seg_offsets[r], seg_offsets[r+1])` and has
    length L_r = seg_offsets[r+1] - seg_offsets[r].
  * Every row is guaranteed to have L_r >= k by construction; row lengths
    are ARBITRARY (different from topk_segmented, which uses one fixed
    length S per row).
  * For row r, `indices[r]` holds the GLOBAL positions in [0, NNZ) of the
    k largest values within that row, ordered DESCENDING by value, ties
    broken by LOWER ORIGINAL GLOBAL INDEX first, `-0.0 == +0.0` one tie
    group. Because the row occupies a contiguous slice, global-order and
    local-order agree inside a row, so the tie-break is unambiguous.
  * `values[r, :] = x.gather(0, indices[r])`, bit for bit, in the INPUT
    dtype. No arithmetic is performed on the values.
  * `indices` is int64.
  * Inputs are guaranteed NaN-free; `-inf`/`+inf` are legal.

There is no single library builtin for ragged top-k. The timing anchor
pads all rows to the maximum length with `-inf` and calls `torch.topk`;
that is a denominator only, its `-inf` fill collides with legitimately
`-inf` valid entries and its tie-break can differ, so it is never used
for correctness. This reference is the definition.

The implementation slices each row and runs a stable descending argsort
over that slice. Sequential; no fused kernel. Deliberately simple.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes topk_ragged
  - establishes: topk_ragged is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: topk_ragged agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.9535 |
| starter fraction of anchor (measured) | 0.883725 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| float32 tolerance | atol 0.0, rtol 0.0 |
| bfloat16 tolerance | atol 0.0, rtol 0.0 |
| comparison | exact equality |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes topk_ragged. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: ragged descending top-k with variable-length segments (SURFACE fwd).

Selection family. Everything category-specific the generic verifier consumes
lives here (protocol: seed/forge/core/taskdef_api.py).

Operator semantics (reference.py is the definition; this is a summary):
    topk_ragged(x[NNZ], seg_offsets[R+1] int64, k)
        -> (values[R, k], indices[R, k] int64)
    x is a 1D flat tensor; seg_offsets is an int64 prefix-sum: row r covers
    the flat slice [seg_offsets[r], seg_offsets[r+1]). Row lengths are
    ARBITRARY (this is the variable-length segments hardness lever, distinct
    from topk_segmented's fixed length S). Every row L_r >= k.
    Descending top-k inside each row; ties broken by LOWER original GLOBAL
    index first (global and local order agree inside a row); -0.0 == +0.0
    one tie group; values are BIT-EXACT gathers of the input element; indices
    are GLOBAL positions in [0, NNZ); int64. EXACT equality; `compare`
    overrides the dense walk.

Hard lever (spec.yaml justifies at length): NO single torch/vendor builtin
computes ragged top-k. The natural denominator pads all rows to the maximum
length with `-inf` and calls `torch.topk` -- correct only when no valid
entry is itself `-inf`, and it wastes device memory in proportion to the
length skew (a batch with one long row and many short ones spends most of
its bandwidth on padding). A from-scratch kernel must extract each row's
top-k from a variable-length slice, and it must NOT pay the pad-to-max
bandwidth tax the anchor pays, so the reachable fraction depends on the
skew of every batch.

Anchor: pad-to-max + torch.topk. TIMING DENOMINATOR ONLY. No vendorable
source tree: VENDOR_LIB_SUBDIRS empty; the FORBIDDEN symbol scan still runs.

Distribution invariance is made real exactly as for the rest of the family
(hidden scale + hidden distribution shape per draw; timed == checked).
The ROW-LENGTH DISTRIBUTION is also hidden per invocation (drawn inside
make_inputs from a uniform range specified by shape["Lmin"]/["Lmax"]) so a
candidate cannot precompute a fixed schedule. process_checks drops the
magnitude-branch lint and keeps no_grader_reference.

STATUS: authored-draft -- needs GPU calibration. TARGET_FRACTION_OF_SOTA
and the golden/controls numbers below are placeholders, NOT measured on
hardware.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `6d7ff6ec5a0438ec` |
| `kernel.py` | `ac8bcd6b9237c220` |
| `bench.py` | `0fab3c84f79e2b8e` |
| `instruction.md` | `9feb2b9607fe32b2` |
| `hidden_shapes.json` | `1d3946131d982142` |
| `solution/oracle_kernel.py` | `bfe7bd01413d53e5` |

- **seed_base:** `73501`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | R=128 Lmin=512 Lmax=4096 k=32 | float32, bfloat16 |
| `g2` | R=256 Lmin=1024 Lmax=8192 k=64 | float32, bfloat16 |
| `g3` | R=64 Lmin=2048 Lmax=65536 k=128 | float32, bfloat16 |

- **forbidden symbols (18):** `torch.topk`, `torch.sort`, `torch.argsort`, `torch.msort`, `torch.kthvalue`, `torch.unique`, `torch.median`, `torch.nanmedian`, `torch.quantile`, `torch.nanquantile`, `.topk`, `.sort`, `.argsort`, `.msort`, `.kthvalue`, `.unique`, `.median`, `.quantile`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 6c929f40021012d3fc5347a991e0c6fb274ffbfc9f7f24122fc4a34b5534855c
FORGE-CANARY-SLOT-1 4ff9f70f7f0cf0489a84a05379a2b13c29097216dfa01de2674a3043e0417b6d
FORGE-CANARY-SLOT-2 bc32f0dbf9a92764528e598342a9ecd2c09c651984b8c9737ba8c4a4e0aba4fc
FORGE-CANARY-SLOT-3 ed11e1f65bf47fbebf39521f6b503c8cf022e8fb15ff3fe355377d4750f1f106
FORGE-CANARY-END
