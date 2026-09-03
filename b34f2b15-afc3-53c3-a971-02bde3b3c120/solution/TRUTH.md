# truth.md - Golden Solve Path

## Task: hephaestus `selection_topk_giant_row_fwd` (anchor `torch.topk`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward` | **Entry:** `topk_giant_row` | **Editable:** `kernel.py`
- **Objective:** implement topk_giant_row in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is a row-wise DESCENDING top-k over VERY LARGE rows (N in the
256K to 4M range) with SMALL k (16 to 128). The huge N is the hardness
lever: the row does not fit in shared memory, an iterative masked-max
extraction pays O(k * N) HBM bandwidth (crushing for small k), and even a
radix select must land its histogram/scatter work in a small number of
streaming passes.

    topk_giant_row(x[R, N], k) -> (values[R, k], indices[R, k])

Semantics, exactly:

  * For row r, `indices[r]` holds the positions of the k LARGEST values in
    x[r, :], ordered DESCENDING by value, ties broken by LOWER ORIGINAL
    INDEX first, `-0.0 == +0.0` one tie group.
  * Indices are int64 positions in [0, N).
  * `values[r, :] = x[r].gather(0, indices[r])`, bit for bit, in the INPUT
    dtype. No arithmetic is performed on the values.
  * Inputs are guaranteed NaN-free; `-inf`/`+inf` are legal.
  * N >= 262144 at every graded shape; 1 <= k <= 128 at every graded shape.

There is one correct answer per input, so correctness is graded as EXACT
equality against this function -- indices by integer equality, values at
the bit level (via a raw-bits view). The `compare` override in taskdef.py
enforces this and rejects any dense-tolerance escape hatch.

The implementation is a stable descending argsort over the whole row
followed by a gather of the first k. Deliberately simple and slow; on the
graded shapes it materializes multi-hundred-megabyte permutations, which
is exactly why this task exists.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes topk_giant_row
  - establishes: topk_giant_row is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: topk_giant_row agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.3003 |
| starter fraction of anchor (measured) | 0.156639 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| float32 tolerance | atol 0.0, rtol 0.0 |
| bfloat16 tolerance | atol 0.0, rtol 0.0 |
| comparison | exact equality |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes topk_giant_row. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: row-wise descending top-k over VERY LARGE N with SMALL k (SURFACE fwd).

Selection family. Everything category-specific the generic verifier consumes
lives here (protocol: seed/forge/core/taskdef_api.py).

Operator semantics (reference.py is the definition; this is a summary):
    topk_giant_row(x[R, N], k) -> (values[R, k], indices[R, k] int64)
    Descending top-k of each row. N in [262144, 4194304] at every graded
    shape and k in [16, 128] at every graded shape. Ties broken by LOWER
    original index first; -0.0 == +0.0 one tie group; values are BIT-EXACT
    gathers of the input element (input dtype); indices int64. Exactly one
    correct answer per input -> EXACT equality; `compare` overrides the
    dense walk.

Hard lever (spec.yaml justifies at length): NO torch/vendor builtin is
purpose-tuned for k << sqrt(N) with N in the multi-megabyte range. CUB's
radix top-k is tuned for a moderate ratio; at k=16 over a 4M row the
theoretical work is dominated by the SINGLE STREAMING READ of the row and
the log(N/k) radix passes, and every wasted pass costs full HBM
bandwidth. An iterative masked-max starter pays O(k) full-row reads and
scores near zero; a competitive candidate must implement two-phase select
(block-local top-k stored, then a reduce phase across blocks) or a
histogram-based threshold descent.

Anchor: CUDA torch.topk (CUB radix select) as the TIMING DENOMINATOR only.
It handles arbitrary (N, k) and is the strongest general vendor path;
correctness is graded against reference.py's stable argsort. No vendorable
source tree: VENDOR_LIB_SUBDIRS empty; the FORBIDDEN symbol scan still runs.

Distribution invariance is made real exactly as for the rest of the family
(hidden scale + hidden distribution shape per draw; timed == checked).
process_checks drops the magnitude-branch lint (selection math IS magnitude
comparison) and keeps no_grader_reference.

STATUS: authored-draft -- needs GPU calibration. TARGET_FRACTION_OF_SOTA
and the golden/controls numbers below are placeholders, NOT measured on
hardware.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `b4b03aa1a42fcc83` |
| `kernel.py` | `631e274f482139d4` |
| `bench.py` | `5b93d8e1b5d09fa1` |
| `instruction.md` | `7fbbbffd8622bb98` |
| `hidden_shapes.json` | `ac57a46c9888a8fe` |
| `solution/oracle_kernel.py` | `3d9f93b7e9e731de` |

- **seed_base:** `72439`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | R=16 N=262144 k=16 | float32, bfloat16 |
| `g2` | R=32 N=524288 k=32 | float32, bfloat16 |
| `g3` | R=8 N=1048576 k=64 | float32, bfloat16 |

- **forbidden symbols (18):** `torch.topk`, `torch.sort`, `torch.argsort`, `torch.msort`, `torch.kthvalue`, `torch.unique`, `torch.median`, `torch.nanmedian`, `torch.quantile`, `torch.nanquantile`, `.topk`, `.sort`, `.argsort`, `.msort`, `.kthvalue`, `.unique`, `.median`, `.quantile`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 8f5ee58ab8b39857e50c03beeea7a3fe78a11f30ac5c7a077d9739b51feab9f9
FORGE-CANARY-SLOT-1 b31fe674e853ba5d9b6250c4b26c42bf20a59df2460e5acd0d203a8041ddc10f
FORGE-CANARY-SLOT-2 016816dab60a3cf6855248a0153321e1ebb58cda5888aa5d4018d9c767b5cacf
FORGE-CANARY-SLOT-3 4887a50f15741a1af5a39225429b1dd6b56d4101d2a794a1650d88c257cc9b15
FORGE-CANARY-END
