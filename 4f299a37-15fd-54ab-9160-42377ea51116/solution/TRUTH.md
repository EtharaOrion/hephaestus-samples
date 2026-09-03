# truth.md - Golden Solve Path

## Task: hephaestus `selection_topk_segmented_fwd` (anchor `torch.topk (per-segment over the [R, G, S] view)`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward` | **Entry:** `topk_segmented` | **Editable:** `kernel.py`
- **Objective:** implement topk_segmented in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is SEGMENTED top-k: each row is cut into contiguous fixed-length
segments and a descending top-k is taken inside every segment independently:

    topk_segmented(x[R, N], k, segment) -> (values[R, G, k], indices[R, G, k])

with segment length S = `segment`, S dividing N, and G = N // S segments per
row. 1 <= k <= S.

Semantics, exactly:

  * For row r and segment s (columns [s*S, s*S + S) of x[r]), `indices[r, s]`
    holds the positions of the k LARGEST elements WITHIN that segment, ordered
    by DESCENDING value, ties broken by LOWER ORIGINAL INDEX first, `-0.0 ==
    +0.0` one tie group.
  * Indices are GLOBAL positions within the row (in [0, N)): index = s*S + (the
    local position inside the segment). Global order and local order agree
    inside a segment, so the tie-break is unambiguous.
  * `values[r, s] = x[r].gather(indices[r, s])`, bit for bit, in the INPUT
    dtype. No arithmetic is performed on the values.
  * `indices` is int64.
  * Inputs are guaranteed NaN-free by construction; `-inf`/`+inf` are legal.

There is no single library builtin for this; the timing anchor applies
torch.topk to the [R, G, S] segment view (that is the denominator only, and its
tie-break differs from the rule above, so it is never used for correctness).
Correctness is EXACT equality against this function.

The implementation reshapes to the [R, G, S] segment view, stable-descending
argsorts the last (length-S) axis, gathers the first k, and adds the segment
offset s*S to make indices global. Deliberately simple and slow.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes topk_segmented
  - establishes: topk_segmented is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: topk_segmented agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.12 |
| starter fraction of anchor (measured) | 0.197247 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| float32 tolerance | atol 0.0, rtol 0.0 |
| bfloat16 tolerance | atol 0.0, rtol 0.0 |
| comparison | exact equality |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes topk_segmented. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: segmented descending top-k within fixed-length row segments (fwd).

Selection family -- the family's HARD task. Everything category-specific the
generic verifier consumes lives here (protocol: seed/forge/core/taskdef_api.py).

Operator semantics (reference.py is the definition; this is a summary):
    topk_segmented(x[R, N], k, segment) -> (values[R, G, k], indices[R, G, k] int64)
    Each row is cut into G = N // segment contiguous length-S segments (S =
    segment divides N) and a DESCENDING top-k is taken inside every segment
    independently. Indices are GLOBAL row positions (in [0, N)): index =
    s*S + local, with local the position inside segment s. Ties broken by LOWER
    original index first (global and local order agree inside a segment);
    -0.0 == +0.0 one tie group; values are BIT-EXACT gathers of the input
    element (input dtype); indices int64. 1 <= k <= S. Exactly one correct
    answer per input -> EXACT equality; `compare` overrides the dense walk.

Anchor: there is NO single torch builtin. The timing denominator applies
CUDA torch.topk to the [R, G, S] segment view (batched CUB radix select over
many short rows -- a near-roofline vendor path) and globalises the indices.
It is a TIMING DENOMINATOR ONLY (its tie-break differs), never consulted for
correctness. No vendorable source tree: VENDOR_LIB_SUBDIRS empty; the FORBIDDEN
symbol scan still runs.

Distribution invariance is made real exactly as for the rest of the family
(hidden scale + hidden distribution shape per draw; timed == checked).
process_checks drops the magnitude-branch lint and keeps no_grader_reference.

STATUS: authored-draft -- needs GPU calibration. TARGET_FRACTION_OF_SOTA and
the golden/controls numbers below are placeholders, NOT measured on hardware.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `c545a7e065f6a4a5` |
| `kernel.py` | `a3783a89615b807c` |
| `bench.py` | `6589e7005a28c6a9` |
| `instruction.md` | `0ca58966013e89a0` |
| `hidden_shapes.json` | `fc2f1dca5677be8a` |
| `solution/oracle_kernel.py` | `1817595436e4850e` |

- **seed_base:** `92383`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | R=128 N=16384 k=32 segment=512 | float32, bfloat16 |
| `g2` | R=256 N=65536 k=64 segment=1024 | float32, bfloat16 |
| `g3` | R=64 N=262144 k=16 segment=256 | float32, bfloat16 |

- **forbidden symbols (18):** `torch.topk`, `torch.sort`, `torch.argsort`, `torch.msort`, `torch.kthvalue`, `torch.unique`, `torch.median`, `torch.nanmedian`, `torch.quantile`, `torch.nanquantile`, `.topk`, `.sort`, `.argsort`, `.msort`, `.kthvalue`, `.unique`, `.median`, `.quantile`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 4f246aa291e433a8e030563862493822331fb1392d77d71f86ad3e7c8f1cfe27
FORGE-CANARY-SLOT-1 996489b2cafa7a5b8fe7e241faa945ee167ffbc6692ef43396b04ef87eb6d764
FORGE-CANARY-SLOT-2 869d801445f7bc24159c0b502b1cd254bd2da38ae8b4574cf1751281f4ba7413
FORGE-CANARY-SLOT-3 7803ba2d4ce892e0da20b10fa5754e1928e0caad133f596c2147ea29f1f338bd
FORGE-CANARY-END
