# truth.md - Golden Solve Path

## Task: hephaestus `selection_fused_softmax_topk_fwd` (anchor `torch.softmax + torch.topk`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward` | **Entry:** `fused_softmax_topk` | **Editable:** `kernel.py`
- **Objective:** implement fused_softmax_topk in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is a fused row-wise softmax followed by a top-k selection,
canonical shape of a speculative-decoding gate:

    fused_softmax_topk(logits[R, N], k) -> (probs[R, k] float32, indices[R, k] int64)

Semantics, exactly:

  * Row-wise softmax (canonical fp32 reduction):
        m       = logits.max(dim=-1, keepdim=True).values.float()
        e       = (logits.float() - m).exp()
        s       = e.sum(dim=-1, keepdim=True)
        probs_full = e / s                                            # [R, N] fp32
    The intermediate math is float32 regardless of input dtype; this
    prevents bf16 subtract-max cancellation and is the standard fused
    softmax numerical policy.
  * `indices[r]` holds the positions of the k LARGEST values of
    logits[r, :] (equivalently, of probs_full[r, :]) in [0, N), ordered
    DESCENDING by value, ties broken by LOWER ORIGINAL INDEX first,
    `-0.0 == +0.0` one tie group. Softmax is strictly increasing in the
    input, so top-k over the logits is exactly top-k over the probs;
    tie groups are preserved.
  * `probs[r, :] = probs_full[r].gather(0, indices[r])` in float32.
    Values are the softmax probabilities at the selected indices; they
    are compared with a dtype-dependent tolerance (see taskdef.TOL).
  * `indices` is int64.
  * Inputs are guaranteed NaN-free; `-inf`/`+inf` are legal on the logits
    (a `+inf` logit yields a probability of 1 at that index, everything
    else 0, and it wins the top-k unambiguously).

Correctness grading: indices EXACT (integer equality); probs by tolerance
per dtype (TOL in taskdef.py). The `compare` override enforces this split;
a candidate that matches indices but returns probs computed with a
different reduction policy (e.g. bf16 accumulator) is caught by the probs
tolerance on bfloat16 inputs.

The implementation runs a stable descending argsort on logits, gathers the
first k local indices, and computes the softmax reduction on the row's
fp32 upcast then gathers probs at those indices. Deliberately simple and
slow.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes fused_softmax_topk
  - establishes: fused_softmax_topk is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: fused_softmax_topk agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.3 |
| starter fraction of anchor (measured) | 0.255759 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| float32 tolerance | atol 1e-06, rtol 1e-05 |
| bfloat16 tolerance | atol 0.0005, rtol 0.005 |
| comparison | within dtype tolerance |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes fused_softmax_topk. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: fused row-wise softmax + top-k speculative-decoding gate (SURFACE fwd).

Selection family. Everything category-specific the generic verifier consumes
lives here (protocol: seed/forge/core/taskdef_api.py).

Operator semantics (reference.py is the definition; this is a summary):
    fused_softmax_topk(logits[R, N], k) -> (probs[R, k] fp32, indices[R, k] int64)
    Canonical fp32 softmax over each row (max-subtract, exp, sum-normalize),
    then top-k of the row (equivalently, top-k of the logits since softmax
    is strictly increasing). indices are DESCENDING by value with LOWER
    ORIGINAL INDEX tie-break; -0.0 == +0.0 one tie group. probs are the
    softmax probabilities at those indices, ALWAYS fp32. Grading is EXACT
    on indices and dtype-tolerance on probs. `compare` overrides the dense
    walk to enforce this split.

Hard lever (spec.yaml justifies at length): NO single torch/vendor builtin
fuses softmax and top-k in one pass. The natural composition
torch.softmax(x, -1) then torch.topk(probs, k) reads the row THREE TIMES
(max, sum, and again for topk); a fused kernel reads it once and
overlaps the softmax reduction with the top-k threshold descent. This is
the speculative-decoding gate operator in every modern LLM serving stack,
and the current sota is a hand-written fused kernel (FlashAttention-style
one-pass online softmax + partial top-k). A candidate must reproduce that
fusion.

Anchor: torch.softmax + torch.topk composition. TIMING DENOMINATOR ONLY.
Its tie-break can differ from ours (CUB's tuned paths break value ties
differently), never consulted for correctness. reference.py's stable
argsort + canonical fp32 softmax is the definition. No vendorable source
tree: VENDOR_LIB_SUBDIRS empty; the FORBIDDEN symbol scan still runs.

Distribution invariance is made real exactly as for the rest of the family
(hidden scale + hidden distribution shape per draw; timed == checked).
process_checks drops the magnitude-branch lint (softmax max-subtract IS a
magnitude branch) and keeps no_grader_reference.

STATUS: authored-draft -- needs GPU calibration. TARGET_FRACTION_OF_SOTA
and the golden/controls numbers below are placeholders, NOT measured on
hardware.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `4f7d25f6154356a9` |
| `kernel.py` | `2bc3d13d463cd36b` |
| `bench.py` | `5910b3c6f6319efb` |
| `instruction.md` | `fa7de58cec594b8e` |
| `hidden_shapes.json` | `616afdf52a9dd87f` |
| `solution/oracle_kernel.py` | `72e88204accfd5f7` |

- **seed_base:** `74653`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | R=256 N=32000 k=16 | float32, bfloat16 |
| `g2` | R=128 N=65536 k=32 | float32, bfloat16 |
| `g3` | R=64 N=128256 k=64 | float32, bfloat16 |

- **forbidden symbols (24):** `torch.topk`, `torch.sort`, `torch.argsort`, `torch.msort`, `torch.kthvalue`, `torch.unique`, `torch.median`, `torch.nanmedian`, `torch.quantile`, `torch.nanquantile`, `.topk`, `.sort`, `.argsort`, `.msort`, `.kthvalue`, `.unique`, `.median`, `.quantile`, `torch.softmax`, `F.softmax`, `torch.nn.functional.softmax`, `.softmax`, `log_softmax`, `_softmax`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 a67c7147dadb026ca22380fe0ae29b13aae8013ae3d9ade671aa63d3f3914cd7
FORGE-CANARY-SLOT-1 36d74bacec9865f8221688e19ad0abadb6fdde1389675a1d4a627add6157d4d0
FORGE-CANARY-SLOT-2 31183c6bdaa23318a7d211bd3c1b4dbf19e583763b2a433fc87d64c4edf95ff5
FORGE-CANARY-SLOT-3 bff5b333d2ac75c2d25ea6280dd8e7bf0268981cc2191487f9467eaf2b50256c
FORGE-CANARY-END
