# truth.md - Golden Solve Path

## Task: hephaestus `chunked_linear_attn_chunked_gdn2_fwdbwd` (anchor `fla.ops.gdn2.chunk_gdn2`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `chunked_gdn2` | **Editable:** `kernel.py`
- **Objective:** implement chunked_gdn2 in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is CHUNK-PARALLEL GDN-2 (Gated DeltaNet 2): a per-key-channel gated
delta rule with SEPARATE per-channel ERASE gate `b` (on the K axis) and per-channel
WRITE gate `w` (on the V axis). Distinct from every other member of the
chunked_linear_attn family because BOTH gates are per-channel AND the state is
the largest of the family (K = V = 256 on the graded distribution). Collapsing
`b = w = beta` (scalar) recovers KDA / gated_delta_rule -- so a starter that
pattern-matches on that special case is wrong on the graded distribution.

For a single (batch, head), with state S in R^{K x V}:

    S_t = ( I - k_t (b_t * k_t)^T ) . Diag(exp(g_t)) . S_{t-1} + k_t (w_t * v_t)^T
    o_t = S_t^T ( q_t / sqrt(K) )

which is implemented per token as:

    S <- Diag(exp(g_t)) . S                                    (per-K-channel gate)
    erase = ((b_t * k_t)^T . S)                                (per-K erase read)  [B,H,V]
    S <- S + k_t . (w_t * v_t - erase)^T                        (per-V write)
    o_t = S^T ( q_t / sqrt(K) )

Everything is accumulated in float32 regardless of the input dtype. Semantics
ported from the library's own ground truth, fla/ops/gdn2/naive.py::
naive_recurrent_gdn2 (read, never called or vendored).
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes chunked_gdn2
  - establishes: chunked_gdn2 is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: chunked_gdn2 agrees with reference.py and is the baseline to beat
  - survives: the published shapes in bench.py are smaller than and different from the graded shapes

**Step 3 (G1).** read the production anchor to understand the fast formulation, without importing or calling it
  - establishes: the decomposition that makes the operator fast is understood
  - survives: a forbidden-symbol scan over the submitted source with comments stripped, so citing a symbol is legal and calling it is not

**Step 4 (G2).** write the kernel and iterate against bench.py, holding the reference semantics fixed
  - establishes: the output and every input gradient still agrees with reference.py on the hidden configurations
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
| target fraction of anchor | 0.66 |
| starter fraction of anchor (measured) | 0.007184 |
| frontier-defeat floor S | 0.15 |
| written-kernel adoption floor | 0.6 |
| bfloat16 tolerance | atol 0.05, rtol 0.05 |
| float16 tolerance | atol 0.01, rtol 0.00390625 |
| float32 tolerance | atol 0.002, rtol 0.002 |
| comparison | within dtype tolerance |

per graded shape r = t_anchor / t_candidate clipped into [0, 1]; R is the geometric mean of r; the score is R relative to the target and is zero unless every gate passes.

## Section 4: Verification

| Checker | Verifies |
| --- | --- |
| `G0` trusted_entrypoint | the submitted module imports and exposes chunked_gdn2. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
| `G1` forbidden_symbols | the submission neither imports nor calls the production kernel or any equivalent library implementation, scanned over the source with comments stripped so that citing a symbol stays legal and calling it does not. |
| `G2` correctness | the output and all input gradients agrees with reference.py on every hidden configuration, on the same randomly scaled draw that is timed, so no branch can tell the checked distribution from the timed one. |
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
Taskdef: chunk-parallel GDN-2 (Gated DeltaNet 2), forward+backward (fwdbwd).

Member of the chunked_linear_attn family. HARD variant. Distinct from every
other member because BOTH erase gate `b` and write gate `w` are per-channel
(b on K, w on V) AND the graded state is the largest of the family (K = V = 256
on g4). Collapsing b = w = beta (scalar) recovers KDA / chunked_gated_delta --
so a starter that pattern-matches on that special case is wrong on the graded
distribution.

Operator (reference.py is the definition; this is a summary):
    chunked_gdn2(q[B,T,H,K], k[B,T,H,K], v[B,T,H,V],
                 g[B,T,H,K], b[B,T,H,K], w[B,T,H,V]) -> o[B,T,H,V]
    S <- Diag(exp(g_t)) . S                                         (per-K decay)
    erase = ((b_t * k_t)^T . S)                                     (per-K erase read)
    S <- S + k_t . (w_t * v_t - erase)^T                            (per-V write)
    o_t = S^T (q_t / sqrt(K))
    SURFACE fwdbwd: gradients of q, k, v, g, b, w are graded, timed and
    determinism-checked.

Anchor: fla.ops.gdn2.chunk_gdn2 -- the production chunk kernel of the pinned
library. sota() calls it with KEYWORDS, scale=None (=> K^-0.5),
use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, and unwraps the
(o, final_state) tuple.

STATUS: authored-draft -- NEEDS GPU CALIBRATION. TOL, TARGET_FRACTION_OF_SOTA, the
vendor-shingle ceiling, the anchor semantic-divergence check and every negative
control's bound-gate behaviour are INHERITED placeholders; see spec.yaml.

Distribution invariance (frontier-defeat-analysis §7.1): q, k L2-normalized along
K; g log-space in (-inf, 0]; b, w sigmoid-shaped in (0, 1); the hidden scale
multiplies v. Timed tensors ARE checked tensors; NC-DP plants the §5.4
magnitude-branch exploit on the oracle under FORGE_ORACLE=1 and must zero at
exactly G2.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `3e38371e8235ea25` |
| `kernel.py` | `79b932932fc708b2` |
| `bench.py` | `506d60a2978c6fc1` |
| `instruction.md` | `d66ca3d24b30e7cf` |
| `hidden_shapes.json` | `e3a41b1e3d85f4ae` |
| `solution/oracle_kernel.py` | `95740ea26faa91ab` |

- **seed_base:** `419773`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=1 T=4096 H=8 K=128 V=128 | bfloat16 |
| `g2` | B=4 T=1024 H=8 K=128 V=128 | bfloat16, float16 |
| `g3` | B=2 T=2048 H=8 K=256 V=128 | bfloat16 |
| `g4` | B=2 T=1024 H=4 K=256 V=256 | bfloat16 |
| `g5` | B=16 T=512 H=8 K=128 V=128 | bfloat16 |

- **forbidden symbols (28):** `fla.ops`, `import fla`, `from fla`, `chunk_gdn2`, `fused_recurrent_gdn2`, `naive_recurrent_gdn2`, `naive_chunk_gdn2`, `chunk_kda`, `chunk_gdn`, `chunk_gated_delta_rule`, `chunk_gated_delta_product`, `chunk_delta_rule`, `chunk_gla`, `chunk_simple_gla`, `chunk_retention`, `chunk_comba`, `chunk_rwkv6`, `chunk_rwkv7`, `chunk_dplr_delta_rule`, `chunk_iplr_delta_rule`, `fused_recurrent_gated_delta_rule`, `fused_recurrent_delta_rule`, `fused_recurrent_gla`, `fused_recurrent_simple_gla`, `fused_recurrent_retention`, `fused_recurrent_gdn`, `fused_recurrent_kda`, `fused_recurrent`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 5f502cffc6145ac72b5802e99e10768d5d4f786f8cc0834efd90a8a95b6c378e
FORGE-CANARY-SLOT-1 f047a424b68f371e8611ef3f76c3eb166ae396a4daa9552e37a198ba573a4eba
FORGE-CANARY-SLOT-2 738e3eba492a7096fff592ac9fd2c597e035e8adb88a0eb447fbac3d0f72894a
FORGE-CANARY-SLOT-3 7d279c030e0d4a58f2bda4d383ff36c0a7d65c1eeb5d49ee6678dc49cf414c77
FORGE-CANARY-END
