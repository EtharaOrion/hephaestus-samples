# truth.md - Golden Solve Path

## Task: hephaestus `chunked_linear_attn_chunked_comba_fwdbwd` (anchor `fla.ops.comba.chunk_comba`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `chunked_comba` | **Editable:** `kernel.py`
- **Objective:** implement chunked_comba in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is CHUNK-PARALLEL COMBA (Compositional Basis Attention): a gated
delta rule whose delta correction erases the state against an AUXILIARY key `p`,
distinct from the write key `k`. Distinct from every other member of the
chunked_linear_attn family: no other variant carries a separate prediction basis
`p` that decouples the erase from the write.

For a single (batch, head), with state S in R^{K x V}:

    S <- diag(exp(g_t)) . S                                       (gate)
    delta_t = ( v_t - S^T p_t ) * beta_t                          (erase against p)
    S <- S + k_t . delta_t^T                                      (write with k)
    o_t = S^T ( q_t / sqrt(K) )

`g` is a per (step, head) scalar log-decay (<= 0). `beta` is a per (step, head)
scalar write strength. `p` and `k` are BOTH length-K but semantically distinct:
`p` is the auxiliary basis the state is REGRESSED against (erase term uses `p`)
while `k` is the WRITE basis (the outer product added to the state uses `k`).
The two are always shipped together and the intra-chunk WY factorization uses
the cross basis `p k^T` (not `k k^T`), so a kernel that uses `k` for the erase
term is wrong on both the output and every gradient.

This is written as a readable sequential scan in float32 and is deliberately slow.
The surface is forward+backward: autograd differentiates this scan, and the input
gradients dq, dk, dv, dp, dg, dbeta ARE the graded gradient oracle. Correctness
is defined as agreement with this function, never with any Triton or CUDA
implementation of it. Semantics ported from the library's own ground truth,
fla/ops/comba/naive.py::naive_recurrent_comba (read, never called or vendored).
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes chunked_comba
  - establishes: chunked_comba is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: chunked_comba agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.68 |
| starter fraction of anchor (measured) | 0.007442 |
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
| `G0` trusted_entrypoint | the submitted module imports and exposes chunked_comba. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: chunk-parallel COMBA (compositional basis attention),
forward+backward (fwdbwd).

Member of the chunked_linear_attn family. HARD variant. Distinct from every
other member because the delta correction is regressed against an AUXILIARY
prediction basis `p` while the state write uses the DIFFERENT basis `k`: the
intra-chunk WY factorization operates on the CROSS matrix `p k^T` rather than
`k k^T`, so no starter that pattern-matches a plain delta-rule kernel is
correct without modification.

Operator (reference.py is the definition; this is a summary):
    chunked_comba(q[B,T,H,K], k[B,T,H,K], v[B,T,H,V], p[B,T,H,K],
                  g[B,T,H], beta[B,T,H]) -> o[B,T,H,V]
    S <- diag(exp(g_t)) S
    delta = (v_t - S^T p_t) * beta_t                                (erase against p)
    S <- S + k_t (x) delta                                          (write with k)
    o_t = S^T (q_t / sqrt(K))
    SURFACE fwdbwd: gradients of q, k, v, p, g, beta are graded, timed and
    determinism-checked.

Anchor: fla.ops.comba.chunk_comba -- the production chunk kernel of the pinned
library. sota() calls it with KEYWORDS, scale=None (=> K^-0.5),
use_qk_l2norm_in_kernel=False; unwraps the (o, final_state) tuple.

STATUS: authored-draft -- NEEDS GPU CALIBRATION. TOL, TARGET_FRACTION_OF_SOTA, the
vendor-shingle ceiling, the anchor semantic-divergence check and every negative
control's bound-gate behaviour are INHERITED placeholders; see spec.yaml.

Distribution invariance (frontier-defeat-analysis §7.1): q, k, p L2-normalized
along K; g log-space in (-inf, 0]; beta in (0.05, 0.95]; the hidden scale
multiplies v. Timed tensors ARE checked tensors; NC-DP plants the §5.4
magnitude-branch exploit on the oracle under FORGE_ORACLE=1 and must zero at
exactly G2.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `7605a38bc9952199` |
| `kernel.py` | `ab457c0fbaabfde1` |
| `bench.py` | `86ba1ef6c1eef298` |
| `instruction.md` | `65b07858a5203ad9` |
| `hidden_shapes.json` | `d30042f91bec1fa5` |
| `solution/oracle_kernel.py` | `d5b2d44f117b6d2e` |

- **seed_base:** `305911`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=1 T=4096 H=8 K=128 V=128 | bfloat16 |
| `g2` | B=4 T=1024 H=8 K=128 V=128 | bfloat16, float16 |
| `g3` | B=2 T=2048 H=16 K=64 V=64 | bfloat16 |
| `g4` | B=2 T=2048 H=8 K=128 V=128 | bfloat16 |
| `g5` | B=16 T=512 H=8 K=128 V=128 | bfloat16 |

- **forbidden symbols (26):** `fla.ops`, `import fla`, `from fla`, `chunk_comba`, `fused_recurrent_comba`, `naive_recurrent_comba`, `naive_chunk_comba`, `chunk_gated_delta_rule`, `chunk_gated_delta_product`, `chunk_delta_rule`, `chunk_gla`, `chunk_simple_gla`, `chunk_retention`, `chunk_gdn`, `chunk_gdn2`, `chunk_kda`, `chunk_rwkv6`, `chunk_rwkv7`, `chunk_dplr_delta_rule`, `chunk_iplr_delta_rule`, `fused_recurrent_gated_delta_rule`, `fused_recurrent_delta_rule`, `fused_recurrent_gla`, `fused_recurrent_simple_gla`, `fused_recurrent_retention`, `fused_recurrent`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 0dbcb37aa49e750934ca346e9728b2478ca3ac23a92fa990c606eaceeb387dc0
FORGE-CANARY-SLOT-1 d74d65d8b8c6b275caac0adc871695e11373e8ef8a88eb37a9bb0ce1334a8523
FORGE-CANARY-SLOT-2 b6dd38b39decebc93bb079d81367a0d7c5d9d1f5c9857133a93ba76b414a9333
FORGE-CANARY-SLOT-3 d84ed212e57d92108a7c6f0019ccc29d9c4568f3c39780a2404ad876defbe034
FORGE-CANARY-END
