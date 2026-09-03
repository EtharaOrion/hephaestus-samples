# truth.md - Golden Solve Path

## Task: hephaestus `chunked_linear_attn_chunked_gated_delta_product_fwdbwd` (anchor `fla.ops.gated_delta_product.chunk_gated_delta_product`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `chunked_gated_delta_product` | **Editable:** `kernel.py`
- **Objective:** implement chunked_gated_delta_product in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is CHUNK-PARALLEL GATED DELTA-PRODUCT: a gated delta rule that
applies `num_householder` (N) delta updates PER TOKEN before the readout.
Distinct from chunked_gated_delta (single Householder) because each token
performs N nested `(I - k (beta*k)^T)` Householder transformations of the
state -- meaning the intra-chunk factorization must invert an N*BT x N*BT
lower-triangular system rather than a BT x BT one. This is the "generalized"
step from a single-write-per-token delta rule to a product-of-writes rule
described in the delta-product paper.

For a single (batch, head), with state S in R^{K x V}, per-token loop:

    S <- diag(exp(g_t)) . S                             (gate)
    for j in 0 .. N-1:                                  (N Householder steps)
        S <- S + (v_{t,j} - S^T k_{t,j})^T * k_{t,j} * beta_{t,j}
    o_t = S^T (q_t / sqrt(K))

Here `q` carries length T and `k, v, beta` carry length T*N: index `t*N + j`
is the j-th Householder tuple of token t. `g` carries length T (one gate per
token, applied ONCE before the N-fold write). `beta` is a per (step, head)
scalar write strength.

The N-fold intra-token write makes the intra-chunk delta term N stacked
`k (beta*k)^T` erasures, which chain multiplicatively along j -- the
lower-triangular `A_ab` matrix that captures the recurrence has (N*BT)
rows and columns instead of BT, and inverting it exactly is the WY term
whose size grows with num_householder.

This is written as a readable sequential scan in float32 and is deliberately slow.
The surface is forward+backward: autograd differentiates this scan, and the input
gradients dq, dk, dv, dg, dbeta ARE the graded gradient oracle. Correctness is
defined as agreement with this function, never with any Triton or CUDA
implementation of it. Semantics ported from the library's own ground truth,
fla/ops/gated_delta_product/naive.py::naive_recurrent_gated_delta_product
(read, never called or vendored).
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes chunked_gated_delta_product
  - establishes: chunked_gated_delta_product is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: chunked_gated_delta_product agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.65 |
| starter fraction of anchor (measured) | 0.004158 |
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
| `G0` trusted_entrypoint | the submitted module imports and exposes chunked_gated_delta_product. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: chunk-parallel gated delta-product (N Householder writes per token),
forward+backward (fwdbwd).

Member of the chunked_linear_attn family. HARD variant. Distinct from every other
member because each token performs `num_householder` NESTED Householder writes
before the readout -- the WY-style intra-chunk factorization must invert a
(N*BT) x (N*BT) lower-triangular system rather than a BT x BT one, and every
`beta` scale carries N times the per-token cost.

Operator (reference.py is the definition; this is a summary):
    chunked_gated_delta_product(q[B,T,H,K], k[B,T*N,H,K], v[B,T*N,H,V],
                                 g[B,T,H], beta[B,T*N,H], num_householder=N)
        -> o[B,T,H,V]
    Per-token: S <- diag(exp(g_t)) S;  then N times:
        S <- S + (v_{t,j} - S^T k_{t,j})^T * k_{t,j} * beta_{t,j}
    Readout: o_t = S^T (q_t / sqrt(K)). N Householder writes chain multiplicatively
    per token; the gate is applied ONCE before the N writes. SURFACE fwdbwd:
    gradients of q, k, v, g, beta are graded, timed and determinism-checked.
    num_householder is a fixed int argument (no grad).

Anchor: fla.ops.gated_delta_product.chunk_gated_delta_product -- the production
chunk kernel of the pinned library. sota() calls it with KEYWORDS,
num_householder=2, scale=None (=> K^-0.5) and unwraps the (o, final_state) tuple.

STATUS: authored-draft -- NEEDS GPU CALIBRATION. TOL, TARGET_FRACTION_OF_SOTA, the
vendor-shingle ceiling, the anchor semantic-divergence check and every negative
control's bound-gate behaviour are INHERITED placeholders; see spec.yaml.

Distribution invariance (frontier-defeat-analysis §7.1): q, k L2-normalized along
K; g log-space in (-inf, 0]; beta in (0.05, 0.95]; the hidden per-invocation scale
multiplies v (the one magnitude carrier). Timed tensors ARE checked tensors;
NC-DP plants the §5.4 magnitude-branch exploit on the oracle under FORGE_ORACLE=1
and must zero at exactly G2.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `477fdb28fdfd2c08` |
| `kernel.py` | `f27062c60d638731` |
| `bench.py` | `b08c1728f85943e7` |
| `instruction.md` | `4a510070c813bcf4` |
| `hidden_shapes.json` | `7dd60c9c096a84b6` |
| `solution/oracle_kernel.py` | `fc7e0633da146cfb` |

- **seed_base:** `214703`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=1 T=4096 H=8 K=128 V=128 | bfloat16 |
| `g2` | B=4 T=1024 H=8 K=128 V=128 | bfloat16, float16 |
| `g3` | B=2 T=2048 H=16 K=64 V=64 | bfloat16 |
| `g4` | B=2 T=2048 H=8 K=128 V=128 | bfloat16 |
| `g5` | B=16 T=512 H=8 K=128 V=128 | bfloat16 |

- **forbidden symbols (30):** `fla.ops`, `import fla`, `from fla`, `chunk_gated_delta_product`, `fused_recurrent_gated_delta_product`, `naive_recurrent_gated_delta_product`, `chunk_gated_delta_rule`, `chunk_gdn`, `chunk_kda`, `chunk_delta_rule`, `chunk_gla`, `chunk_simple_gla`, `chunk_retention`, `chunk_delta_h`, `chunk_comba`, `chunk_gdn2`, `chunk_rwkv6`, `chunk_rwkv7`, `chunk_dplr_delta_rule`, `chunk_iplr_delta_rule`, `fused_recurrent_gated_delta_rule`, `fused_recurrent_gdn`, `fused_recurrent_kda`, `fused_recurrent_delta_rule`, `fused_recurrent_gla`, `fused_recurrent_simple_gla`, `fused_recurrent_retention`, `fused_recurrent`, `naive_recurrent_gated_delta_rule`, `naive_chunk_gated_delta_rule`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 f6ab49de11d2b8c644e451dbbb006d8b7894acb6f0fe89d1db6403266091d78b
FORGE-CANARY-SLOT-1 2b0c7224a9eef2572ceced4db57eaa02dfbf7cfcbce5d1d396f9a736f5b94e8d
FORGE-CANARY-SLOT-2 dc193ff01b034159ce65ff851c3390033cdfe7d8de6fadf586f91fb6f8ad27dd
FORGE-CANARY-SLOT-3 56f261abf5278f2651488981c78e88b4f0191965de952be4c2ff1efdc83c843e
FORGE-CANARY-END
