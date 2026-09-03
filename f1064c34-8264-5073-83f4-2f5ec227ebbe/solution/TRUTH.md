# truth.md - Golden Solve Path

## Task: hephaestus `chunked_linear_attn_chunked_dplr_delta_fwdbwd` (anchor `fla.ops.generalized_delta_rule.chunk_dplr_delta_rule`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `chunked_dplr_delta` | **Editable:** `kernel.py`
- **Objective:** implement chunked_dplr_delta in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is CHUNK-PARALLEL DPLR (Diagonal-Plus-Low-Rank) generalized delta
rule -- the recurrence at the core of RWKV-7. It generalizes gated linear
attention with a per-K-channel log-decay `gk` by ADDING a rank-1 correction
term `beta_t (S^T alpha_t)^T` per step. Distinct from every other member of
the chunked_linear_attn family: no other variant has BOTH a per-channel gate
AND a low-rank additive correction that mixes across the state's V axis.

For a single (batch, head), with state S in R^{K x V}:

    S_t = diag(exp(gk_t)) . S_{t-1}
          + k_t v_t^T
          + beta_t . ( S_{t-1}^T alpha_t )^T                (rank-1 DPLR correction)
    o_t = S_t^T ( q_t / sqrt(K) )

alpha_t, beta_t and k_t are all length-K vectors; the rank-1 update `beta_t
(S^T alpha_t)^T` reads a V-vector from the state through `alpha` and writes
it back through `beta`. Setting alpha == beta == 0 recovers plain gated linear
attention (GLA), and setting alpha == k, beta == -k*beta_scalar recovers the
delta-rule Householder form -- both are graded-distribution wrong. The intra-
chunk factorization must invert the (I + A_ab) matrix that captures the
alpha-beta erasure recurrence AND the standard A_qk gate-decay pattern; see
dplr_chunkwise in the pinned library's naive reference for the full algebra.

The graded distribution stresses this variant with LONG-CONTEXT recurrence
(T = 8192 on g1), where numerical error accumulates over many chunk boundaries
and the state carry across chunks matters more than the intra-chunk fusion. A
solver that keeps the chunk-boundary state in bf16 loses precision at every
carry and drifts far outside tolerance by t = 8192.

This is written as a readable sequential scan in float32 and is deliberately slow.
The surface is forward+backward: autograd differentiates this scan, and the input
gradients dq, dk, dv, dalpha, dbeta, dgk ARE the graded gradient oracle.
Correctness is defined as agreement with this function, never with any Triton or
CUDA implementation of it. Semantics ported from the library's own ground truth,
fla/ops/generalized_delta_rule/dplr/naive.py::dplr_recurrence (read, never
called or vendored).
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes chunked_dplr_delta
  - establishes: chunked_dplr_delta is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: chunked_dplr_delta agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.63 |
| starter fraction of anchor (measured) | 0.002783 |
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
| `G0` trusted_entrypoint | the submitted module imports and exposes chunked_dplr_delta. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: chunk-parallel DPLR generalized delta rule (RWKV-7 family),
forward+backward (fwdbwd).

Member of the chunked_linear_attn family. HARD variant. Distinct from every
other member because the recurrence has BOTH a per-K-channel log-decay `gk`
AND a rank-1 (alpha, beta) additive correction term whose `alpha` reads a
V-vector out of the state and `beta` writes it back at a different K position
-- the Diagonal-Plus-Low-Rank (DPLR) form. This composition drives the
generalized delta family that RWKV-7 is built on and it is what the pinned
library's own naive dplr_recurrence and dplr_chunkwise reference exercise.

Operator (reference.py is the definition; this is a summary):
    chunked_dplr_delta(q[B,T,H,K], k[B,T,H,K], v[B,T,H,V],
                       alpha[B,T,H,K], beta[B,T,H,K], gk[B,T,H,K])
        -> o[B,T,H,V]
    S_t = diag(exp(gk_t)) S_{t-1} + k_t v_t^T + beta_t (S_{t-1}^T alpha_t)^T
    o_t = S_t^T (q_t / sqrt(K))
    SURFACE fwdbwd: gradients of q, k, v, alpha, beta, gk are graded, timed and
    determinism-checked.

Anchor: fla.ops.generalized_delta_rule.chunk_dplr_delta_rule -- the production
chunk kernel of the pinned library. sota() calls it with KEYWORDS and
scale=None (=> K^-0.5); unwraps the (o, final_state) tuple.

STATUS: authored-draft -- NEEDS GPU CALIBRATION. TOL, TARGET_FRACTION_OF_SOTA, the
vendor-shingle ceiling, the anchor semantic-divergence check and every negative
control's bound-gate behaviour are INHERITED placeholders; see spec.yaml. The
long-context graded shape (T = 8192) is this variant's SIGNATURE stress: state
carried across ~128 chunk boundaries where numerical error accumulates.

Distribution invariance (frontier-defeat-analysis §7.1): q, k L2-normalized
along K; gk log-space in (-inf, 0]; alpha, beta drawn small
(std 0.5, clipped) so the rank-1 correction stays bounded; the hidden per-
invocation scale multiplies v. Timed tensors ARE checked tensors; NC-DP plants
the §5.4 magnitude-branch exploit on the oracle under FORGE_ORACLE=1 and must
zero at exactly G2.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `143306c16bfa2ab1` |
| `kernel.py` | `d74d76c19013a918` |
| `bench.py` | `977672478773e04b` |
| `instruction.md` | `68e54f9f7b8a4ba8` |
| `hidden_shapes.json` | `b9b1bc9393fb8891` |
| `solution/oracle_kernel.py` | `8775fddf72742a7b` |

- **seed_base:** `528619`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=1 T=8192 H=4 K=128 V=128 | bfloat16 |
| `g2` | B=4 T=1024 H=8 K=128 V=128 | bfloat16, float16 |
| `g3` | B=2 T=2048 H=16 K=64 V=64 | bfloat16 |
| `g4` | B=2 T=2048 H=8 K=128 V=128 | bfloat16 |
| `g5` | B=16 T=512 H=8 K=128 V=128 | bfloat16 |

- **forbidden symbols (30):** `fla.ops`, `import fla`, `from fla`, `chunk_dplr_delta_rule`, `chunk_iplr_delta_rule`, `fused_recurrent_dplr_delta_rule`, `fused_recurrent_iplr_delta_rule`, `dplr_recurrence`, `dplr_chunkwise`, `chunk_rwkv7`, `fused_recurrent_rwkv7`, `fused_mul_recurrent_rwkv7`, `chunk_rwkv6`, `fused_recurrent_rwkv6`, `chunk_gated_delta_rule`, `chunk_gated_delta_product`, `chunk_delta_rule`, `chunk_gla`, `chunk_simple_gla`, `chunk_retention`, `chunk_gdn`, `chunk_gdn2`, `chunk_kda`, `chunk_comba`, `fused_recurrent_gated_delta_rule`, `fused_recurrent_delta_rule`, `fused_recurrent_gla`, `fused_recurrent_simple_gla`, `fused_recurrent_retention`, `fused_recurrent`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 bd75c14bbcecda2649446976c929276c87f84cb606f835b626dc13d237a7b3c1
FORGE-CANARY-SLOT-1 7b198b9eceb17cf635390975a768934822f0dcde9db54e42af880cf21656f5a9
FORGE-CANARY-SLOT-2 6f1b5d971b8cc6be8974bc414d7c5ab09158c1bfb08989b156ebe83e3c2b39f3
FORGE-CANARY-SLOT-3 f349e824bc001431bdbc4b7f6830926c0b61185fb870fa19d266dcf1ab6f88b8
FORGE-CANARY-END
