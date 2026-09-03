# truth.md - Golden Solve Path

## Task: hephaestus `chunked_linear_attn_chunked_rwkv6_fwdbwd` (anchor `fla.ops.rwkv6.chunk_rwkv6`)

<!-- GENERATED SECTION. DO NOT HAND-EDIT. Source of truth: solution/grounding.yaml -->

- **Surface:** `forward_backward` | **Entry:** `chunked_rwkv6` | **Editable:** `kernel.py`
- **Objective:** implement chunked_rwkv6 in kernel.py for the H100 and get as close as possible to the speed of the production anchor, without changing what it computes.
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

The operator is CHUNK-PARALLEL RWKV-6 (a.k.a. WKV-6): a linear-attention recurrence
with a PER-KEY-CHANNEL log-decay `w` AND a data-independent per-(head, channel)
BONUS `u` applied to the CURRENT step's outer product. Distinct from every other
member of the chunked_linear_attn family: no other variant carries a current-step
bonus that breaks the standard chunk-parallel factorization at the diagonal.

For a single (batch, head), with state S in R^{K x V}:

    S_t = diag(exp(w_t)) . S_{t-1} + k_t v_t^T
    o_t = r_t^T ( S_{t-1}  +  diag(u) . (k_t v_t^T) )

`w_t` is a length-K log-decay vector (typically <= 0; the state row for key channel
k is multiplied by exp(w_t[k]) at every step). `u[h, k]` is a per-head, per-key
learned SCALAR that boosts the CURRENT-step outer product ONLY in the output --
the recurrent state carried into t+1 does NOT see u. This asymmetry is the
distinguishing lever: the intra-chunk kernel must compute two closely-related
terms per position (one weighted by u on the diagonal, one not), and the standard
chunked lower-triangular attention pattern no longer captures the diagonal on its
own -- see naive_chunk_rwkv6 in the pinned library, which is written that way.

The recurrence emits the output on the OLD state (S_{t-1}) plus the u-boosted
current outer product; the state is updated AFTER. Applying the update BEFORE the
readout is a different operator that disagrees with this one by ~u * (k v^T)
per step -- above the bf16 tolerance -- so a solver that swaps the order fails
the correctness gate.

This is written as a readable sequential scan in float32 and is deliberately slow.
The surface is forward+backward: autograd differentiates this scan, and the input
gradients dr, dk, dv, dw, du ARE the graded gradient oracle. Correctness is
defined as agreement with this function, never with any Triton or CUDA
implementation of it. Semantics ported from the library's own ground truth,
fla/ops/rwkv6/recurrent_naive.py::naive_recurrent_rwkv6 (read, never called or
vendored); shape convention is the [B, T, H, ...] layout the library's chunk_rwkv6
wrapper uses.
```

**Step 1 (G0).** read instruction.md, reference.py and kernel.py, and confirm the module imports and exposes chunked_rwkv6
  - establishes: chunked_rwkv6 is importable and is the entry point the grader calls
  - survives: an import that raises, or a submission that exits during import

**Step 2 (G2).** establish that the supplied starter is correct and deliberately slow
  - establishes: chunked_rwkv6 agrees with reference.py and is the baseline to beat
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
| target fraction of anchor | 0.72 |
| starter fraction of anchor (measured) | 0.008929 |
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
| `G0` trusted_entrypoint | the submitted module imports and exposes chunked_rwkv6. A candidate that calls sys.exit during import is trapped rather than terminating the grader. |
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
Taskdef: chunk-parallel RWKV-6 (a.k.a. WKV-6), forward+backward (fwdbwd).

Member of the chunked_linear_attn family. HARD variant: distinct from every other
member because RWKV-6 carries a data-independent per-(head, channel) BONUS `u`
that boosts the CURRENT-step outer product ONLY in the output -- the state
carried to t+1 does not see u. That asymmetry breaks the clean chunked
lower-triangular attention factorization at the diagonal: the intra-chunk kernel
must compute two closely-related terms per position (u-boosted on the diagonal,
plain elsewhere) instead of one.

Operator (reference.py is the definition; this is a summary):
    chunked_rwkv6(r[B,T,H,K], k[B,T,H,K], v[B,T,H,V], w[B,T,H,K], u[H,K])
        -> o[B,T,H,V]
    S_t = diag(exp(w_t)) S_{t-1} + k_t v_t^T;
    o_t = r_t^T (S_{t-1} + diag(u) . k_t v_t^T).
    Decay AFTER readout. Per-key-channel log-decay `w`. `u` shared across (B,T).
    SURFACE fwdbwd: gradients of r, k, v, w, u are graded, timed and
    determinism-checked.

Anchor: fla.ops.rwkv6.chunk_rwkv6 -- the production chunk kernel of the pinned
library. sota() calls it with KEYWORDS and scale=None (=> K^-0.5). Returns
(o, final_state); sota() unwraps the tuple.

STATUS: authored-draft -- NEEDS GPU CALIBRATION. TOL, TARGET_FRACTION_OF_SOTA, the
vendor-shingle ceiling, the anchor semantic-divergence check and every negative
control's bound-gate behaviour are INHERITED placeholders; see spec.yaml.

Distribution invariance (frontier-defeat-analysis §7.1): r, k are L2-normalized
along K; w log-space in (-inf, 0]; the hidden per-invocation scale multiplies v.
u is drawn small (Uniform(-0.2, 0.2)) so its contribution stays comparable to
the recurrent term. Timed tensors ARE checked tensors; NC-DP plants the §5.4
magnitude-branch exploit on the oracle under FORGE_ORACLE=1 and must zero at
exactly G2.
```

## Section 6: Fingerprint

| Artifact | sha256[:16] |
| --- | --- |
| `reference.py` | `5036f5a9d03bccce` |
| `kernel.py` | `5bcea6f4b2436711` |
| `bench.py` | `673e8b7084bf402b` |
| `instruction.md` | `f0db573274e4f333` |
| `hidden_shapes.json` | `e8f607afef9ab702` |
| `solution/oracle_kernel.py` | `9b8d97afa22442a9` |

- **seed_base:** `138217`

| Graded shape | dims | dtypes |
| --- | --- | --- |
| `g1` | B=1 T=4096 H=8 K=128 V=128 | bfloat16 |
| `g2` | B=4 T=1024 H=8 K=128 V=128 | bfloat16, float16 |
| `g3` | B=2 T=2048 H=16 K=64 V=64 | bfloat16 |
| `g4` | B=2 T=2048 H=8 K=128 V=256 | bfloat16 |
| `g5` | B=16 T=512 H=8 K=128 V=128 | bfloat16 |

- **forbidden symbols (26):** `fla.ops`, `import fla`, `from fla`, `chunk_rwkv6`, `fused_recurrent_rwkv6`, `naive_recurrent_rwkv6`, `naive_chunk_rwkv6`, `chunk_rwkv7`, `fused_recurrent_rwkv7`, `chunk_gla`, `chunk_simple_gla`, `chunk_gated_delta_rule`, `chunk_delta_rule`, `chunk_retention`, `chunk_gdn`, `chunk_kda`, `chunk_comba`, `chunk_gdn2`, `chunk_dplr_delta_rule`, `chunk_gated_delta_product`, `fused_recurrent_gla`, `fused_recurrent_simple_gla`, `fused_recurrent_gated_delta_rule`, `fused_recurrent_delta_rule`, `fused_recurrent_retention`, `fused_recurrent`

## What this document does not claim

It does not claim the operator is unsolvable, nor that the anchor is optimal. It
states the path the private oracle took, the values the metric was locked to, and
the checkers that decide the outcome. Every number above is measured or derived
from this bundle's own frozen inputs.
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 beb3630eb2386931f3095668f3f06d7b4e2f0627b5e45a26f7797d6475c30a0a
FORGE-CANARY-SLOT-1 95ad8a25a8c85262317d4d57b0fbd2237a8bb3ac27393641c70b056941aac641
FORGE-CANARY-SLOT-2 1462a0a75eeda29612d75c913ed08043ce8515d77567866dcba50c9dca58e168
FORGE-CANARY-SLOT-3 1dedf61af7593b69390069711c6dedc20133b4403d829b53e6875b0a1e3af5b3
FORGE-CANARY-END
