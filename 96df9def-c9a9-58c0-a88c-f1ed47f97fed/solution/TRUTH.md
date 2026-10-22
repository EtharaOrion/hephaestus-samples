# Golden Solve Path — chunked_dplr_delta

## Task

make `chunked_dplr_delta` in `kernel.py` — the chunk-parallel DPLR (diagonal-plus-low-rank) generalized delta rule at the core of RWKV-7 — as fast as the production `fla` chunked implementation, without changing what it computes.

- **Surface:** forward + backward (the output `o` and all six gradients `dq, dk, dv, dalpha, dbeta, dgk` are graded).
- **Signature:** `chunked_dplr_delta(q[B,T,H,K], k[B,T,H,K], v[B,T,H,V], alpha[B,T,H,K], beta[B,T,H,K], gk[B,T,H,K]) -> o[B,T,H,V]`.
- **Editable:** `kernel.py` only. `reference.py` is the read-only correctness oracle.
- **Budget:** 50 attempts within 6 hours, one H100.
- **Score:** one float in `[0, 1]`, the geometric-mean fraction of the production implementation's speed across the hidden graded shapes. It climbs continuously toward production parity; an invalid attempt scores zero.

---

## The path to max score

Start from the recurrence, not the starter. For a single `(batch, head)` with state `S` in `R^{K x V}`, `reference.py` advances

```
    S_t = diag(exp(gk_t)) . S_{t-1}  +  k_t v_t^T  +  beta_t . ( S_{t-1}^T alpha_t )^T
    o_t = S_t^T ( q_t / sqrt(K) )
```

as a readable sequential scan in float32. The third term is the DPLR correction: it reads a V-vector out of the decayed state **through `alpha`** and writes it back **through `beta`**. That is the whole difficulty. Setting `alpha == beta == 0` collapses the operator to gated linear attention, and setting `alpha == k, beta == -k*beta_scalar` collapses it to the delta-rule Householder form — both are graded-distribution wrong, and neither reproduces this oracle. There is no `v - S^T k` residual anywhere in a correct implementation; the state is never read through `k`.

The route to parity is to stop scanning per timestep and correct a whole chunk at a time in matrix form.

- **Build the intra-chunk matrices.** With a cumulative log-decay `c` over each chunk and its exclusive prefix `c_excl`, form `A_qk[i,j] = <q_i, k_j> exp(c_i - c_j)` and `A_qb[i,j] = <q_i, beta_j> exp(c_i - c_j)` on the inclusive lower triangle, and `A_ab[i,j] = <alpha_i, beta_j> exp(c_excl_i - c_j)`, `A_ak[i,j] = <alpha_i, k_j> exp(c_excl_i - c_j)` on the strict lower triangle. `A_ab` is the alpha-beta erasure coupling; it is strictly lower triangular, hence nilpotent.
- **Invert the coupling exactly, once per chunk.** The chunk factorization inverts the `(I + A_ab)` erasure matrix, as `reference.py` states, by a unit-lower triangular solve. Because `A_ab` is nilpotent, a Neumann/doubling expansion carried to termination is that same exact inverse, not an approximation — but a *truncated* series, a diagonal-only approximation, or a smaller effective chunk for the `ab` term is not, and is rejected.
- **Carry the state in two projected quantities.** Set `u = T_wy (A_ak v)` and `w = T_wy (alpha * exp(c_excl))`. Then per chunk `v2 = w @ S + u` is the corrected V-vector written through `beta`, and the carry is `S <- diag(exp(c_last)) S + (k * exp(c_last - c))^T v + (beta * exp(c_last - c))^T v2`. This is the chunk-collapsed form of the per-step rank-1 update; a per-timestep loop over the sequence computes the same thing and does not reach parity.
- **Emit the output from the same tiles.** `o = A_qk v + A_qb v2 + (q * exp(c)) @ S_chunk_start`, all as chunk-tile matmuls.
- **Close the backward the same way.** Reverse-scan the state adjoint by chunk, reuse `T_wy^T` for the `w`/`u` adjoints, and accumulate `dq, dk, dv, dalpha, dbeta, dgk` from the same tile structure rather than replaying timesteps.

Keep the recurrent state and the transition-inverse accumulator at float32 or wider. The graded distribution stresses long-context recurrence (`T = 8192` on `g1`), where the chunk-boundary state accumulates rank-1 error across roughly 128 carries; a bf16 boundary state loses precision at every carry and drifts outside tolerance well before `t = 8192`. Finish by tuning chunk size, tile widths, `num_warps` and `num_stages` per shape, and by re-deriving every formula from `reference.py` so no `fla` symbol is imported or called.

**Convergence:** the reward is the geometric-mean fraction of the production `fla` implementation across the hidden graded shapes. It climbs continuously with speed and saturates at production parity. The reward is zero for an invalid attempt: a wrong output or any wrong gradient, a submission that imports or calls the reference library, less than 60% of graded device time in your own declared Triton kernels, or a trajectory that games the measurement.

---

## Reference values

```
  SCORE                   = geometric-mean fraction of the production implementation    zero for an invalid attempt
  TARGET_FRACTION         = 0.63       informational frontier reference; score is the raw fraction, not divided by it
  STARTER_FRACTION        = 0.002783   the shipped starter, measured on the hidden graded set
  WRITTEN_KERNEL_FLOOR    = 0.60       min share of forward+backward device time in your own triton.jit kernels
  FRONTIER_DEFEAT_FLOOR   = 0.15
  OUTPUT_TOLERANCE        = float32 atol/rtol 2e-3;   bfloat16 atol/rtol 5e-2;   float16 atol 1e-2 / rtol 3.90625e-3
  GRADIENT_TOLERANCE      = same per-dtype tolerances; all six input gradients graded
  GRADED_TENSORS          = [o, dq, dk, dv, dalpha, dbeta, dgk]
  DTYPES_GRADED           = [bfloat16, float16, float32]      both checked and timed
  STATE_PRECISION         = recurrent state and transition-inverse accumulator float32 or wider
  ERASURE_OPERATOR        = exact inverse of (I + A_ab), whole chunk;  A_ab strictly lower triangular
  CORRECTION_BASIS        = state read through alpha, written through beta;  never read through k
  DEGENERATE_FORMS        = alpha==beta==0 -> GLA;  alpha==k, beta==-k*s -> Householder delta rule;  both wrong here
  SMOKE_SHAPE             = B=2  T=128  H=4  K=64  V=64
  DETERMINISM_SHAPE       = B=4  T=512  H=8  K=128 V=128
  STABILITY_SHAPE         = B=2  T=512  H=8  K=128 V=128
  STABILITY_MODES         = no_decay, max_decay, ab_zero, ab_large, v_dynamic, v_underflow
  DETERMINISM             = 3 runs on one input, bitwise identical incl. gradients, after a warmup
  LONG_CONTEXT_STRESS     = T = 8192 on graded shape g1  (~128 chunk-boundary carries)
  HIDDEN_SHAPES           = 5 graded, 4 correctness-only
  FORBIDDEN               = fla.ops, import fla, from fla, chunk_dplr_delta_rule, dplr_recurrence,
                            dplr_chunkwise, chunk_rwkv7, chunk_gated_delta_rule, chunk_gla, and the
                            rest of the fla chunked/fused_recurrent family
  SEED_BASE               = 528619
  ATTEMPT_CAP             = 50
  MAX_TIMEOUT_SEC         = 21600.0
```
