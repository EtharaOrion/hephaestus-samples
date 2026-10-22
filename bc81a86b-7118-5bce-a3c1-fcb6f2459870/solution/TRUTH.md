# Golden Solve Path — kda_chunk_gdr_fwdbwd

## Task

make `gated_delta_rule` in `kernel.py` — the chunk-parallel gated delta rule — as fast as the production chunked implementation, without changing what it computes.

- **Surface:** forward + backward (the output and all five input gradients `dq, dk, dv, dg, dbeta` are graded).
- **Signature:** `gated_delta_rule(q, k, v, g, beta)`, fixed, with autograd working through it.
- **Editable:** `kernel.py` only. `reference.py` is the read-only correctness oracle.
- **Budget:** 100 attempts within 1.5 hours, one H100.
- **Score:** one float in `[0, 1]`, the geometric-mean fraction of the production implementation's speed across the hidden graded shapes. It climbs continuously toward parity and saturates at 1.0; an invalid attempt scores zero.

---

## The path to max score

Start from the recurrence, not the code. The starter walks `T` timesteps, each performing a rank-1 state update `S <- g_t S + k_t (v_t - k_t^T S beta_t)^T` followed by a read `q_t S`. That is `T` host-visible sequential steps, so every timestep pays a kernel launch and a DRAM round trip on a state that never leaves memory. The route to parity is to rewrite the same algebra so the sequential axis becomes `T/BT` chunks of dense matmuls.

- **Chunk the recurrence.** Replace the per-timestep loop with a loop over chunks, taking the launch count to `O(T/BT)` and expressing the intra-chunk term as matmuls: a dense intra-chunk interaction plus a cross-chunk state carry. The chunked forward must still match the reference under randomized input scale, on non-power-of-two sequence lengths, in both graded dtypes.
- **Solve the intra-chunk system on the tile.** The gated delta rule's intra-chunk coupling is a `BT x BT` triangular system; solve it once per chunk on that tile (WY form) rather than iterating it per timestep.
- **Write the backward yourself.** Do not lean on autograd through the chunk loop. The backward mirrors the forward with the scan reversed: a reverse-order chunk pass accumulates `dS`, solves the transposed system on the same `BT x BT` tile, recomputes the gate cumulative sum from `log g` rather than storing it, and closes with a reverse cumsum for `dg`. Keep every per-chunk step inside Triton kernels, with no Python loop inside the autograd graph.
- **Keep the device time in your own kernels.** Graded device time must be dominated by `triton.jit` kernels the submitted module itself declares. A framework or compiler path emits kernels under generated names that cannot match a declared name, so it does not count toward the written-kernel share.
- **Tune, then keep only what holds.** Sweep tile sizes, `num_warps` and `num_stages` against the production kernel on the published shapes, and keep only changes that survive the hidden set. The published shapes in `bench.py` are smaller than and different from the graded shapes, so a candidate tuned only to what it can see does not transfer.

Read the production implementation to understand the chunked formulation if you wish, but never import or call it: the submitted source is scanned for forbidden symbols with comments stripped, so citing a symbol is legal and calling it is not. Submit a module that imports cleanly and computes from its actual arguments on every call.

**Convergence:** every gate clean on all 15 configurations, no anti-cheat rubric failed, and the score is the geometric-mean fraction of the production kernel, saturating at 1.0 at parity. The 0.95 target is informational and does not gate: the fraction is paid directly, and only a failed gate pays zero.

---

## Reference values

```
  SCORE                   = geometric-mean fraction of the production implementation    zero for an invalid attempt
  TARGET_FRACTION         = 0.95       informational; the score is the raw fraction, not divided by it
  STARTER_FRACTION        = 0.008235   the shipped starter, measured on the hidden graded set
  WRITTEN_KERNEL_FLOOR    = 0.60       min share of forward+backward device time in your own triton.jit kernels
  FRONTIER_DEFEAT_FLOOR   = 0.15
  BFLOAT16_TOLERANCE      = atol 0.02,  rtol 0.03125
  FLOAT16_TOLERANCE       = atol 0.01,  rtol 0.00390625
  FLOAT32_TOLERANCE       = atol 0.002, rtol 0.002        verified, never timed
  GRADED_TENSORS          = [out, dq, dk, dv, dg, dbeta]
  DTYPES_GRADED           = [bfloat16, float16]
  DTYPES_VERIFIED         = [bfloat16, float16, float32]
  GRADED_SHAPES           = 7  (8 shape x dtype configurations)
  CORRECTNESS_ONLY_SHAPES = 3  (7 configurations)
  CORRECTNESS_STAGES      = 5  (smoke, shape sweep, stability, determinism, edge cases)
  DETERMINISM             = 3 runs, bitwise identical incl. gradients, after a warmup
  RUBRIC_GATE             = every rubric must pass
  VENDOR_CONTAINMENT      = 0.40 ceiling on 32-token structure, identifiers erased
  SEED_BASE               = 8817
  ATTEMPT_CAP             = 100
  MAX_TIMEOUT_SEC         = 5400.0
```
