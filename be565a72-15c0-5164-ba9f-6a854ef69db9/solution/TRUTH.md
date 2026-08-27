# Golden Solve Path — moe_routed_fwdbwd

## Task

make fused_moe in kernel.py — a routed fused mixture-of-experts layer whose routing decision lives inside the operator — as fast as the production grouped-GEMM composition, without changing what it computes.

- **Surface:** forward + backward (the output `y` and all four gradients `dx, drouter_w, dw1, dw2` are graded; `router_b` and `A` carry no gradient).
- **Editable:** `kernel.py` only. `reference.py` is the read-only correctness oracle.
- **Budget:** 100 attempts within 1.5 hours, one H100, no network.
- **Score:** one float in `[0, 1]`, the geometric-mean fraction of the production composition's speed across the hidden graded shapes. It climbs continuously toward production parity; an invalid attempt scores zero.

---

## The path to max score

Start from the operator, not the starter. `reference.py` scores every token with `s = sigmoid(x @ router_w)` in float32, selects the top-`A` experts by the biased score `s + router_b` with ties broken to the lower expert index, normalizes the raw selected scores into combine weights `w_a = s_sel / sum_a s_sel`, and runs each expert as SwiGLU `(silu(h @ w1[:, :F]) * (h @ w1[:, F:])) @ w2`, combining `y[t] = sum_a w_a f(sel_a, x[t])`. The starter realises this as a Python loop over experts with boolean-mask gathers — one separate dense matmul chain per expert, most of them tiny — so it is launch-bound and re-reads the tokens once per expert. The route to parity is to dispatch every (token, expert) pair exactly once and run the expert compute as a few large segmented GEMMs.
 - Route in one kernel. Compute the router logits with one dense matmul, then a single Triton kernel per token row: sigmoid in float32, pack each biased score's order-preserving float32 bits together with the inverted expert index into one integer key so ties can never exist, extract the top-`A` in registers, and normalize the raw scores of the selected experts. The bias enters the keys only, never the stored weights, and carries no gradient. - Dispatch by sorting. Stable-argsort the `[T*A]` expert ids — the permitted `torch.sort`/`argsort` carve-out on routing metadata — derive per-expert segment bounds, and build a tile table so each 64-row tile of an expert's token segment becomes one program of the segment GEMMs. - Compute in segmented GEMMs. One shared segment-GEMM kernel does the gate|up projection with a fused token gather, the down projection, and the two backward data GEMMs against the transposed weights; a shared segment-dW kernel accumulates the weight gradients; a SwiGLU elementwise pair sits between them; ordered per-token combine kernels write `y` in ascending slot order so the result is bitwise reproducible. - Close the router backward. Combine-weight gradients feed the normalization backward, then the sigmoid backward at the selected entries only, then two small dense matmuls for `drouter_w` and the router half of `dx`.
 Keep float32 accumulation everywhere, and carry the operands feeding the weight gradients at float32-effective precision (an exact bf16 hi/lo pair on the tensor cores) so the long per-expert row sums do not decay into noise outside the graded tolerance. No capacity limit, no dropped tokens: every selected (token, expert) pair is computed, and an expert no token selects still produces exact zero weight gradients.
 Finish by tuning the tile sizes, `num_warps` and `num_stages` per shape across the sweep, and by re-deriving every formula from `reference.py` so no grouped-GEMM library symbol is imported or called.

**Convergence:** the reward is the geometric-mean fraction of the production composition — the router matmul plus integer-key top-`A` in torch feeding a `torch._grouped_mm` expert pipeline over tokens sorted by expert — across the hidden graded shapes in both float32 and bfloat16. It climbs continuously with speed and saturates at production parity. The reward is zero for an invalid attempt: a wrong output or gradient, a submission that calls a grouped-GEMM or MoE-serving library, less than 60% of graded device time in your own declared Triton kernels, or a trajectory that games the measurement.

---

## Reference values

```
  SCORE                   = geometric-mean fraction of the production composition     zero for an invalid attempt
  TARGET_FRACTION         = 0.34       informational frontier reference; score is the raw fraction, not divided by it
  STARTER_FRACTION        = 0.083188   the shipped starter, measured on the hidden graded set
  WRITTEN_KERNEL_FLOOR    = 0.60       min share of forward+backward device time in your own triton.jit kernels
  FRONTIER_DEFEAT_FLOOR   = 0.15
  OUTPUT_TOLERANCE        = float32 atol/rtol 2e-4;   bfloat16 atol/rtol 4.5e-2
  GRADIENT_TOLERANCE      = float32 atol/rtol 4e-3;   bfloat16 atol/rtol 4.5e-2
  SELECTION_MARGIN        = A-th vs (A+1)-th biased score exactly 0 (a tie) or >= 1e-4;  float32 scores required
  TIE_BREAK               = lower expert index first;  +/-0.0 form one tie group
  GRADED_TENSORS          = [y, dx, drouter_w, dw1, dw2]      router_b and A carry no gradient
  DTYPES_GRADED           = [float32, bfloat16]      both checked and timed
  DETERMINISM             = 3 runs on one tie-forcing input, bitwise identical incl. gradients, after a warmup
  ROUTING_CARVE_OUT       = torch.topk / torch.sort / torch.argsort on [T, E] scores or [T*A] routing metadata only
  FORBIDDEN               = torch._grouped_mm, _grouped_mm, vllm, sglang, grouped_gemm, transformer_engine, flashinfer
  SEED_BASE               = 74317
  ATTEMPT_CAP             = 100
  MAX_TIMEOUT_SEC         = 5400.0
```
FORGE-CANARY-BEGIN forge-canary/v1
FORGE-CANARY-SLOT-0 8215a22e4225cfc30e275ec814e7ddb124a14cd403662c0e21e1589362f4731a
FORGE-CANARY-SLOT-1 f5410e9508bb5f2fef3c8124cea98af61d159d201a1609420842427ac38b4278
FORGE-CANARY-SLOT-2 cdd60bafda970634ffb7c64018f6a7d1506f9afb2c3f710b5609f3c3b0444e4b
FORGE-CANARY-SLOT-3 07b9e669b7aba77eae0eda15405545bc94fa27b78418c7644e3e61266f38da52
FORGE-CANARY-END
