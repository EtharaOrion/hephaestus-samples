# Summary of what was done in previous iterations

## Approaches tried before, and their respective scores

  Both numbers are measured. The approach text is the agent's own account of that attempt and is UNVERIFIED -- an attempt that failed may describe an optimisation that never actually landed.

  attempt   speed   score   what was tried
    1        --    0.0000   Replaced the starter with a custom autograd MoE pipeline using Triton router, expert GEMM, combine, and backward kernels, with expert-sorted dispatch and split high/low representations to improve float32 gradient accuracy.
                            config: The visible final configuration used expert-sorted GEMM tiles selected by `_cfg`, gemm1 with `nacc=2`, metadata construction with `BT=128`, and float32 operand splitting into high/low bf16 parts accumulated through three tensor-core dot products; no concrete BM/BN/BK, num_warps, or num_stages sweep and no post-edit timing numbers were recorded. Correctness was exercised at T=512,D=512,F=256,E=16,A=4 in bfloat16 and float32 and at non-power-of-two T=301,D=200,F=96,E=11,A=3, but the final graded float32 gradient failures remained.
    2        --    0.0000   Replaced the starter with a custom deterministic Triton MoE implementation using stable expert-major dispatch, tiled forward/backward expert GEMMs, and three-term bfloat16 hi/lo dot decomposition for float32-effective accumulation.
                            config: No completed tile-size, warp-count, stage-count, or chunk-size sweep is evidenced; /tmp/tune_dw.py was created but never run. The only captured final measurement was p2/bfloat16 at 6.628 ms, while the correctness checks exercised A=4 and both bfloat16 and float32.

  speed  = fraction of the production kernel's throughput; 1.0 would match it. `--` means the attempt was never timed because it computed the wrong answer.
  score  = what this attempt actually scored. It EQUALS the speed above when every gate and rule passed, and is 0.0000 if anything at all was wrong with the attempt -- so the two columns differ only on failures.
  config = the tuning values that attempt actually ran (tile sizes, warps, stages) and what its sweeps showed. Do not re-search values already listed; start from what they learned.
  Nothing has earned a score above 0.0000 yet, so there is no best attempt to beat. The speed column still shows how far each one got.

  EXPLORE A NEW IDEA, OR EXPLOIT ONE THAT ALREADY PERFORMS -- your call, and say which in your final message. Exploit means taking an approach above that already ran and making it faster, or fixing what stopped it. Explore means a structurally different idea, worth it only if the ones that ran have stopped improving. Repeating an approach that scored 0.0000, without changing what made it score 0.0000, is neither.

## Rules this session has already broken

  deterministic_routing_and_combine  (iteration 1)
  exact_selection_semantics  (iteration 1)
  routing_primitive_scope  (iteration 1)

  These cost a full zero each time, whatever the kernel measured. Check your submission against every line here before you finish, not only against the one the last attempt broke.

## Where iteration 2 fell short

  g1/float32 scale=2.432: drouter_w[13 elements outside tolerance, max abs 6.812e-02]
  g3/float32 scale=1.573: drouter_w[1 elements outside tolerance, max abs 4.565e-02] dw1[31 elements outside tolerance, max abs 1.546e-02] dw2[40 elements outside tolerance, max abs 1.617e-02]
  c1/float32 scale=2.053: drouter_w[1 elements outside tolerance, max abs 2.625e-02]

  This same failure has now happened 2 times (attempts 1, 2). Whatever you have been changing between attempts is not what is causing it.

## What iteration 2 said about itself (UNVERIFIED)

  what it says stopped it : The float32 path computed inaccurate gradients: drouter_w failed for g1, g3, and c1 with maximum absolute errors of 6.812e-02, 4.565e-02, and 2.625e-02 respectively, while g3 also had 31 incorrect dw1 elements up to 1.546e-02 and 40 incorrect dw2 elements up to 1.617e-02; the three-term _dot3 approximation is the evident precision gap because it drops the low*low product.
  what it wanted to try next: Add the omitted bfloat16 low*low tensor-core dot to _dot3, or replace that path with an IEEE-equivalent float32 dot, and use it consistently in router forward and all weight/router-gradient reductions to remove the observed float32 accumulation error.

  This is the attempt's own account, not a measurement. Treat the next idea as a lead worth checking, not as a conclusion -- and if you disagree with it, say why and do something else.

## What your implementation actually is (iteration 2, assessed, NOT part of your score)

  yes  diag_dispatch_strategy
  NO   diag_measured_before_submit
  NO   diag_router_backward
  NO   diag_tie_handling
  NO   diag_weight_grad_precision

  The 'NO' answers are why you are slow. They do not lower your reward directly -- the clock already does that -- but each one names a structural property the fast implementations have and this one does not:
    [diag_tie_handling] In `_router_kernel`, `key = (u << 31) | (E - 1 - re[None, :]).to(tl.int64)` creates a composite integer key whose low bits invert the expert index, so `tl.max` implements biased-score descending with lower expert index first; however, the trajectory only shows generic runs such as `[33] timeout 900 python /tmp/chk.py`, `[36] timeout 900 python /tmp/chk.py`, and `[37] timeout 1800 ./bench.py`, with no visible exact-tie shape or assertion, so there is no evidence the required exact-tie case was tested before submission.
    [diag_weight_grad_precision] `_dw1_kernel` and `_dw2_kernel` call `_dot3`, whose code performs `tl.dot(ah, bh, acc)`, then optional `tl.dot(al, bh, acc)` and `tl.dot(ah, bl, acc)` after bf16 hi/lo splitting; this identifies the weight-gradient strategy as three bf16 tensor-core dots with fp32 accumulation, but the only accuracy claim is the module comment `reconstructs the product with float32-effective accuracy`, with no specific disclosed gradient tolerance cited or used to justify the strategy.
    [diag_router_backward] In `_dweights_kernel`, `dl = (rd - inner) / ssum * sv * (1.0 - sv)` propagates through selected-weight normalization and sigmoid, while `acc += tl.where(e[:, None] == re[None, :], dl[:, None], 0.0)` writes router-logit gradients only at experts in `SEL`, matching the gather graph; however, the trajectory provides no visible numerical comparison, finite-difference/gradcheck, or reported error specifically validating `drw`/`drouter_w` (steps 33 and 36 only show truncated warning output), so the required validation is not evidenced.
    [diag_measured_before_submit] Trajectory step [37] shows `timeout 1800 ./bench.py 2>&1 | tail -25`, but the captured output contains only a NumPy warning and no benchmark results; therefore there is no evidence that any reported numbers came from a gated `bench.py` run on the final `kernel.py` bytes.

## What to do next

THE SAME WRONG-ANSWER BUG HAS NOW KILLED 2 ATTEMPTS (iterations 1, 2):
    g1/float32 scale=2.432: drouter_w[15 elements outside tolerance, max abs 7.959e-02]

It has recurred across implementations that share almost no code with each other, so it is not a property of any one design and rewriting from scratch will not remove it. You are shipping a kernel whose gradients you have not checked.

`./bench.py` compares your output AND all five gradients against `reference.py` before it times anything, and refuses to print a speed number when they disagree. Run it. Do not submit until it prints `correctness ok`.

A forward pass can be right while the backward is garbage: that is this exact failure, every time. If bench.py reports correctness ok and the graded run still fails you, say so explicitly in your final message -- the difference is then the shape, not the algebra, and that is worth knowing.

## Rules that hold for every iteration

  - Your final edit to kernel.py must be followed by a bench.py run. Any number you report must come from a run on the EXACT bytes you shipped.
  - Do not report a speedup measured on a variant you did not submit. Your numbers are re-measured independently and a mismatch is a violation of the honest-measurement requirement, not a rounding difference.
  - Time the combined forward-and-backward pass with CUDA events or an explicit synchronise after discarded warmup runs. Forward-only timing is a violation.
