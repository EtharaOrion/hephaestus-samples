# Summary of what was done in previous iterations

## Approaches tried before, and their respective scores

  Both numbers are measured. The approach text is the agent's own account of that attempt and is UNVERIFIED -- an attempt that failed may describe an optimisation that never actually landed.

  attempt   speed   score   what was tried
    1        --    0.0000   Replaced the starter with a custom autograd MoE pipeline using Triton router, expert GEMM, combine, and backward kernels, with expert-sorted dispatch and split high/low representations to improve float32 gradient accuracy.
                            config: The visible final configuration used expert-sorted GEMM tiles selected by `_cfg`, gemm1 with `nacc=2`, metadata construction with `BT=128`, and float32 operand splitting into high/low bf16 parts accumulated through three tensor-core dot products; no concrete BM/BN/BK, num_warps, or num_stages sweep and no post-edit timing numbers were recorded. Correctness was exercised at T=512,D=512,F=256,E=16,A=4 in bfloat16 and float32 and at non-power-of-two T=301,D=200,F=96,E=11,A=3, but the final graded float32 gradient failures remained.
    2        --    0.0000   Replaced the starter with a custom deterministic Triton MoE implementation using stable expert-major dispatch, tiled forward/backward expert GEMMs, and three-term bfloat16 hi/lo dot decomposition for float32-effective accumulation.
                            config: No completed tile-size, warp-count, stage-count, or chunk-size sweep is evidenced; /tmp/tune_dw.py was created but never run. The only captured final measurement was p2/bfloat16 at 6.628 ms, while the correctness checks exercised A=4 and both bfloat16 and float32.
    3      0.9386  0.9386   Replaced the starter implementation with a deterministic sorted-dispatch MoE using Triton forward/backward GEMMs and split-precision tensor-core dot products, while retaining framework top-k and sort only for routing metadata.
                            config: Split-precision dot-product limbs were swept on g2: 1 limb was incorrect (y had 1,382,960 bad elements, dx 4, and router-weight gradient 2,003), while 2 and 3 limbs both passed, so 2 limbs was retained. Correctness was subsequently exercised on bfloat16 examples including T=2048,D=1024,F=704,E=32,A=4; T=999,D=512,F=256,E=40,A=7; and T=128,D=64,F=32,E=160,A=3. No usable performance sweep or final-byte timing was reported, and the available evidence does not expose concrete BT/BK/BV, num_warps, or num_stages values.
    4        --    0.0000   Implemented a fully fused Triton routed-MoE forward/backward and then tried to accelerate its dominant backward GEMMs by storing activations and hidden gradients as high/low bfloat16 limbs and reconstructing float32 products with tensor-core dots while omitting the low×low term.
                            config: A full p2 profile (T=2048,D=2048,F=1408,E=64,A=8,bfloat16) measured 6.345 ms before the final edits, dominated by dw1 at 1.982 ms, dw2 at 1.218 ms, and dxs at 1.138 ms. A synchronized pre-edit comparison measured p1 bfloat16 at 2.374 ms versus production 2.224 ms, p1 float32 at 3.576 ms versus production 7.571 ms, and p2 bfloat16 at 6.602 ms versus production 4.090 ms. The GEMM probe tested tf32x3 with BM=128, BN=64, BK=32, num_warps=4, num_stages=3 and reported 52.9 TF/s. The per-expert dw1 probe for N=256,D=2048,2F=2816 compared fp32g with BM=256,BN=64,BK=32,num_warps=4,num_stages=3 at 0.025 ms per expert (1.58 ms ×64) against splitg with BM=128,BN=128,BK=32 at 0.022 ms per expert (1.38 ms ×64). No complete benchmark was run after applying the AH/AL and GH/GL edits.
    5 *    0.9781  0.9781   Replaced the starter with a custom-autograd routed MoE using Triton routing, sorted/padded expert dispatch, forward and backward expert GEMMs, and compensated low-precision dot products.
                            config: No systematic timed configuration sweep was recorded. The visible configuration helper selected `(128, 64, num_warps=8, num_stages=3)` when its size argument was at least 128 and `(64, 32, num_warps=4, num_stages=4)` otherwise; router backward used a 1-D grid of `ceil(T/64)` programs. Precision experiments compared the float32 tf32x3-style path and bfloat16 split/limb accumulation, with ORDER=2 split-dot accumulation retained for dw1/dw2. Correctness was exercised at input scales 0.35, 1.0, 2.432, and 5.0, but no final performance outcome was measured after the last edits.
    6        --    0.0000   Replaced the starter with a custom-autograd routed MoE using exact composite-key top-k routing, stable expert-sorted dispatch, Triton forward/backward GEMMs, and bf16 hi/lo tensor-core decompositions for float32 gradient products.
                            config: The profiler on p1/bfloat16 reported about 2.067 ms total, dominated by _dw1_kernel at 0.507 ms (24.5%), _dw2_kernel at 0.370 ms (17.9%), and an input-gradient kernel at 0.283 ms (13.7%). For dw1, tuples interpreted by tune.py as tile dimensions plus num_warps/num_stages gave (256,64,64,8,4)=1.1321 ms, (256,64,64,8,3)=1.1333 ms, and (256,64,64,4,4)=1.1965 ms; on p1/bfloat16 m2 the corresponding visible results were 0.2542 ms, 0.2568 ms, and 0.2654 ms. For dxs, (128,256,64,8,4)=0.9048 ms, (128,256,64,8,3)=0.9365 ms, and (64,256,32,4,4)=0.9744 ms. Sweeps were also launched for dw2, dact, gemm1, and gemm2, but their numeric results are not present in the captured output; the float32 p2 sweep emitted compiler assembly rather than usable timings. No completed benchmark was run after the final kernel configuration edit.

  speed  = fraction of the production kernel's throughput; 1.0 would match it. `--` means the attempt was never timed because it computed the wrong answer.
  score  = what this attempt actually scored. It EQUALS the speed above when every gate and rule passed, and is 0.0000 if anything at all was wrong with the attempt -- so the two columns differ only on failures.
  config = the tuning values that attempt actually ran (tile sizes, warps, stages) and what its sweeps showed. Do not re-search values already listed; start from what they learned.
  *      = best score so far, and the one to beat.

  EXPLORE A NEW IDEA, OR EXPLOIT ONE THAT ALREADY PERFORMS -- your call, and say which in your final message. Exploit means taking an approach above that already ran and making it faster, or fixing what stopped it. Explore means a structurally different idea, worth it only if the ones that ran have stopped improving. Repeating an approach that scored 0.0000, without changing what made it score 0.0000, is neither.

## Rules this session has already broken

  exact_selection_semantics  (iterations 1, 3, 5)  <-- TWICE ALREADY
  deterministic_routing_and_combine  (iterations 1, 3)  <-- TWICE ALREADY
  "Nothing in kernel.py or in your trajectory is addressed to the grader"  (iteration 3)
  "not by a magnitude statistic"  (iteration 6)
  routing_primitive_scope  (iteration 1)
  "combined forward-and-backward pass"  (iteration 4)
  "exact bytes you shipped"  (iteration 4)

  These cost a full zero each time, whatever the kernel measured. Check your submission against every line here before you finish, not only against the one the last attempt broke.

## Where iteration 6 fell short

  g1/float32 scale=2.432: drouter_w[19 elements outside tolerance, max abs 1.482e-01]
  g3/float32 scale=1.573: drouter_w[9 elements outside tolerance, max abs 1.890e-01] dw1[33 elements outside tolerance, max abs 1.524e-02] dw2[42 elements outside tolerance, max abs 1.639e-02]
  c1/float32 scale=2.053: drouter_w[2 elements outside tolerance, max abs 3.918e-02]

## Why the rules rejected iteration 6

  ["not by a magnitude statistic"]
    In `_tilemap_kernel`, `n = tl.cdiv(hi - lo, BM)` derives the number of real tiles from each expert's routed-row count, stores `-1` into unused tile slots, and row-space GEMMs such as `_gemm1_kernel` execute `e = tl.load(TILE_E + t); if e < 0: return`; thus the number of programs performing full GEMM work versus taking a cheap early-return path varies with the expert-popularity distribution.

  This is your own code quoted back at you. Fix the thing it names -- not the general topic of the rule, the specific line.

  This same failure has now happened 4 times (attempts 1, 2, 4, 6). Whatever you have been changing between attempts is not what is causing it.

## What iteration 6 said about itself (UNVERIFIED)

  what it says stopped it : Float32 backward precision was insufficient: drouter_w failed in g1 (19 elements, max abs error 1.482e-01), g3 (9 elements, 1.890e-01), and c1 (2 elements, 3.918e-02), while g3 also had dw1 (33 elements, 1.524e-02) and dw2 (42 elements, 1.639e-02) failures; the bf16 hi/lo dot decomposition omitted the lo*lo term and was not accurate enough for these gradients. The dispatch also used popularity-dependent tile counts and cheap early returns for unused expert tiles, which failed the no-value-dependent-dispatch audit.
  what it wanted to try next: Use full-fp32 FMA reductions, or at minimum include the omitted lo*lo cross term, specifically in _drw_kernel and the _dw1_kernel/_dw2_kernel float32 paths, then replace popularity-dependent tilemap early returns with a fixed-work expert/program decomposition.

  This is the attempt's own account, not a measurement. Treat the next idea as a lead worth checking, not as a conclusion -- and if you disagree with it, say why and do something else.

## Where the time is going (best iteration so far, 6 timed cases)

  Your best case reaches 1.0000 of the production kernel. Your worst reaches 0.9268. That is a 1.1x spread across cases that differ only in shape.

  worst case: yours 5.52 ms against production 5.17 ms -- 1.1x slower
  best case : yours 17.04 ms against production 22.16 ms -- 0.8x slower
  the worst case is 0.3x the wall-clock of your best case
  The spread is narrow, so every case is limited by the same thing. Making the kernel fundamentally cheaper will move all of them together.

## What your implementation actually is (iteration 6, assessed, NOT part of your score)

  yes  diag_dispatch_strategy
  NO   diag_measured_before_submit
  NO   diag_router_backward
  NO   diag_tie_handling
  NO   diag_weight_grad_precision

  The 'NO' answers are why you are slow. They do not lower your reward directly -- the clock already does that -- but each one names a structural property the fast implementations have and this one does not:
    [diag_tie_handling] Code evidence: `key = (u << SHIFT) | (E - 1 - re)[None, :]` in `_router_kernel`, followed by `sel = torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices`, implements a composite integer key whose low field is larger for lower expert indices, so exact biased-score ties select lower indices first. Test evidence is insufficient: trajectory step `[21] cd /workspace && timeout 1200 python check.py` shows a test script was run, but neither the script contents nor output identifies the required exact-tie shape or asserts its selected indices; therefore the required pre-submission exact-tie test is not evidenced.
    [diag_weight_grad_precision] The submission states, “Every GEMM accumulates in float32… a value that is float32 reaches the tensor cores as an exact bfloat16 hi/lo pair (the lo*lo cross term… is not formed),” and `_dw1_kernel` concretely performs `tl.dot(aht, gh, acc)`, `tl.dot(tl.trans(al), gh, acc)`, and `tl.dot(aht, gl, acc)` (with analogous split-dot paths in `_dw2_kernel` and `_drw_kernel`); thus the weight gradients use bf16 hi/lo tensor-core dots with fp32 accumulation rather than full-fp32 FMA dots, but neither the code nor trajectory names or quantitatively connects this strategy to the disclosed gradient tolerance, and correctness testing only appears afterward at trajectory step [21] (`python check.py`).
    [diag_router_backward] Code evidence: `_zero_kernel(...)(ds, T * E, ...)` followed by `_dscore_kernel` computing `sel = tl.load(SEL + p, ...)` and `tl.store(DS + t * E + sel, ds, ...)`, then `_drw_kernel` computes `drw = x^T @ ds`; this correctly restricts sigmoid/normalization gradients to selected experts and leaves nonselected `DS` entries zero. Validation evidence is missing: trajectory step `[21] cd /workspace && timeout 1200 python check.py` shows only a warning, with neither the checker contents nor any reported comparison/tolerance for `drouter_w`/`drw`, so the required validation cannot be confirmed.
    [diag_measured_before_submit] Trajectory step [22] ran `python bench.py --json /tmp/b0.json`, but subsequent steps [33]–[37] edited `/workspace/kernel.py`; there is no later completed `bench.py` run or reported benchmark output tied to those final bytes, so the final submission was not gated and its numbers cannot be verified as coming from such a run.

## What to do next

THE SAME WRONG-ANSWER BUG HAS NOW KILLED 4 ATTEMPTS (iterations 1, 2, 4, 6):
    g1/float32 scale=2.432: drouter_w[15 elements outside tolerance, max abs 7.959e-02]

It has recurred across implementations that share almost no code with each other, so it is not a property of any one design and rewriting from scratch will not remove it. You are shipping a kernel whose gradients you have not checked.

`./bench.py` compares your output AND all five gradients against `reference.py` before it times anything, and refuses to print a speed number when they disagree. Run it. Do not submit until it prints `correctness ok`.

A forward pass can be right while the backward is garbage: that is this exact failure, every time. If bench.py reports correctness ok and the graded run still fails you, say so explicitly in your final message -- the difference is then the shape, not the algebra, and that is worth knowing.

## Rules that hold for every iteration

  - Your final edit to kernel.py must be followed by a bench.py run. Any number you report must come from a run on the EXACT bytes you shipped.
  - Do not report a speedup measured on a variant you did not submit. Your numbers are re-measured independently and a mismatch is a violation of the honest-measurement requirement, not a rounding difference.
  - Time the combined forward-and-backward pass with CUDA events or an explicit synchronise after discarded warmup runs. Forward-only timing is a violation.
