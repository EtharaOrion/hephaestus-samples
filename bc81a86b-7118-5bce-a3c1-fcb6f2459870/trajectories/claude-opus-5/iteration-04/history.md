# Summary of what was done in previous iterations

## Approaches tried before, and their respective rewards

  Both numbers are measured. The approach text is the agent's own account of that attempt and is UNVERIFIED -- an attempt that failed may describe an optimisation that never actually landed.

  attempt   speed   score   what was tried
    1        --    0.0000   Replaced the implementation with a custom Triton fused-chunk gated-delta-rule using chunkwise gate cumsums, an exact WY triangular inverse, float32 recurrent-state carry, dense chunk output products, and a Triton backward scan.
                            config: No performance sweep was recorded. The visible static dispatch used NS=1 for inputs wider than 2 bytes and NS=2 otherwise; BV=min(64 if KP<=128 else 32, pow2(V)); KP and VP were next powers of two; the triangular inverse used BS=16 in the visible branch; float32 selected IEEE dot precision and a shallower pipeline. Functional runs covered bfloat16/float16/float32 at B=1,T=128,H=HV=2,K=V=64 and B=1,T=127,H=HV=3,K=V=96, plus bfloat16 at B=2,T=300,H=2,HV=4,K=64,V=128; those local cases passed, but the harness's g5 bfloat16 case did not.
    2        --    0.0000   Replaced the starter with a BT=64 chunk-parallel Triton implementation that forms a dense triangular WY inverse per chunk, carries the recurrent state across chunks, and implements a fully chunked custom backward.
                            config: Fixed BT=64 with NLEV=6 and BK=next_power_of_two(K); initially used BV=min(64,next_power_of_two(V)), num_warps=4, and generally num_stages=2, which hit shared-memory failures of 270336 and then 253952 bytes on B=1,T=127,H=3,K=96,V=96 bfloat16 while resource edits were being made. The final policy was wide=(element_size>2), BV=min(32 if wide else 64,next_power_of_two(V)), and NS=1 for float32 versus NS=2 for bfloat16/float16, with num_warps=4; thus bfloat16 still used BV=64/NS=2 and later required 279040 bytes on g5. Program decomposition included WY and input-gradient work over (NT,NBH), state work over (ceil(V/BV),NBH), and output work over (NT,ceil(V/BV),NBH). IEEE versus tf32x3 dot precision was also tried; tf32x3 retained the reported base correctness. No latency or block-size sweep with numerical timing results was reported.
    3 *    0.8503  0.8503   Replaced the starter with a fully Triton chunk-parallel gated-delta implementation using BT=64 exact block-doubling WY inversion, float32 chunk state, dense chunk matmuls, and forward/reverse chunk scans for forward and backward.
                            config: BT=64 with NLEV=6 was used throughout. For K=V=96, the initial BK=128/BV=64 configuration exceeded H100 shared memory (303104 bytes); changing the selector to BK=128/BV=32 still produced 241664-byte and later 249856/262144-byte failures in backward kernels. Backward launches were tried with num_warps=8 and num_warps=4, but both still hit resource failures before diag_bwd_inputs was split/capped to BK=64; the final split launch used BK=min(BK,64), num_warps=4, num_stages=1, while another backward launch was restored from 4 to 8 warps. No successful speed sweep over BT, BK, BV, warps, or stages was run.

  speed  = fraction of the production kernel's throughput; 1.0 would match it. `--` means the attempt was never timed because it computed the wrong answer.
  score  = what this attempt actually scored. It EQUALS the speed above when every gate and rule passed, and is 0.0000 if anything at all was wrong with the attempt -- so the two columns differ only on failures.
  config = the tuning values that attempt actually ran (tile sizes, warps, stages) and what its sweeps showed. Do not re-search values already listed; start from what they learned.
  *      = best score so far, and the one to beat.

  EXPLORE A NEW IDEA, OR EXPLOIT ONE THAT ALREADY PERFORMS -- your call, and say which in your final message. Exploit means taking an approach above that already ran and making it faster, or fixing what stopped it. Explore means a structurally different idea, worth it only if the ones that ran have stopped improving. Repeating an approach that scored 0.0000, without changing what made it score 0.0000, is neither.

## Rules this session has already broken

  "Keep the recurrent state and the transition-inverse accumulator in"  (iteration 1)

  These cost a full zero each time, whatever the kernel measured. Check your submission against every line here before you finish, not only against the one the last attempt broke.

## What iteration 3 said about itself (UNVERIFIED)

  what it says stopped it : The attempt spent its remaining work eliminating compilation, shared-memory, precision, and stability failures and never profiled or tuned the now-correct implementation; performance remained especially poor for T=16384,B=1 and T=256,B=64, while the code serializes all NT chunks inside each forward state-scan program and all NT chunks in reverse inside each backward state-scan program, and no kernel-level timing was collected to establish which scan or auxiliary kernel dominates.
  what it wanted to try next: Profile the forward and reverse state-scan kernels first, then replace their single-program serial NT traversal with a hierarchical/segmented chunk-state scan so long sequences expose inter-chunk parallelism instead of carrying the state through every chunk in one program.

  This is the attempt's own account, not a measurement. Treat the next idea as a lead worth checking, not as a conclusion -- and if you disagree with it, say why and do something else.

## Where the time is going (best iteration so far, 8 timed cases)

  Your best case reaches 1.0000 of the production kernel. Your worst reaches 0.4094. That is a 2.4x spread across cases that differ only in shape.

  worst case: yours 5.49 ms against production 2.23 ms -- 2.5x slower
  best case : yours 1.50 ms against production 1.68 ms -- 0.9x slower
  the worst case is 3.7x the wall-clock of your best case
  The spread is narrow, so every case is limited by the same thing. Making the kernel fundamentally cheaper will move all of them together.

## What your implementation actually is (iteration 3, assessed, NOT part of your score)

  yes  diag_bwd_chunked
  yes  diag_chunked_matmul
  NO   diag_gate_cumsum
  yes  diag_wy_form

  The 'NO' answers are why you are slow. They do not lower your reward directly -- the clock already does that -- but each one names a structural property the fast implementations have and this one does not:
    [diag_gate_cumsum] In `diag_gate_cumsum`, `b_g = tl.load(g + p, mask=m_t, other=0.0).to(tl.float32)` followed by `tl.store(aa + p, tl.cumsum(b_g, axis=0) * 1.4426950408889634, mask=m_t)` does form a per-chunk float32 cumulative log-gate representation, but consumers such as `b_ea = tl.exp2(b_aa)`, `b_edk = tl.exp2(atot - b_aa)`, and `tl.exp2(b_aa[:, None] - b_aa[None, :])` apply the exponential without any clamp or other bound on the exponent argument, violating the required bounded-before-exp condition.

## What to do next

THE APPROACH ABOVE WORKED: correct, your own Triton, and it reached 0.8503 of the production kernel's throughput. You are starting from the shipped starter again, so re-implement that approach rather than inventing a new one, then make it faster. Its description and its speed are in the table above.

## Rules that hold for every iteration

  - Your final edit to kernel.py must be followed by a bench.py run. Any number you report must come from a run on the EXACT bytes you shipped.
  - Do not report a speedup measured on a variant you did not submit. Your numbers are re-measured independently and a mismatch is a violation of the honest-measurement requirement, not a rounding difference.
  - Time the combined forward-and-backward pass with CUDA events or an explicit synchronise after discarded warmup runs. Forward-only timing is a violation.
