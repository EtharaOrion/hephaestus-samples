# Summary of what was done in previous iterations

## Approaches tried before, and their respective rewards

  Both numbers are measured. The approach text is the agent's own account of that attempt and is UNVERIFIED -- an attempt that failed may describe an optimisation that never actually landed.

  attempt   speed   score   what was tried
    1        --    0.0000   Replaced the implementation with a custom Triton fused-chunk gated-delta-rule using chunkwise gate cumsums, an exact WY triangular inverse, float32 recurrent-state carry, dense chunk output products, and a Triton backward scan.
                            config: No performance sweep was recorded. The visible static dispatch used NS=1 for inputs wider than 2 bytes and NS=2 otherwise; BV=min(64 if KP<=128 else 32, pow2(V)); KP and VP were next powers of two; the triangular inverse used BS=16 in the visible branch; float32 selected IEEE dot precision and a shallower pipeline. Functional runs covered bfloat16/float16/float32 at B=1,T=128,H=HV=2,K=V=64 and B=1,T=127,H=HV=3,K=V=96, plus bfloat16 at B=2,T=300,H=2,HV=4,K=64,V=128; those local cases passed, but the harness's g5 bfloat16 case did not.
    2        --    0.0000   Replaced the starter with a BT=64 chunk-parallel Triton implementation that forms a dense triangular WY inverse per chunk, carries the recurrent state across chunks, and implements a fully chunked custom backward.
                            config: Fixed BT=64 with NLEV=6 and BK=next_power_of_two(K); initially used BV=min(64,next_power_of_two(V)), num_warps=4, and generally num_stages=2, which hit shared-memory failures of 270336 and then 253952 bytes on B=1,T=127,H=3,K=96,V=96 bfloat16 while resource edits were being made. The final policy was wide=(element_size>2), BV=min(32 if wide else 64,next_power_of_two(V)), and NS=1 for float32 versus NS=2 for bfloat16/float16, with num_warps=4; thus bfloat16 still used BV=64/NS=2 and later required 279040 bytes on g5. Program decomposition included WY and input-gradient work over (NT,NBH), state work over (ceil(V/BV),NBH), and output work over (NT,ceil(V/BV),NBH). IEEE versus tf32x3 dot precision was also tried; tf32x3 retained the reported base correctness. No latency or block-size sweep with numerical timing results was reported.

  speed  = fraction of the production kernel's throughput; 1.0 would match it. `--` means the attempt was never timed because it computed the wrong answer.
  score  = what this attempt actually scored. It EQUALS the speed above when every gate and rule passed, and is 0.0000 if anything at all was wrong with the attempt -- so the two columns differ only on failures.
  config = the tuning values that attempt actually ran (tile sizes, warps, stages) and what its sweeps showed. Do not re-search values already listed; start from what they learned.
  Nothing has earned a score above 0.0000 yet, so there is no best attempt to beat. The speed column still shows how far each one got.

  EXPLORE A NEW IDEA, OR EXPLOIT ONE THAT ALREADY PERFORMS -- your call, and say which in your final message. Exploit means taking an approach above that already ran and making it faster, or fixing what stopped it. Explore means a structurally different idea, worth it only if the ones that ran have stopped improving. Repeating an approach that scored 0.0000, without changing what made it score 0.0000, is neither.

## Rules this session has already broken

  "Keep the recurrent state and the transition-inverse accumulator in"  (iteration 1)

  These cost a full zero each time, whatever the kernel measured. Check your submission against every line here before you finish, not only against the one the last attempt broke.

## Where iteration 2 fell short

  g5/bfloat16: candidate raised OutOfResources: out of resource: shared memory, Required: 279040, Hardware limit: 232448. Reducing block sizes or `num_stages` may help.

## What iteration 2 said about itself (UNVERIFIED)

  what it says stopped it : The g5 bfloat16 configuration still failed before producing a tensor because a launch required 279040 bytes of shared memory, exceeding the H100 limit of 232448 bytes; the retained bfloat16 BV=64, num_stages=2 path was not resource-safe for that shape.
  what it wanted to try next: Add a shape-based resource fallback for large BK in bfloat16—launch the resource-heavy tile with BV=32 and num_stages=1, or split its V tile into separate programs—so its shared-memory footprint stays below 232448 bytes.

  This is the attempt's own account, not a measurement. Treat the next idea as a lead worth checking, not as a conclusion -- and if you disagree with it, say why and do something else.

## What your implementation actually is (iteration 2, assessed, NOT part of your score)

  yes  diag_bwd_chunked
  yes  diag_chunked_matmul
  yes  diag_gate_cumsum
  yes  diag_wy_form

## What to do next

THE DESIGN MAY BE FINE. The previous attempt computed the wrong answer; that is a bug, not a bad design, so do not throw the approach away over it. You are starting from the shipped starter again, so re-implement that approach and get the specific error quoted above right this time. `./bench.py` checks all five gradients before it will print a speed number -- run it, and do not submit until it prints `correctness ok`.

## Rules that hold for every iteration

  - Your final edit to kernel.py must be followed by a bench.py run. Any number you report must come from a run on the EXACT bytes you shipped.
  - Do not report a speedup measured on a variant you did not submit. Your numbers are re-measured independently and a mismatch is a violation of the honest-measurement requirement, not a rounding difference.
  - Time the combined forward-and-backward pass with CUDA events or an explicit synchronise after discarded warmup runs. Forward-only timing is a violation.
