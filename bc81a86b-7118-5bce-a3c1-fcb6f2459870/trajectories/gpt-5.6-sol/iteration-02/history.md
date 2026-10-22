# Summary of what was done in previous iterations

## Approaches tried before, and their respective rewards

  Both numbers are measured. The approach text is the agent's own account of that attempt and is UNVERIFIED -- an attempt that failed may describe an optimisation that never actually landed.

  attempt   speed   score   what was tried
    1 *    0.0522  0.0522   Replaced the plain-PyTorch starter with a hand-written Triton implementation of the chunked gated delta rule: a single fused forward kernel (`_chunk_fwd`) that walks every chunk of the sequence sequentially inside one program while carrying the float32 state and checkpointing per-chunk states, a mirrored reverse-scan backward kernel (`_chunk_bwd`), plus two small reduction kernels (`_reduce_qk`, `_reduce_s`) to fold the per-V-block partial gradients.
                            config: Final shipped configuration: CHUNK(BT)=16, VALUE_BLOCK(BV)=32, BK=next_power_of_2(K) (whole K per program), grid `(B*HV*NV,)`, `input_precision='ieee'` on all dots. The agent edited the CHUNK/VALUE_BLOCK constant block three times (steps 35, 37, 39) between its own timing loops on bench SHAPES['p1'] with bench.step and `statistics`, and then swept num_warps 4 -> 8 on the BT=CHUNK launches via sed; the printed timings from those runs are truncated in the log, so the outcome of each constant/num_warps value is not recoverable — a later attempt should re-measure rather than trust these. An earlier scratch draft used CHUNK=32 but was never the version benchmarked. No num_stages was set anywhere.

  speed  = fraction of the production kernel's throughput; 1.0 would match it. `--` means the attempt was never timed because it computed the wrong answer.
  score  = what this attempt actually scored. It EQUALS the speed above when every gate and rule passed, and is 0.0000 if anything at all was wrong with the attempt -- so the two columns differ only on failures.
  config = the tuning values that attempt actually ran (tile sizes, warps, stages) and what its sweeps showed. Do not re-search values already listed; start from what they learned.
  *      = best score so far, and the one to beat.

  EXPLORE A NEW IDEA, OR EXPLOIT ONE THAT ALREADY PERFORMS -- your call, and say which in your final message. Exploit means taking an approach above that already ran and making it faster, or fixing what stopped it. Explore means a structurally different idea, worth it only if the ones that ran have stopped improving. Repeating an approach that scored 0.0000, without changing what made it score 0.0000, is neither.

## What iteration 1 said about itself (UNVERIFIED)

  what it says stopped it : It ran and was correct on every stage but stayed far slower than production because the design has no parallelism over time: the grid is only `(B*HV*NV,)` and each program serially scans all NC = T/16 chunks for both forward and backward, so at B=1 there are only a handful of CTAs on an H100 and the slowdown grows with T (worst at T=16384, B=1; best at T=3072, B=2 with more CTAs); on top of that the intra-chunk triangular solve is a BT=16 row-by-row `[expr]`/`[expr]` forward-substitution loop (and its transposed back-substitution in the backward) that runs on CUDA cores instead of tensor cores, and BK is forced to next_power_of_2(K) so the whole K dimension stays resident in registers, capping the usable chunk size.
  what it wanted to try next: Break the single whole-sequence scan into the standard three-launch structure so time becomes parallel: (1) a kernel parallel over (B, HV, chunk) that computes the gate cumsum, the WY/UT transform and each chunk's local state contribution, (2) a tiny sequential kernel that scans only the NC chunk-level states (and its reverse for dst), (3) a kernel parallel over (B, HV, chunk, V-block) that emits outputs/gradients; and inside that, replace the row-by-row substitution with a blocked triangular inverse expressed as `[expr]` on strictly-lower sub-blocks so the UT transform uses tensor cores, which should also make BT=64 affordable.

  This is the attempt's own account, not a measurement. Treat the next idea as a lead worth checking, not as a conclusion -- and if you disagree with it, say why and do something else.

## Where the time is going (best iteration so far, 8 timed cases)

  Your best case reaches 0.2205 of the production kernel. Your worst reaches 0.0138. That is a 16.0x spread across cases that differ only in shape.

  worst case: yours 201.74 ms against production 2.78 ms -- 72.5x slower
  best case : yours 8.56 ms against production 1.90 ms -- 4.5x slower
  the worst case is 23.6x the wall-clock of your best case
  A spread that wide is not a tuning problem: some cases are hitting a different bottleneck than the rest.
  Measured on your own timings: at 256 tokens you reach 0.0274 of production, and at 16384 tokens you reach 0.0138 -- 2.0x worse over a 64x longer sequence. Your cost grows faster with sequence length than the production kernel's does.
  Fixing the worst cases is worth far more than shaving the best ones.

## What your implementation actually is (iteration 1, assessed, NOT part of your score)

  yes  diag_bwd_chunked
  yes  diag_chunked_matmul
  NO   diag_gate_cumsum
  yes  diag_wy_form

  The 'NO' answers are why you are slow. They do not lower your reward directly -- the clock already does that -- but each one names a structural property the fast implementations have and this one does not:
    [diag_gate_cumsum] In _chunk_fwd the gate is computed as `[expr]` followed by `[expr]` (and similarly in _chunk_bwd: `[expr]`). While this does factor the gate as a float32 per-chunk cumulative sum of log-gates rather than applying decay per timestep, the exponent argument `cg`[expr]`cg` anywhere, and the code even computes `xx = kk / ee[:, None]`, dividing by an unbounded exp, so the required bounding of the exponent before exp is absent.
## What to do next

THE APPROACH ABOVE WORKED: correct, your own Triton, and it reached 0.0522 of the production kernel's throughput. You are starting from the shipped starter again, so re-implement that approach rather than inventing a new one, then make it faster. Its description and its speed are in the table above.

## Rules that hold for every iteration

  - Your final edit to kernel.py must be followed by a bench.py run. Any number you report must come from a run on the EXACT bytes you shipped.
  - Do not report a speedup measured on a variant you did not submit. Your numbers are re-measured independently and a mismatch is a violation of the honest-measurement requirement, not a rounding difference.
  - Time the combined forward-and-backward pass with CUDA events or an explicit synchronise after discarded warmup runs. Forward-only timing is a violation.
