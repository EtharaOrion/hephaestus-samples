# Summary of what was done in previous iterations

## Approaches tried before, and their respective rewards

  Both numbers are measured. The approach text is the agent's own account of that attempt and is UNVERIFIED -- an attempt that failed may describe an optimisation that never actually landed.

  attempt   speed   score   what was tried
    1        --    0.0000   Replaced the implementation with a custom Triton fused-chunk gated-delta-rule using chunkwise gate cumsums, an exact WY triangular inverse, float32 recurrent-state carry, dense chunk output products, and a Triton backward scan.
                            config: No performance sweep was recorded. The visible static dispatch used NS=1 for inputs wider than 2 bytes and NS=2 otherwise; BV=min(64 if KP<=128 else 32, pow2(V)); KP and VP were next powers of two; the triangular inverse used BS=16 in the visible branch; float32 selected IEEE dot precision and a shallower pipeline. Functional runs covered bfloat16/float16/float32 at B=1,T=128,H=HV=2,K=V=64 and B=1,T=127,H=HV=3,K=V=96, plus bfloat16 at B=2,T=300,H=2,HV=4,K=64,V=128; those local cases passed, but the harness's g5 bfloat16 case did not.

  speed  = fraction of the production kernel's throughput; 1.0 would match it. `--` means the attempt was never timed because it computed the wrong answer.
  score  = what this attempt actually scored. It EQUALS the speed above when every gate and rule passed, and is 0.0000 if anything at all was wrong with the attempt -- so the two columns differ only on failures.
  config = the tuning values that attempt actually ran (tile sizes, warps, stages) and what its sweeps showed. Do not re-search values already listed; start from what they learned.
  Nothing has earned a score above 0.0000 yet, so there is no best attempt to beat. The speed column still shows how far each one got.

  EXPLORE A NEW IDEA, OR EXPLOIT ONE THAT ALREADY PERFORMS -- your call, and say which in your final message. Exploit means taking an approach above that already ran and making it faster, or fixing what stopped it. Explore means a structurally different idea, worth it only if the ones that ran have stopped improving. Repeating an approach that scored 0.0000, without changing what made it score 0.0000, is neither.

## Rules this session has already broken

  "Keep the recurrent state and the transition-inverse accumulator in"  (iteration 1)

  These cost a full zero each time, whatever the kernel measured. Check your submission against every line here before you finish, not only against the one the last attempt broke.

## Where iteration 1 fell short

  g5/bfloat16 scale=1.045: out[188 elements outside tolerance, max abs 4.559e-02]

## Why the rules rejected iteration 1

  ["Keep the recurrent state and the transition-inverse accumulator in"]
    In `_wy_kernel`, `b_Xc = b_X.to(b_k.dtype)` explicitly narrows the float32 transition inverse before computing `b_w` and `b_u`; similarly, `_state_kernel` uses `b_h.to(b_w.dtype)` in `tl.dot(b_w, b_h.to(b_w.dtype), ...)`, narrowing the recurrent state during the state transition, which violates the requirement to keep both in float32 inside the kernel.

  This is your own code quoted back at you. Fix the thing it names -- not the general topic of the rule, the specific line.

## What iteration 1 said about itself (UNVERIFIED)

  what it says stopped it : The shipped kernel still narrowed critical float32 intermediates—_wy_kernel cast the float32 inverse as b_X.to(b_k.dtype), and _state_kernel cast the recurrent state with b_h.to(b_w.dtype)—and consequently produced an inaccurate bfloat16 output for g5 at scale=1.045: 188 output elements exceeded tolerance with maximum absolute error 4.559e-02. This also violated the no-accumulator-narrowing contract, so speed was never evaluated.
  what it wanted to try next: Keep the WY inverse and recurrent state operands in float32 through their matrix products—remove b_X.to(b_k.dtype) and b_h.to(b_w.dtype), using float32/IEEE dot paths or an explicitly tiled float32 multiply-accumulate if Triton dot typing requires matching operands—to eliminate the g5 accumulation error and satisfy the accumulator rule.

  This is the attempt's own account, not a measurement. Treat the next idea as a lead worth checking, not as a conclusion -- and if you disagree with it, say why and do something else.

## What your implementation actually is (iteration 1, assessed, NOT part of your score)

  yes  diag_bwd_chunked
  yes  diag_chunked_matmul
  NO   diag_gate_cumsum
  yes  diag_wy_form

  The 'NO' answers are why you are slow. They do not lower your reward directly -- the clock already does that -- but each one names a structural property the fast implementations have and this one does not:
    [diag_gate_cumsum] `b_g = tl.load(g + p, mask=m_t, other=0.0).to(tl.float32)` followed by `tl.store(gc + p, tl.cumsum(b_g, axis=0), mask=m_t)` in `_cumsum_kernel` correctly forms a per-`BT`-chunk float32 cumulative sum, but exponentials such as `tl.exp(b_g[:, None] - b_g[None, :])`, `tl.exp(b_g)`, and `tl.exp(b_gl - b_g)` are applied without any clamp or other bound on their arguments, so the required bounded exponent argument is absent.

## What to do next

THE DESIGN MAY BE FINE. The previous attempt computed the wrong answer; that is a bug, not a bad design, so do not throw the approach away over it. You are starting from the shipped starter again, so re-implement that approach and get the specific error quoted above right this time. `./bench.py` checks all five gradients before it will print a speed number -- run it, and do not submit until it prints `correctness ok`.

## Rules that hold for every iteration

  - Your final edit to kernel.py must be followed by a bench.py run. Any number you report must come from a run on the EXACT bytes you shipped.
  - Do not report a speedup measured on a variant you did not submit. Your numbers are re-measured independently and a mismatch is a violation of the honest-measurement requirement, not a rounding difference.
  - Time the combined forward-and-backward pass with CUDA events or an explicit synchronise after discarded warmup runs. Forward-only timing is a violation.
