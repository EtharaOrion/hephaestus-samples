# Summary of what was done in previous iterations

## Approaches tried before, and their respective scores

  Both numbers are measured. The approach text is the agent's own account of that attempt and is UNVERIFIED -- an attempt that failed may describe an optimisation that never actually landed.

  attempt   speed   score   what was tried
    1      0.3290  0.3290   (no narrator account; from its tool calls: edited kernel.py, ran bench.py 5x, 40 tool calls)
    2      0.2352  0.2352   (no narrator account; from its tool calls: edited kernel.py, ran bench.py 5x, 40 tool calls)
    3      0.2424  0.2424   (no narrator account; from its tool calls: edited kernel.py, ran bench.py 5x, 40 tool calls)
    4        --    0.0000   (no narrator account; from its tool calls: edited kernel.py, ran bench.py 6x, 1 tool errors, 40 tool calls)
    5        --    0.0000   (no narrator account; from its tool calls: edited kernel.py, ran bench.py 6x, 40 tool calls)
    6        --    0.0000   (no narrator account; from its tool calls: edited kernel.py, ran bench.py 1x, 40 tool calls)
    7        --    0.0000   (no narrator account; from its tool calls: edited kernel.py, ran bench.py 5x, 40 tool calls)
    8 *    0.3348  0.3348   (no narrator account; from its tool calls: edited kernel.py, ran bench.py 3x, 40 tool calls)
    9        --    0.0000   (no narrator account; from its tool calls: edited kernel.py, ran bench.py 4x, 40 tool calls)

  speed  = fraction of the production kernel's throughput; 1.0 would match it. `--` means the attempt was never timed because it computed the wrong answer.
  score  = what this attempt actually scored. It EQUALS the speed above when every gate and rule passed, and is 0.0000 if anything at all was wrong with the attempt -- so the two columns differ only on failures.
  *      = best score so far, and the one to beat.

  EXPLORE A NEW IDEA, OR EXPLOIT ONE THAT ALREADY PERFORMS -- your call, and say which in your final message. Exploit means taking an approach above that already ran and making it faster, or fixing what stopped it. Explore means a structurally different idea, worth it only if the ones that ran have stopped improving. Repeating an approach that scored 0.0000, without changing what made it score 0.0000, is neither.

## Where iteration 9 fell short

  g1/float32 scale=2.432: drouter_w[36 elements outside tolerance, max abs 1.685e-01]
  g3/float32 scale=1.573: drouter_w[15 elements outside tolerance, max abs 2.332e-01] dw1[179 elements outside tolerance, max abs 1.695e-02] dw2[206 elements outside tolerance, max abs 1.868e-02]
  c1/float32 scale=2.053: drouter_w[1 elements outside tolerance, max abs 5.273e-02]

  This same failure has now happened 4 times (attempts 4, 5, 7, 9). Whatever you have been changing between attempts is not what is causing it.

## Where the time is going (best iteration so far, 6 timed cases)

  Your best case reaches 0.7953 of the production kernel. Your worst reaches 0.1318. That is a 6.0x spread across cases that differ only in shape.

  worst case: yours 43.39 ms against production 5.72 ms -- 7.6x slower
  best case : yours 27.86 ms against production 22.15 ms -- 1.3x slower
  the worst case is 1.6x the wall-clock of your best case
  A spread that wide is not a tuning problem: some cases are hitting a different bottleneck than the rest.
  Measured on your own timings: you do BETTER on the longest case (0.7953 at 16384 tokens) than the shortest (0.1318 at 4096), so length is not what is hurting you -- look at the small cases and at per-launch overhead.
  Fixing the worst cases is worth far more than shaving the best ones.

## What to do next

THE SAME WRONG-ANSWER BUG HAS NOW KILLED 4 ATTEMPTS (iterations 4, 5, 7, 9):
    g1/float32 scale=2.432: drouter_w[5 elements outside tolerance, max abs 5.493e-02]

It has recurred across implementations that share almost no code with each other, so it is not a property of any one design and rewriting from scratch will not remove it. You are shipping a kernel whose gradients you have not checked.

`./bench.py` compares your output AND all five gradients against `reference.py` before it times anything, and refuses to print a speed number when they disagree. Run it. Do not submit until it prints `correctness ok`.

A forward pass can be right while the backward is garbage: that is this exact failure, every time. If bench.py reports correctness ok and the graded run still fails you, say so explicitly in your final message -- the difference is then the shape, not the algebra, and that is worth knowing.

## Rules that hold for every iteration

  - Your final edit to kernel.py must be followed by a bench.py run. Any number you report must come from a run on the EXACT bytes you shipped.
  - Do not report a speedup measured on a variant you did not submit. Your numbers are re-measured independently and a mismatch is a violation of the honest-measurement requirement, not a rounding difference.
  - Time the combined forward-and-backward pass with CUDA events or an explicit synchronise after discarded warmup runs. Forward-only timing is a violation.
