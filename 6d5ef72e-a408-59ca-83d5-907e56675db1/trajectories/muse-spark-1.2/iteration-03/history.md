# Summary of what was done in previous iterations

## Approaches tried before, and their respective scores

  Both numbers are measured. The approach text is the agent's own account of that attempt and is UNVERIFIED -- an attempt that failed may describe an optimisation that never actually landed.

  attempt   speed   score   what was tried
    1        --    0.0000   (no narrator account; from its tool calls: edited kernel.py, ran bench.py 3x, 40 tool calls)
    2        --    0.0000   (no narrator account; from its tool calls: edited kernel.py, ran bench.py 5x, 32 tool calls)

  speed  = fraction of the production kernel's throughput; 1.0 would match it. `--` means the attempt was never timed because it computed the wrong answer.
  score  = what this attempt actually scored. It EQUALS the speed above when every gate and rule passed, and is 0.0000 if anything at all was wrong with the attempt -- so the two columns differ only on failures.
  Nothing has earned a score above 0.0000 yet, so there is no best attempt to beat. The speed column still shows how far each one got.

  EXPLORE A NEW IDEA, OR EXPLOIT ONE THAT ALREADY PERFORMS -- your call, and say which in your final message. Exploit means taking an approach above that already ran and making it faster, or fixing what stopped it. Explore means a structurally different idea, worth it only if the ones that ran have stopped improving. Repeating an approach that scored 0.0000, without changing what made it score 0.0000, is neither.

## Where iteration 2 fell short

  g1/float32 scale=2.432: drouter_w[26 elements outside tolerance, max abs 1.355e-01]
  g2/float32 scale=1.451: drouter_w[1 elements outside tolerance, max abs 1.483e-02]
  g3/float32 scale=1.573: drouter_w[8 elements outside tolerance, max abs 1.914e-01] dw1[34 elements outside tolerance, max abs 1.559e-02] dw2[46 elements outside tolerance, max abs 1.701e-02]
  c1/float32 scale=2.053: drouter_w[1 elements outside tolerance, max abs 4.260e-02]

## What to do next

THE DESIGN MAY BE FINE. The previous attempt computed the wrong answer; that is a bug, not a bad design, so do not throw the approach away over it. You are starting from the shipped starter again, so re-implement that approach and get the specific error quoted above right this time. `./bench.py` checks all five gradients before it will print a speed number -- run it, and do not submit until it prints `correctness ok`.

## Rules that hold for every iteration

  - Your final edit to kernel.py must be followed by a bench.py run. Any number you report must come from a run on the EXACT bytes you shipped.
  - Do not report a speedup measured on a variant you did not submit. Your numbers are re-measured independently and a mismatch is a violation of the honest-measurement requirement, not a rounding difference.
  - Time the combined forward-and-backward pass with CUDA events or an explicit synchronise after discarded warmup runs. Forward-only timing is a violation.
