# Summary of what was done in previous iterations

## Approaches tried before, and their respective scores

  Both numbers are measured. The approach text is the agent's own account of that attempt and is UNVERIFIED -- an attempt that failed may describe an optimisation that never actually landed.

  attempt   speed   score   what was tried
    1        --    0.0000   Replaced the starter with a custom autograd MoE pipeline using Triton router, expert GEMM, combine, and backward kernels, with expert-sorted dispatch and split high/low representations to improve float32 gradient accuracy.
                            config: The visible final configuration used expert-sorted GEMM tiles selected by `_cfg`, gemm1 with `nacc=2`, metadata construction with `BT=128`, and float32 operand splitting into high/low bf16 parts accumulated through three tensor-core dot products; no concrete BM/BN/BK, num_warps, or num_stages sweep and no post-edit timing numbers were recorded. Correctness was exercised at T=512,D=512,F=256,E=16,A=4 in bfloat16 and float32 and at non-power-of-two T=301,D=200,F=96,E=11,A=3, but the final graded float32 gradient failures remained.

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

## Where iteration 1 fell short

  g1/float32 scale=2.432: drouter_w[15 elements outside tolerance, max abs 7.959e-02]
  g3/float32 scale=1.573: drouter_w[1 elements outside tolerance, max abs 5.542e-02] dw1[32 elements outside tolerance, max abs 1.552e-02] dw2[40 elements outside tolerance, max abs 1.630e-02]
  c1/float32 scale=2.053: drouter_w[1 elements outside tolerance, max abs 2.881e-02]

## Why the rules rejected iteration 1

  [deterministic_routing_and_combine]
    In `_router_kernel`, `key = (u << 10) | (E - 1 - re[None, :]).to(tl.int64)`, followed by `sel = torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices`, uses only 10 reserved bits for expert tie-breaking without enforcing `E <= 1024`; for larger expert counts distinct experts can receive identical keys and therefore compete in `torch.topk`, whose tie ordering is not guaranteed, violating deterministic expert selection.

  [exact_selection_semantics]
    In `_router_kernel`, `key = (u << 10) | (E - 1 - re[None, :]).to(tl.int64)` reserves only 10 bits for the expert-index tie breaker, while `forward` imposes no `E <= 1024` check; for `E > 1024` the index field overlaps score bits and can even reverse equal-score ordering (for example, at `E=1025` when the shifted score already has bit 10 set), violating exact lower-index tie breaking on every input.

  [routing_primitive_scope]
    In `_FusedMoe.forward`, `bep = _pow2(E, 16)`, `keys = torch.empty((T, bep), ...)`, followed by `sel = torch.topk(keys, A, dim=-1, largest=True, sorted=True).indices`; this applies the framework selection primitive to a padded `[T, bep]` tensor rather than the permitted `[T, E]` score matrix (for non-power-of-two or `E < 16`, `bep != E`), violating the rule’s shape restriction.

  This is your own code quoted back at you. Fix the thing it names -- not the general topic of the rule, the specific line.

## What iteration 1 said about itself (UNVERIFIED)

  what it says stopped it : The final float32 backward remained numerically inaccurate: drouter_w had up to 15 out-of-tolerance elements with max absolute error 7.959e-02, and one configuration also had 32 bad dw1 elements and 40 bad dw2 elements with roughly 1.6e-02 maximum errors; the attempt stalled while revising split-precision gradient kernels, and the last scripted edit did not execute.
  what it wanted to try next: Add the omitted low×low term—or use an IEEE-fp32 accumulation path—in `_dot3` for drouter_w, dw1, and dw2, because the current three-term bf16 high/low decomposition leaves cancellation-sensitive float32 gradient residuals.

  This is the attempt's own account, not a measurement. Treat the next idea as a lead worth checking, not as a conclusion -- and if you disagree with it, say why and do something else.

## What your implementation actually is (iteration 1, assessed, NOT part of your score)

  yes  diag_dispatch_strategy
  NO   diag_measured_before_submit
  NO   diag_router_backward
  NO   diag_tie_handling
  yes  diag_weight_grad_precision

  The 'NO' answers are why you are slow. They do not lower your reward directly -- the clock already does that -- but each one names a structural property the fast implementations have and this one does not:
    [diag_tie_handling] In `_router_kernel`, `key = (u << 10) | (E - 1 - re[None, :]).to(tl.int64)` and `torch.topk(keys, A, dim=-1, largest=True, sorted=True)` implement a composite integer key whose low bits favor smaller expert indices on equal biased scores; however, the trajectory only shows generic checks such as step [36] (`check {'T': 512, 'D': 512, 'F': 256, 'E': 16, 'A': 4}` and `npot ...`) and contains no specific exact-tie-shape test before submission, so the rule's testing requirement is not satisfied.
    [diag_router_backward] `_dweights_kernel` computes `dl = ... (dwa - inner[:, None]) / ssum[:, None] * z * (1.0 - z)` from `sel` and scatters only where `re == sel_a`, after which `_drw_kernel` forms `drouter_w = x^T @ dlogit`; this correctly follows normalization and sigmoid only at gathered experts, but the trajectory shows only generic commands such as `python /tmp/diag.py` and truncated `check ...` labels, with no explicit numerical comparison, tolerance, finite-difference/gradcheck, or reported error for `drw` versus the reference `drouter_w`, so the required validation is not evidenced.
    [diag_measured_before_submit] Trajectory step [14] runs `cd /workspace && python /tmp/prodbench.py`, but the agent subsequently replaces `/workspace/kernel.py` at step [19] and edits it repeatedly through steps [23], [29]-[34], and [41]-[44]; therefore that measurement was not run on the final kernel bytes, and no later `bench.py` gated run with reported numbers is shown.

## What to do next

THE DESIGN MAY BE FINE. The previous attempt computed the wrong answer; that is a bug, not a bad design, so do not throw the approach away over it. You are starting from the shipped starter again, so re-implement that approach and get the specific error quoted above right this time. `./bench.py` checks all five gradients before it will print a speed number -- run it, and do not submit until it prints `correctness ok`.

## Rules that hold for every iteration

  - Your final edit to kernel.py must be followed by a bench.py run. Any number you report must come from a run on the EXACT bytes you shipped.
  - Do not report a speedup measured on a variant you did not submit. Your numbers are re-measured independently and a mismatch is a violation of the honest-measurement requirement, not a rounding difference.
  - Time the combined forward-and-backward pass with CUDA events or an explicit synchronise after discarded warmup runs. Forward-only timing is a violation.
