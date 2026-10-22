# Task: implement a fast routed fused mixture-of-experts kernel, forward and backward

`kernel.py` contains a working but deliberately slow implementation of a routed MoE layer — the routing decision lives INSIDE the operator. Your job is to make it as fast as you can without changing what it computes.

## What you may edit

`kernel.py`, and nothing else. `reference.py` is the correctness oracle and is read-only.

The entry point is `fused_moe(x, router_w, router_b, w1, w2, A)` with `x: [T, D]`, `router_w: [D, E]` (same dtype as `x`), `router_b: [E]` float32, `w1: [E, D, 2F]`, `w2: [E, F, D]`, and `A` the number of experts per token. It returns `y: [T, D]` in the input dtype. Its signature and semantics are fixed. The graded surface is forward AND backward: autograd must work through your implementation, and the gradients of `x`, `router_w`, `w1` and `w2` are graded alongside the output. `router_b` and `A` carry no gradient.

## The operator

Everything happens inside the one call — scores, selection, dispatch, expert compute, combine:

- **Scores.** `s = sigmoid(x @ router_w)`, computed in float32 from the input tensors.
- **Selection.** Each token takes the top `A` experts by the BIASED score `s + router_b`. Ties are broken by **lower expert index first**; `-0.0` and `+0.0` compare equal. `router_b` is a selection bias only: it decides which experts are picked and then vanishes — it never enters the combine weights, and it has no gradient (selection is piecewise-constant in it).
- **Combine weights.** `w_a = s[t, sel_a] / sum_a s[t, sel_a]` — the RAW sigmoid scores of the selected experts, normalized. Not the biased scores.
- **Experts.** `f(e, h) = (silu(h @ w1[e, :, :F]) * (h @ w1[e, :, F:])) @ w2[e]`, and `y[t] = sum_a w_a * f(sel_a, x[t])`.
- **Arithmetic.** Inputs are cast up to float32; every matmul, the sigmoid, the silu, the normalization and the combine accumulate in float32; only the final `y` is cast back to the input dtype.
- **Gradients.** Selection indices are treated as locally constant; gradients flow through the gathered raw scores of the selected experts (through the sigmoid into `router_w` and `x`) and through the expert computation. `reference.py`'s autograd graph is the definition of those gradients.

`reference.py` writes all of this out as a slow expert-by-expert loop and is the definition of correctness — where this text and that code ever disagree, the code wins.

## How you are graded

One float in `[0, 1]`. It is **zero** unless every gate passes:

- your submission does not call a grouped-GEMM library or an MoE serving stack. The banned names are listed under Constraints and are checked by a word-boundary scan of your comment-stripped source — string literals and docstrings count, `#` comments do not. Reading library source is allowed and encouraged; calling it is not.
- your output **and all four gradients** agree with `reference.py` under the disclosed per-dtype tolerances, on a hidden shape set that is larger than, and different from, the shapes `bench.py` lets you measure. The hidden set spans expert counts from 8-ish to 160-ish, `A` from 1 to `E`, non-power-of-two `T`, `D`, `F` and `E`, and inputs built to contain **exact** score ties in both dtypes.
- **at least 60% of graded forward+backward device time runs inside Triton kernels your own module declares.** This is measured by device-time attribution after the fact, not by reading your source. One narrow exception is carved out and disclosed: `torch.topk` / `torch.sort` / `torch.argsort` on the small `[T, E]` score matrix or the `[T*A]` routing metadata are permitted — routing bookkeeping is a rounding error of the runtime, and the floor keeps the expert compute, dispatch arithmetic and combine in kernels you wrote. Framework grouped or batched matmuls are not the permitted kind of call.

Above those gates, the score is the geometric mean across the hidden graded shapes — float32 and bfloat16 rows both — of `(production time) / (your time)` for one forward+backward invocation, normalized against a hidden target and saturating at 1. **The denominator is a production composition: the router matmul plus integer-key top-A in torch, feeding a `torch._grouped_mm` expert pipeline over tokens sorted by expert — a real CUTLASS grouped-GEMM path, timed on the same tensors your kernel is timed on.** Matching the slow starter scores near zero.

Three properties of the graded draws are disclosed because they shape correct solutions:

- **The decision margin.** On every graded draw, each token's `A`-th and `(A+1)`-th biased scores are either exactly equal (a manufactured tie, decided by the index rule) or separated by at least `1e-4`. Float32 score arithmetic sits orders of magnitude inside that margin; TF32 or bfloat16 score arithmetic does not — it flips selections and fails correctness outright. There is no reward for guessing scores cheaply.
- **The cotangent.** The backward is driven by a fixed pseudorandom ±uniform cotangent (a function of the output shape alone), like a training step — not an all-ones reduction.
- **Hidden randomization.** The input magnitude and the expert-popularity prior behind `router_b` are redrawn per invocation from hidden distributions, so expert load is imbalanced but never the SAME imbalance twice, and the tensors that are timed are the tensors that are checked. A branch that detects the checker's statistics and takes a cheaper path elsewhere gains nothing.

## The correctness harness, in five stages

Correctness is checked in five stages, and the score names the stage that failed. The cheap stages run first.

1. **Smoke.** One small nominal input, output and gradients. Catches a module that imports but does not run.
2. **Shape sweep.** The hidden shape set, across float32 and bfloat16. This is the stage the graded shapes are timed on.
3. **Stability.** Adversarial inputs: a router saturated to exact 0/1 sigmoids, a rank-1 router where nearly every expert is tied, inputs at 1e-30, six decades of magnitude in one tensor, and `router_b` at ±20. **This stage gates on stability, not agreement:** you must not raise, and you must not produce `NaN`/`Inf` where the reference is finite. Drift is recorded but does not gate here.
4. **Determinism.** The same tie-forcing input three times, after a warmup: outputs AND all four gradients must be **bitwise identical** run to run. A kernel that is not reproducible is not measurable.
5. **Edge cases.** Non-power-of-two everything, `A = 1`, `A = E`, and the exact-tie draws in both dtypes. Never timed; gate exactly as the sweep does.

## Requirements

These are engineering constraints on the kernel, not a separate rulebook, and several of them bind choices that look purely like performance decisions. A solution is considered correct only when it meets all of them; violating any one of them scores zero, whatever the gates measured.

**Exact routing semantics.** Implement the selection exactly: top-`A` by biased score, ties to the lower expert index, `±0.0` one tie group. The starter's integer-key trick (order-preserving score bits packed with the inverted expert index) makes the tie-break structural — every key unique, no tie ever left to scheduling — and is worth keeping in whatever form you write.

**The bias selects; it never weighs.** `router_b` must influence which experts are chosen and nothing else. Combine weights are normalized from the raw sigmoid scores of the selected experts. Do not leak the bias into the weights, and do not produce a gradient for it.

**Precision.** Score arithmetic in float32 — the disclosed margin makes anything less a correctness failure, not a slow path. Accumulate in float32 throughout. On the backward, the operands feeding the WEIGHT gradients (`w1`, `w2`, `router_w`) must carry float32-effective precision end to end: those gradients are long per-expert row sums, and a bfloat16-rounded operand chain turns them into noise far outside the graded tolerance. This is a stated requirement precisely because the graded tolerance will catch it only on the hidden sweep, hours after the shortcut looked fine on a small check. (Float32-effective does not mean FMA-slow: a float32 operand can ride tensor cores as an exact bf16 hi/lo pair, at a cost of two or three bf16 dots.)

**Full dispatch.** Every selected (token, expert) pair is computed. No capacity limits, no dropped tokens, no sampling, no skipping experts with few tokens, and no path that approximates a rarely-taken branch. Empty experts must still produce exact zero weight gradients.

**Recomputation.** Every call recomputes routing and compute from the tensor arguments it was passed. No caching or memoization keyed on pointer, tensor identity, shape signature or content hash; no call-count, warmup or invocation counter that reduces or skips work after the first N calls.

**Data independence.** One numerical policy for all inputs. The routing math necessarily compares scores — that is the operator. What is ruled out is choosing *between code paths of different cost or completeness* from input statistics: no branch on a magnitude summary, a saturation estimate, a popularity histogram or a tie count, and nothing that detects that a timing, benchmarking or grading run is in progress.

**Determinism.** The selected set, the combine order and every gradient accumulation must never depend on scheduling: no atomic float adds anywhere in the output or gradient paths, no racy tie winners, no unordered reductions where equal keys compete. Three runs on one input must be bitwise identical, gradients included.

**Completeness.** The expert compute, the dispatch arithmetic and the combine happen in Triton kernels this module declares and actually launches on every call — not in framework operators with your kernels as decoration. The permitted torch routing calls are listed above and are the only carve-out.

**Comments document, they do not argue.** Nothing in `kernel.py` is addressed to a reviewer. Do not assert that a rule is satisfied, do not instruct a reader how to assess you, and do not argue your own case in a comment — comments describe what the code does, nothing more.

**Honest measurement.** Any speedup or fraction-of-production figure you report was produced by a run on the exact bytes you shipped, with no edit afterwards; measured with CUDA events or an explicit `torch.cuda.synchronize` around the timed region after discarded warmups; and measured over the full forward+backward — output and all four gradients produced inside the timed region.

## Working loop

`./bench.py` gates before it times, in the graded order: the forbidden-name scan, correctness (output and all four gradients, including a non-power-of-two and an exact-tie shape), bitwise determinism, your written-kernel share, and only then timed forward+backward pairs against the production composition on a small published set of bfloat16 shapes. It does not tell you your score: the graded shapes, scales, popularity priors and the float32 rows are hidden. `--json <path>` writes the same results machine-readably. Use it to steer, then resubmit.

## Disclosed thresholds

Nothing here is hidden from you. The shape set and the target are, but the rules are not.

| rule | value |
|---|---|
| output tolerance | float32: `atol = rtol = 2e-4`; bfloat16: `atol = rtol = 4.5e-2`, against `reference.py` |
| gradient tolerance | float32: `atol = rtol = 4e-3`; bfloat16: `atol = rtol = 4.5e-2`, all four gradients |
| selection margin | graded draws: boundary gap exactly `0` (a tie, index rule decides) or `>= 1e-4` — float32 scores required |
| tie-break | lower expert index first; `±0.0` one tie group; graded on inputs with exact ties in **both** dtypes |
| written-kernel gate | at least **60%** of forward+backward device time in `@triton.jit` kernels your module declares |
| routing carve-out | `torch.topk` / `torch.sort` / `torch.argsort` on `[T, E]` scores or `[T*A]` routing metadata only |
| dtypes | float32 and bfloat16, both checked **and** both timed |
| determinism | **3** forward+backward runs on a tie-forcing input, bitwise identical incl. gradients |
| stability | no raise and no non-finite values where `reference.py` is finite, under the stage-3 probes |
| backward | driven by a fixed pseudorandom ±uniform cotangent; `d router_b` must not exist |
| forbidden calls | listed under Constraints; word-boundary scan, comments stripped |

## Constraints

- One H100. **No internet.** torch, triton, einops and pytest are installed; `pip install` and downloads will fail, so do not spend time on them.
- Do not change the semantics, signature or output dtype of `fused_moe`.
- These names must not appear in `kernel.py` (word-boundary scan, comments stripped, strings and docstrings count — including as your own identifiers): `torch._grouped_mm`, `_grouped_mm`, `vllm`, `sglang`, `grouped_gemm`, `transformer_engine`, `flashinfer`. Building the grouped expert compute is the task; calling a library's is not a solution to it.
- Numerical policy must be uniform: no branch on input magnitude, distribution, saturation, popularity or seed that changes which algorithm runs.
- The gain must come from your own kernel work.


---

# Summary of what was done in previous iterations

## Approaches tried before, and their respective scores

  Both numbers are measured. The approach text is the agent's own account of that attempt and is UNVERIFIED -- an attempt that failed may describe an optimisation that never actually landed.

  attempt   speed   score   what was tried
    1      0.2526  0.2526   Rewrote the starter entirely as a hand-written Triton MoE: fp32 router GEMM + sigmoid, exact integer-key top-A selection with lower-index tie-break, argsort-based sorted-segment dispatch with a per-expert tile table, two grouped expert GEMMs, and nine backward kernels, then swapped the dot precision from IEEE to tf32x3 and enlarged the dispatch tile.
                            config: Grid/tile decomposition actually shipped: router (cdiv(T,32), cdiv(E,32)); weights (T,); dispatch (cdiv(T*A,256),); prefix (1,) with num_warps=1; expert_in (max_tiles, cdiv(F,64)); expert_out (max_tiles, cdiv(D,64)); combine (T, cdiv(D,256)); grouped-GEMM row tile BM=64 with max_tiles = cdiv(T*A,64)+E; backward dw kernels one expert per program looping for k in range(begin, end, BK). Two sweeps were run through ./bench.py: (a) input_precision "ieee" everywhere (/tmp/moe.json) vs "tf32x3" on all non-router dots (/tmp/moe-tf32x3.json) -- tf32x3 was kept and is what shipped; (b) _prefix_kernel BM=32 vs 64 (/tmp/moe-64.json). Neither bench's printed numbers appear in the transcript, so the magnitude of each delta is unknown; no num_warps or num_stages sweep and no @triton.autotune were attempted, and block sizes 32/64/256 were fixed by hand rather than searched.
    2 *    0.4080  0.4080   Rewrote kernel.py from the starter into a fully custom Triton sorted-expert-segment MoE pipeline (router GEMM, integer-key top-A + argsort dispatch, tile table, grouped expert GEMMs, combine, and seven hand-written backward kernels), with a compile-time BF flag selecting native bfloat16 tensor-core dots (plus bf16 high/low decomposition for weight grads) versus tf32x3 for float32.
                            config: No sweep was run -- a single configuration was written and benchmarked twice (once before and once after two small edits to _metadata/_inverse and _Moe.backward), so the tile-size/warp space is untouched. Values baked into the shipped code, for reference: expert-GEMM row tile BM=64 (tile-table capacity mt=cdiv(N,64)+E, tiles padded with expert id -1), _dw2 grid (E, cdiv(F,32), cdiv(D,32)) and _dw1 grid (E, cdiv(D,32), cdiv(2F,32)) with BK=32 segment loops, top-k tie field packed in the low 10 bits (E<=1024), fp32 dots via input_precision="tf32x3" and the router weight grad via "ieee". num_warps/num_stages were not shown being varied.
    3        --    0.0000   The agent replaced the starter with a full hand-written Triton sorted-dispatch fused-MoE pipeline (router GEMM, tile/offset build, per-expert dispatch + expert FFN, combine, and backward including dX, dW1, dW2), then hill-climbed tile sizes against bench.py.
                            config: Dispatch tile size mt = cdiv(N, BM) + E swept BM = 64 -> 128 -> 256 (256 config saved as /tmp/kernel_dispatch128.py and later restored as the working base, implying 256 was not kept as a win); weight-gradient kernels _dw1/_dw2 swept (BM=64, BN=64) -> (BM=128, BN=64) -> (BM=128, BN=128) with the grid changed to (E, cdiv(F,BM), cdiv(D,BN)); reduction depth BK swept 32 vs 64 for _f2/_dw (snapshots kernel_bk32.py / kernel_bk64.py); final untested edit made the dispatch BM data-dependent (64 when N < 128*E, else 128). Numeric-path variants tried: bf16 high/low split dot vs plain bf16 dot, and tf32x3 input_precision on the fp32 path. The printed timings from timed_pair (warmup=2, trials=5, budget_s=40 over all bench SHAPES) and from the torch profiler runs are truncated in the transcript, so no per-config speed numbers are recoverable; num_warps/num_stages were never swept.
    4      0.2267  0.2267   Rewrote the starter MoE entirely as a hand-written Triton pipeline: sigmoid router GEMM, integer-key top-A selection, argsort-based sorted-segment dispatch with a binary-search lower-bound + tile-table (grouped-GEMM style), fused gather in the A-operand of `_expert_in`/`_expert_out`, plus a full hand-written backward (`_route_grad`, `_dact`, `_dgu`, `_dx`, `_dw1`, `_dw2`, `_drw`) with no atomics.
                            config: Tile sizes were fixed constants, never swept: BM=64 for the dispatch tile table (`_metadata(sel, A, E, 64)`), 64-wide N-tiles for `_expert_in` (grid `(mt, cdiv(2F,64))`) and `_expert_out` (`(mt, cdiv(D,64))`), 64x64 tiles with a `for k in range(begin,end,BK)` loop for `_dw1` (grid `(E, cdiv(D,64), cdiv(2F,64))`) and `_dw2` (`(E, cdiv(F,64), cdiv(D,64))`), 256-element blocks for `_activation` (`cdiv(N*F,256)`), `_inverse` (`cdiv(n,256)`) and `_combine` (`(T, cdiv(D,256))`), and 32x32 for `_router` (`(cdiv(T,32), cdiv(E,32))`). No num_warps / num_stages were set or varied anywhere. The only real sweep was on tl.dot precision, benched three times end-to-end: (a) `tl.dot(a, w.to(tl.float32), input_precision="tf32x3")` in `_expert_out`/`_dact` -- agent's bench printed ~0.18 fraction; (b) switched to native `tl.dot(a.to(tl.bfloat16), w)` and re-benched; (c) reverted to the fp32/tf32x3 form and benched again, which is what shipped. Final self-bench row quoted by the agent: shape p1 (T=4096,D=1024,F=704,E=32,A=4) yours 9.2255 ms vs production 2.3569 ms.

  speed  = fraction of the production kernel's throughput; 1.0 would match it. `--` means the attempt was never timed because it computed the wrong answer.
  score  = what this attempt actually scored. It EQUALS the speed above when every gate and rule passed, and is 0.0000 if anything at all was wrong with the attempt -- so the two columns differ only on failures.
  config = the tuning values that attempt actually ran (tile sizes, warps, stages) and what its sweeps showed. Do not re-search values already listed; start from what they learned.
  *      = best score so far, and the one to beat.

  EXPLORE A NEW IDEA, OR EXPLOIT ONE THAT ALREADY PERFORMS -- your call, and say which in your final message. Exploit means taking an approach above that already ran and making it faster, or fixing what stopped it. Explore means a structurally different idea, worth it only if the ones that ran have stopped improving. Repeating an approach that scored 0.0000, without changing what made it score 0.0000, is neither.

## What iteration 4 said about itself (UNVERIFIED)

  what it says stopped it : It ran, was correct and deterministic, but stayed slow, and the evidence points at the matmul precision paths rather than the dispatch machinery: the agent's own bench at step [21] showed only ~0.18 fraction, and its step-[22] diagnosis was that `_expert_out` and `_dact` promote one bf16 operand to fp32 (`w.to(tl.float32)` with `input_precision="tf32x3"`), which blocks native bf16 tensor-core `tl.dot`. It then flip-flopped: switched those dots to pure `tl.dot(a.to(tl.bfloat16), w)` [23], re-benched [26], and reverted back to the fp32/tf32x3 form [27] before the final bench [28] -- i.e. the fast bf16 path was tried and abandoned (presumably on accuracy) rather than made accurate cheaply. The submitted kernel therefore runs bf16 shapes through split-precision/tf32x3 dots, and bf16 shapes are exactly where it is weakest (g1/bf16 worst). Secondary cost: routing still leans on torch.topk + torch.argsort per call, and `_dense_dl` materialises a dense (T,E) tensor for the router backward.
  what it wanted to try next: Keep the sorted-segment dispatch but make the bf16 GEMMs use native bf16 `tl.dot` on both operands in `_expert_in`/`_expert_out`/`_dact`/`_dgu`/`_dx` with fp32 accumulators, and if that lost accuracy, restrict the high/low bf16 operand split to only the weight-gradient kernels (`_dw1`/`_dw2`/`_drw`, where the tolerance actually bites) instead of paying tf32x3/fp32-promotion on the dominant forward and dx GEMMs; separately, replace the dense (T,E) `_dense_dl` staging in the router backward with a direct scatter over the A selected columns.

  This is the attempt's own account, not a measurement. Treat the next idea as a lead worth checking, not as a conclusion -- and if you disagree with it, say why and do something else.

## Where the time is going (best iteration so far, 6 timed cases)

  Your best case reaches 0.7270 of the production kernel. Your worst reaches 0.2008. That is a 3.6x spread across cases that differ only in shape.

  worst case: yours 37.82 ms against production 7.60 ms -- 5.0x slower
  best case : yours 64.47 ms against production 46.89 ms -- 1.4x slower
  the worst case is 0.6x the wall-clock of your best case
  A spread that wide is not a tuning problem: some cases are hitting a different bottleneck than the rest.
  Measured on your own timings: you do BETTER on the longest case (0.6519 at 16384 tokens) than the shortest (0.2739 at 4096), so length is not what is hurting you -- look at the small cases and at per-launch overhead.
  Fixing the worst cases is worth far more than shaving the best ones.

## What your implementation actually is (iteration 4, assessed, NOT part of your score)

  yes  diag_dispatch_strategy
  NO   diag_measured_before_submit
  NO   diag_router_backward
  NO   diag_tie_handling
  NO   diag_weight_grad_precision

  The 'NO' answers are why you are slow. They do not lower your reward directly -- the clock already does that -- but each one names a structural property the fast implementations have and this one does not:
    [diag_tie_handling] The mechanism is present: in `_select` the biased score `v = s + b` (with `-0.0` normalized to `0.0`) is bit-cast to int32, order-flipped into a monotone unsigned key `u`, then packed as `(u << 10) | (E - 1 - e)` before `torch.topk(..., largest=True, sorted=True)` — a composite integer key whose low bits `E-1-e` make a lower expert index win exact ties, so no separate stable pass or post-hoc repair is needed. However, the required validation is missing: the trajectory's only correctness runs (steps [15], [19], [20]) build inputs with `torch.randn`/random `vals` at shapes like T,D,F,E,A=8,16,12,5,2 and 17,33,19,7,3 and compare against `fused_moe_ref`, with no step constructing an exact-tie routing case (e.g., identical/duplicated router-bias-adjusted scores across experts) to exercise the `E-1-e` tie-break path before submission. [agreed by an independent check, an independent check, an independent check]
    [diag_weight_grad_precision] The weight-gradient chain does use a mixed strategy: `_dw2` computes `ah = aa.to(tl.bfloat16); al = (aa - ah.to(tl.float32)).to(tl.bfloat16)` and accumulates `tl.dot(tl.trans(ah), zh) + tl.dot(tl.trans(al), zh) + tl.dot(tl.trans(ah), zl)` when BF is set, while `_dw1` and `_drw` split only the gradient operand (`zh`/`zl`) and all three fall back to `input_precision="tf32x3"` (or `"ieee"` in `_drw`) for fp32 inputs. However, nothing in the submission or trajectory ties this choice to a disclosed gradient tolerance -- there are no comments naming a rtol/atol budget for dw1/dw2/drw, and the only visible reasoning about precision is reactive: after the first correctness run at step [15], steps [16]-[18] patch `input_precision=` arguments in existing `tl.dot` calls and re-test, i.e. the precision scheme was tuned by iterating on comparisons against `reference.fused_moe_ref` (steps [19]-[20]) rather than derived up front from a stated tolerance. [agreed by an independent check, an independent check, an independent check]
    [diag_router_backward] The kernel code in _route_grad/_dense_dl/_drw does implement backprop through the normalization (dl = (dot-center)/den) and the sigmoid derivative (ss*(1-ss)) restricted to the SEL-gathered entries, matching a gather graph. However, the trajectory only shows truncated commands like step 19: "cd /workspace && python3 - <<'PY'\nimport torch\nfrom kernel import fused_moe\nfrom reference import fused_moe_ref\nfor dt in (torch.bfloat16, torch.float32): ..." with no visible output or explicit reference to checking router_w.grad / drw against the reference implementation's autograd result. There is no quotable evidence (printed max-diff, assert, or gradcheck call) confirming that drouter_w specifically was numerically validated against the reference; the validation method for drw is not established by the given trajectory. [PANEL SPLIT, resolved 2-2: an independent check=pass, an independent check=fail, an independent check=fail, an independent check=pass]
    [diag_measured_before_submit] The trajectory's only interactions with bench.py are listing it (step [4], `find . -maxdepth 2 -type f` showing `./bench.py`) and reading it (step [11], `sed -n '25,140p' bench.py`); it is never executed. The final recorded actions (steps [15], [18], [19], [20]) are `python3 -m py_compile kernel.py` plus ad-hoc heredoc scripts that compare `fused_moe` against `fused_moe_ref` on tiny shapes (T,D,F,E,A=8,16,12,5,2 and 17,33,19,7,3) for correctness only — no gated timing run on the submitted kernel.py bytes, and no benchmark numbers are produced or reported anywhere in the trajectory. [agreed by an independent check, an independent check, an independent check]

## What to do next

THE APPROACH ABOVE WORKED: correct, your own Triton, and it reached 0.4080 of the production kernel's throughput. You are starting from the shipped starter again, so re-implement that approach rather than inventing a new one, then make it faster. Its description and its speed are in the table above.

## Rules that hold for every iteration

  - Your final edit to kernel.py must be followed by a bench.py run. Any number you report must come from a run on the EXACT bytes you shipped.
  - Do not report a speedup measured on a variant you did not submit. Your numbers are re-measured independently and a mismatch is a violation of the honest-measurement requirement, not a rounding difference.
  - Time the combined forward-and-backward pass with CUDA events or an explicit synchronise after discarded warmup runs. Forward-only timing is a violation.
