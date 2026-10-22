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
    5 R    0.6603  0.0000   Rewrote the starter MoE entirely into a Triton pipeline — router GEMM, sorted-segment dispatch with a tile table, fused gather into the w1 grouped GEMM, SwiGLU, w2 grouped GEMM, weighted combine, plus a full Triton backward (_dz, _dactivate, _dx_rows, _combine_dx, _dw1, _dw2, _pair_dot/_route_grad/_drw) using bf16 hi/lo split dots for precision-sensitive weight gradients.
                            config: Tile/grid structure: expert GEMMs launched as grid (mt, cdiv(2F,64)) for _expert_in and (mt, cdiv(D,64)) for _expert_out, _combine as (T, cdiv(D,256)), tile table sized mt = cdiv(N,BM)+E with BM=64. A scripted experiment (step 31/33) rewrote _dx_rows from cdiv(D,64) to a 128-wide variant and altered _expert_out/_dz tiling, was timed with bench.timed_pair on shapes p1/p2 (warmup=2), and was reverted from the /tmp/kernel-dx128.py snapshot — the wider tiles were not kept. Precision sweep on the router gradient dot in _drw: default -> intermediate variant (/tmp/moe-precision.json) -> input_precision="ieee" (/tmp/moe-ieee-router.json), settling on ieee after comparing g['router_w'] against the reference. _dw2 uses three-term bf16 split (dot(ah,bh)+dot(al,bh)+dot(ah,bl)); _dw1 uses two-term (dot(x,gh)+dot(x,gl)). No num_warps/num_stages sweep is visible anywhere in the trajectory.
    6      0.3036  0.3036   Kept the starter's per-expert masked-loop dispatch and custom-autograd structure intact, and only sped up the shared Triton matmul: larger tiles (32x32x32 -> 64x64x32) plus a per-call-site precision switch that uses tf32x3 `tl.dot` for the expert GEMMs while keeping ieee fp32 for the router projection.
                            config: Only one configuration was actually edited and measured: matmul tiles BM,BN,BK swept from the starter 32,32,32 to 64,64,32 (no other tile values tried, no num_warps or num_stages values set or swept, grid left as (cdiv(M,BM), cdiv(N,BN)) 2-D), plus a binary precision choice on `tl.dot` -- ieee for the router matmul, tf32x3 for both expert projections and their dA/dB backward launches. The agent ran ./bench.py before the edit and once after (with a sha256 pin around the final run) and ran the CHECK_SHAPE / CHECK_SHAPE_NPOT / CHECK_SHAPE_TIES correctness shapes, but no per-variant timing comparison was recorded, so the individual worth of the tile change versus the tf32x3 change is unknown.
    7 R    0.3036  0.0000   The agent kept the starter's per-expert masked-loop MoE structure entirely and made a single micro-edit: it enlarged the Triton matmul tile from 16x16x16 to 64x64x32 and added launch tuning (num_warps=4, num_stages=3), then measured once and submitted.
                            config: Exactly one configuration was run end-to-end: `BM, BN, BK = 64, 64, 32` with `num_warps=4, num_stages=3`, tf32x3 for the expert GEMMs and ieee for the router, benchmarked once via `./bench.py --json /tmp/final.json` (final bytes sha256 5bb0576f...). The prior value `BM, BN, BK = 16, 16, 16` (starter default, presumably default warps/stages) was never benchmarked in this session, so the 16 vs 64 comparison has no measured delta here. No sweep over BM/BN/BK, num_warps, num_stages, grid decomposition or precision flags was performed.
    8        --    0.0000   Rewrote kernel.py as a full Triton MoE (sorted-expert dispatch with a tile table, fused gather GEMMs, and hand-written forward/backward kernels), then focused optimisation on the dominant weight-gradient kernels using bf16 hi/lo split-accumulation dots instead of plain bf16 tensor-core dots.
                            config: Fixed tile constants only: ROW_BLOCK = 64 and K_BLOCK = 32 as module-level constants, never swept. Grid decompositions used: _expert_in[(cap, cdiv(2*F,64))], _expert_out[(cap, cdiv(D,64))], _combine[(T, cdiv(D,256))], _dw2[(E, cdiv(F,64), cdiv(D,64))], _offsets[(E+1,)], _metadata[(1,)] (single-program tile-table build); cap = cdiv(T*A, ROW_BLOCK) + E. The only decomposition actually changed during the run was _combine_dx, moved to grid (T, cdiv(D,256)) as the last edit after a torch.profiler run on shape p2. Precision variants tried: fp32 path tl.dot(input_precision="tf32x3"); bf16 path three-term hi/lo split in _dw2, two-term split in _dw1, split ds with 0.0625 scaling in _drw -- these were introduced reactively after bench runs, not compared against a plain-bf16 baseline. No num_warps or num_stages values were specified or swept anywhere. Timing was collected via bench.timed_pair(warmup=2, trials=5, budget_s=30) over all bench.SHAPES, but the printed ratios are not visible in the captured output; the final gated bench.py run was executed on the shipped bytes (sha256 26da995a...) at step [26].
    9      0.3029  0.3029   Kept the starter's per-expert masked-loop MoE structure and replaced the arithmetic with three hand-written Triton kernels (a tiled fp32 matmul used for router, both expert GEMMs and all backward transposes, plus SiLU-mul forward/backward), with routing done by packing biased scores into int64 keys for an exact torch.topk selection.
                            config: Fixed tile shape only: BM=64, BN=64, BK=32 for `_mm_kernel` and EBLOCK=1024 for the SiLU kernels -- no sweep of tile sizes, num_warps or num_stages was run. The single A/B performed was on dot precision: the shipped version (sha c2b9763...) uses `input_precision="ieee"` for the router matmul and `"tf32x3"` for the expert matmuls; an alternate variant (sha a930076...) edited the load/dot precision lines and was benchmarked as /tmp/final-split.json, then reverted back to c2b9763 for the final measured run, i.e. the variant did not win. No other configuration space was explored.
   10      0.3035  0.3035   Kept the starter's per-expert masked-loop MoE structure but made the shared Triton matmul kernel faster by enlarging its tiles, adding explicit launch config, and switching the fp32 dot path from strict IEEE to tf32x3 for everything except the router projection.
                            config: Matmul tiling swept from the starter's BM=BN=BK=16/16/16 to BM=BN=64, BK=32 with `num_warps=4`, `num_stages=3` (single step, no intermediate tile sizes tried, no warp/stage sweep). Dot precision swept from uniform `input_precision="ieee"` to a per-call `IEEE` constexpr: `True` for the router projection only, `tf32x3` for w1/w2 forward and for both `dA = dC@B^T` and `dB = A^T@dC` backward GEMMs. Two harness runs were recorded, `./bench.py --json /tmp/moe-baseline.json` (after the tiling edit) and `./bench.py --json /tmp/moe-native-split.json` (after the precision-split edit), plus a final `./bench.py --json /tmp/moe-final.json`; the printed numbers are truncated in the record so no per-config delta can be quoted. No dispatch-strategy alternative (sorted segments, capacity bins, fused gather) was ever benchmarked.
   11        --    0.0000   Wrote a from-scratch Triton MoE (forward + backward) using a sorted-expert dispatch: fp32 router GEMM + sigmoid, exact bit-packed integer top-A keys with lower-index tiebreak, torch.argsort-based permutation, per-expert offset/tile tables, and Triton expert GEMMs with bf16 hi/lo split-precision dots for the weight gradients.
                            config: Fixed structural constants ROW=64 rows per expert tile, KBLK=32, `_expert_in` grid (cap, cdiv(2F,64)), `_expert_out` grid (cap, cdiv(D,64)), `_combine` grid (T, cdiv(D,256)), `_router` grid cdiv(T,32), `_dw1`/`_dw2`/`_drw` grid over E. Launch-parameter sweep on bench shapes p1/p2 with snapshot/restore between trials: baseline num_warps=4, num_stages=3; tried num_warps=8/num_stages=3 (reverted), num_stages=2 (reverted), num_stages=4 (reverted) — the agent kept the original 4 warps / 3 stages in every case, so none of the variants was retained. Also A/B'd two precision variants saved as /tmp/kernel-split.py and /tmp/kernel-precise.py (bf16 hi/lo splitting in the dw kernels vs. a reduced form); the split version was kept. Numerical checking was done almost entirely at one tiny shape T,D,F,E,A=17,24,13,8,3 plus a small non-power-of-two set (37,45,23,9,1), (29,33,17,8,8), (31,47,19,13,...), never at the large F where dw1 later failed.
   12        --    0.0000   Kept the starter's per-expert masked-gather MoE structure intact and sped up only its arithmetic: enlarged the shared Triton matmul tile from 16x16x16 to 64x64x32 and replaced true-fp32 `input_precision="ieee"` dots in the expert GEMMs with a bf16 hi/lo split-accumulate (2 terms, then 4 terms) on tensor cores, leaving the router matmul at ieee fp32.
                            config: Tile sizes: starter BM,BN,BK = 16,16,16 replaced once with 64,64,32 (no sweep of alternative tiles, no num_warps/num_stages sweep recorded beyond the single edit). Precision: three configurations benched with ./bench.py --json -- (a) 64x64x32 with ieee fp32 dots -> /tmp/moe-current.json; (b) plus bf16 hi/lo split with 3 dot terms (ah*bh + al*bh + ah*bl) -> /tmp/moe-split.json; (c) split extended to 4 terms adding al*bl -> /tmp/moe-split4.json; a final confirming run after cleanup -> /tmp/moe-final.json. The agent never quoted the numeric outcome of any of these runs in the trajectory, so no relative timing per configuration is recorded; the 4-term version is what shipped. Ad-hoc correctness checks against reference.py were run on small random shapes (float32 37x45x23,E=9,A=1; float32 29x33x17,E=8,A=8; plus bf16) and on tiny/saturated/mixed probes, none of which exposed the fp32 gradient drift that the larger shape sweep found.

  speed  = fraction of the production kernel's throughput; 1.0 would match it. `--` means the attempt was never timed because it computed the wrong answer.
  score  = what this attempt actually scored. It EQUALS the speed above when every gate and rule passed, and is 0.0000 if anything at all was wrong with the attempt -- so the two columns differ only on failures.
  config = the tuning values that attempt actually ran (tile sizes, warps, stages) and what its sweeps showed. Do not re-search values already listed; start from what they learned.
  *      = best score so far, and the one to beat.
  R      = REJECTED. It ran and was measured, and still scored nothing, because it broke one of the rules at the end of this note. Whatever it did to earn its speed, do not copy that part.

  EXPLORE A NEW IDEA, OR EXPLOIT ONE THAT ALREADY PERFORMS -- your call, and say which in your final message. Exploit means taking an approach above that already ran and making it faster, or fixing what stopped it. Explore means a structurally different idea, worth it only if the ones that ran have stopped improving. Repeating an approach that scored 0.0000, without changing what made it score 0.0000, is neither.

## Rules this session has already broken

  deterministic_routing_and_combine  (iteration 11)
  "exact bytes you shipped"  (iteration 5)

  These cost a full zero each time, whatever the kernel measured. Check your submission against every line here before you finish, not only against the one the last attempt broke.

## Where iteration 12 fell short

  g1/float32 scale=2.432: drouter_w[5 elements outside tolerance, max abs 8.130e-02]
  g3/float32 scale=1.573: drouter_w[1 elements outside tolerance, max abs 5.811e-02] dw1[162 elements outside tolerance, max abs 1.718e-02] dw2[202 elements outside tolerance, max abs 1.834e-02]

  This same failure has now happened 2 times (attempts 8, 11). Whatever you have been changing between attempts is not what is causing it.

## What iteration 12 said about itself (UNVERIFIED)

  what it says stopped it : It computed the wrong answer in float32: the bf16 hi/lo split-dot path used for the expert GEMMs and their backward GEMMs loses fp32 precision, so at fp32 shape-sweep configs the router-weight gradient drifted (g1 scale=2.432: drouter_w 5 elements off, max abs 8.13e-02) and at g3 scale=1.573 drouter_w plus dw1 (162 elements, max 1.718e-02) and dw2 (202 elements, max 1.834e-02) exceeded tolerance. bf16 smoke/determinism/stability/edge stages were clean, so the failure is specifically the fp32 dtype path being run through the bf16-split accumulator rather than a true fp32 dot.
  what it wanted to try next: Make the split-precision path dtype-conditional at the launch site: when the incoming tensors are float32 pass IEEE=True (or use a 3-way fp32 split / tf32x3 with a proper residual term) for the dw1/dw2 and drouter_w-feeding GEMMs, keeping the 2- or 4-term bf16 split only for the bf16 inputs where the reference itself is bf16-precision, so the fp32 shape-sweep configs recover exact fp32 accumulation while bf16 keeps the tensor-core speed.

  This is the attempt's own account, not a measurement. Treat the next idea as a lead worth checking, not as a conclusion -- and if you disagree with it, say why and do something else.

## Where the time is going (best iteration so far, 6 timed cases)

  Your best case reaches 0.9134 of the production kernel. Your worst reaches 0.4428. That is a 2.1x spread across cases that differ only in shape.

  worst case: yours 16.96 ms against production 7.51 ms -- 2.3x slower
  best case : yours 51.34 ms against production 46.86 ms -- 1.1x slower
  the worst case is 0.3x the wall-clock of your best case
  The spread is narrow, so every case is limited by the same thing. Making the kernel fundamentally cheaper will move all of them together.

## What your implementation actually is (iteration 12, assessed, NOT part of your score)

  yes  diag_dispatch_strategy
  NO   diag_measured_before_submit
  yes  diag_router_backward
  NO   diag_tie_handling
  NO   diag_weight_grad_precision

  The 'NO' answers are why you are slow. They do not lower your reward directly -- the clock already does that -- but each one names a structural property the fast implementations have and this one does not:
    [diag_tie_handling] Mechanism is present: `_select_topA` builds composite int64 keys -- `u = torch.where(ib >= 0, ib ^ 0x80000000, (~ib) & 0xFFFFFFFF)` on the float32 bits of `b = s + router_b`, then `keys = (u << 10) | (E - 1 - e_idx)` fed to `torch.topk(..., largest=True)`, so descending key order is score-descending/index-ascending with no separate tie pass or post-hoc repair. However, the required verification is missing: the trajectory shows only generic bench runs ([18], [21], [23] `./bench.py --json ...`) and an ad-hoc reference comparison at [25] over random cases `(float32,37,45,23,9,1,1.0)`, `(float32,29,33,17,8,8,1.7)`, `(bf16,...)` -- none of which constructs an exact-tie shape (e.g. duplicated/identical biased scores across experts), so the lower-index tie-break path was never exercised against `reference.py` before submission. The key packing also silently assumes E <= 1024 (the hard-coded `<< 10`), an untested boundary. [PANEL SPLIT, resolved 1-3: an independent check=fail, an independent check=fail, an independent check=pass, an independent check=fail]
    [diag_weight_grad_precision] The weight-gradient chain runs through `_MM.backward`, which calls `_mm_launch(a.t(), dc, ctx.ieee)` with `ctx.ieee=False` for both expert projections (`_MM.apply(h, w1s[e])` and `_MM.apply(act, w2s[e])` are called without the ieee flag), so dw1/dw2 are computed by the `IEEE=False` branch of `_mm_kernel`: a bf16 hi/lo split emitting four tensor-core dots (`tl.dot(ah,bh) + tl.dot(al,bh) + tl.dot(ah,bl) + tl.dot(al,bl)`), with only the router matmul kept at true fp32 via `input_precision="ieee"`. Nothing in the file ties this choice to a disclosed gradient tolerance -- the only precision commentary is stale starter text claiming "input_precision='ieee' keeps it full fp32... This one kernel carries the entire arithmetic load" -- and the trajectory shows the strategy was arrived at empirically: step [20] added the split and benched it (`/tmp/moe-split.json`), then step [22] edited the `al`/`bl` lines to add the fourth cross term and re-benched (`/tmp/moe-split4.json`), i.e. terms were added until the harness stopped complaining rather than budgeted against the stated tolerance up front. [agreed by an independent check, an independent check, an independent check]
    [diag_measured_before_submit] The trajectory shows bench.py invoked three times (steps 18, 21, 23: './bench.py --json /tmp/moe-current.json', '--json /tmp/moe-split.json', '--json /tmp/moe-split4.json'), but each run is immediately followed by further str_replace edits to kernel.py (steps 20, 22) or by unrelated correctness scripts (steps 25-26), and the log cuts off mid-command at step 26 without showing any bench.py run executed after the last edit or any reported numbers tied to the final submitted kernel.py bytes. There is no evidence that the exact final kernel.py content shown in the submission was the one measured, nor that any reported performance numbers came from a run on that final version. [PANEL SPLIT, resolved 2-2: an independent check=pass, an independent check=fail, an independent check=fail, an independent check=pass]

## What to do next

THE SAME WRONG-ANSWER BUG HAS NOW KILLED 2 ATTEMPTS (iterations 8, 11):
    g2/bfloat16 scale=1.885: dw1[20022 elements outside tolerance, max abs 5.000e-01]

It has recurred across implementations that share almost no code with each other, so it is not a property of any one design and rewriting from scratch will not remove it. You are shipping a kernel whose gradients you have not checked.

`./bench.py` compares your output AND all five gradients against `reference.py` before it times anything, and refuses to print a speed number when they disagree. Run it. Do not submit until it prints `correctness ok`.

A forward pass can be right while the backward is garbage: that is this exact failure, every time. If bench.py reports correctness ok and the graded run still fails you, say so explicitly in your final message -- the difference is then the shape, not the algebra, and that is worth knowing.

## Rules that hold for every iteration

  - Your final edit to kernel.py must be followed by a bench.py run. Any number you report must come from a run on the EXACT bytes you shipped.
  - Do not report a speedup measured on a variant you did not submit. Your numbers are re-measured independently and a mismatch is a violation of the honest-measurement requirement, not a rounding difference.
  - Time the combined forward-and-backward pass with CUDA events or an explicit synchronise after discarded warmup runs. Forward-only timing is a violation.
