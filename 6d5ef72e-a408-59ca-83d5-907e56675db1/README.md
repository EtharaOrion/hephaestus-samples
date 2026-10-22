# `6d5ef72e-a408-59ca-83d5-907e56675db1` - `AutoKernel: Routed MoE Forward+Backward`

Three model cohorts (`openai/muse-spark-1.2`, `openai/gpt-5.6-sol`, `anthropic/claude-opus-5`), one refinement session each, packaged side by side on one axis.

The task is to make `fused_moe` in `kernel.py` — a routed mixture-of-experts layer whose routing decision lives inside the operator — as fast as the production grouped-GEMM composition without changing what it computes, on one H100. The graded surface is forward **and** backward: the output and all four gradients (`dx`, `drouter_w`, `dw1`, `dw2`) must match `reference.py` under disclosed tolerances on a hidden shape set. The score is the geometric-mean fraction of the production composition's speed across those hidden shapes — a bounded continuous objective that climbs toward production parity rather than a target to hit and stop at.

```
task                    hephaestus/moe_routed_fwdbwd
budget                  100 attempts, 1.5 h / 5400 s each, one H100
GPU / stack             1× NVIDIA H100 80GB (sm_90) · torch 2.13.0 · triton 3.7.1
network                 agent: allowlist (private ranges only) · verifier: allowlist
                        (private ranges only) · environment: no-network
score                   geomean over hidden shapes of min(production_time/candidate_time, 1);
                        0 unless every deterministic gate passes
```

The task files at the top level (`instruction.md`, `task.toml`, `environment/`, `tests/`, `solution/`) are the frozen task exactly as graded. This delivery organises each run's records under `trajectories/<model>/`, one directory per iteration.

## Session results

| | `muse-spark-1.2` | `gpt-5.6-sol` | `claude-opus-5` |
|---|---|---|---|
| iterations packaged | 50 | 50 | **16** |
| iterations scored above zero | 35 | 29 | 8 |
| selection | `best` → iteration 4 | `best` → iteration 34 | `best` → **iteration 16** |
| best score | **0.3499** | **0.7100** | **1.0000** |
| final iteration score | 0.0835 † | 0.3448 | 1.0000 |
| speedup over the shipped starter | 4.2× | 8.5× | **12.0×** (starter measures 0.083188) |
| vs. the informational frontier reference (0.34) | 1.03× | 2.09× | 2.94× |
| cumulative tokens | 818,487,109 | 93,871,304 * | 1,065,490,994 |
| cumulative cost | $739.94 | $278.16 | $619.07 |
| session wall clock | ≈42.3 h | ≈29.1 h | ≈58.5 h |

`claude-opus-5` is the only cohort to reach production parity: iteration 16 matches the production grouped-GEMM composition on every timed shape (geomean fraction of SOTA 1.0), a Triton rewrite of the starter's per-expert Python loop into sort-based dispatch feeding segmented grouped GEMMs. `gpt-5.6-sol` climbed to 0.71 with the same family of approach; `muse-spark-1.2` topped out at 0.35, near the informational frontier but well short of parity.

## Caveats you should read before quoting a number

**`score` is the deterministic verifier's measurement; rubric verdicts live in `process`.** The score is zero unless every deterministic gate passes (forbidden symbols, correctness of output and all four gradients, bitwise determinism, written-kernel share). The trajectory rubrics are judged by a cross-family LLM panel against `tests/rubrics.jsonl`; the verdicts are recorded per iteration in `score_metadata.json` / `rubric_verdicts.json` and folded into the separate `process` metric — a positive `score` does not by itself mean every rubric passed.

**`cumulative tokens` is not uniformly reconstructable.** For `muse-spark-1.2` and
`claude-opus-5` the per-iteration `tokens.cumulative_total` is a true running total (the
iteration records are simply not stored in chronological order, so the field decreases as
you read down `rollout.json` by index). For `gpt-5.6-sol` that same field holds the
*per-iteration* total instead, and `cache_read`/`cache_write` are null in all 50 records —
so the 93,871,304 marked `*` above is the sum of the recorded input+output only, i.e. a
floor, not the exact session total.

**`muse-spark-1.2` iteration 50 is a reconstructed placeholder, not an agent submission.**
The `†` above marks it. Its `kernel.py` is byte-identical to the shipped starter
(`environment/kernel.py`, sha256 `8d8ef78a4eba…`), `speedup_vs_starter` is 1.0032, it records
`tool_calls: 0`, carries no trajectory and no narrative, and its own `action_summary` reads
`(reconstructed; unmodified starter, no trajectory)`. Read muse's "final iteration score" as
"the starter re-timed", not as a result the model produced. It is nonetheless counted in
muse's 35 iterations-scored-above-zero, so that count is 34 agent submissions plus this one.
It is also the single iteration in the bundle whose `process` is 1.0 while its
`rubric_verdicts.json` holds zero verdicts and `score.json` reports `rubrics: 0.0`.

**The agents' own summaries are unverified.** Each iteration's `history.md` and the `narrative` blocks in `attempts.json` are the agent's account as summarised from the trajectory, not measured. Where a narrative disagrees with `score.json` or `score_metadata.json`, the verifier is correct.

**The oracle ships in this bundle.** `solution/TRUTH.md` sets out the golden solve path step by step, including the routing, dispatch and precision tricks the score rewards. Anyone holding this directory can reproduce a top-ranked result directly, so the task cannot be used to evaluate a model that has had access to it.

**No per-iteration task snapshot ships.** Every iteration was handed the same frozen task — the top-level `environment/` staged into `/workspace` — plus the history block preserved as that iteration's `history.md`; each run's `iteration-01` has no history by construction.

## Files

```
README.md                   this file
inspector.html              browsable view: task contract, where the models fail, the
                            per-model best-so-far staircases, and every iteration of all
                            three runs opened up
plots/                      best-so-far score staircases, all three models, as SVG
  iterations.svg            compact score-per-iteration summary (dots + staircase)
  score_by_iteration.svg    verifier score (0-1) vs attempt index, best-so-far staircase
  score_by_tokens.svg       verifier score (0-1) vs cumulative tokens (log-scale x)
instruction.md              the objective handed to the agent
task.toml                   manifest: budget, refinement loop, GPU, network mode
environment/                the agent-visible starting files and their image
  kernel.py                 the deliberately slow starter (the only editable file)
  reference.py              the read-only correctness oracle
  bench.py                  the agent-side gate-then-time harness
tests/                      the grader that produced every score in this delivery,
                            rubrics.jsonl, hidden shape set
solution/                   PRIVATE — TRUTH.md, the golden solve path (see caveats)
trajectories/               one directory per run
  muse-spark-1.2/           the three packaged cohort runs (this delivery)
  gpt-5.6-sol/
  claude-opus-5/
  <run>/
    attempts.json           the run ledger, one record per iteration: outcome, process,
                            tokens, timing, mechanical counters, agent narrative
    rollout.json            the session record: model identity, per-iteration rewards,
                            submission digests, final selection
    final_submission.py     the submission the declared selection rule chose
    final_submission.json   which iteration it came from, and why
    iteration-NN/           one directory per iteration, in the harbor trial layout
      config.json           the trial as configured
      result.json           the trial as it ended: timestamps, tokens, cost, rewards
      rubric_verdicts.json  the rubric review: overall pass, per-rubric verdicts
      history.md            the record of prior iterations this one was handed
      agent/trajectory.json   the structured agent trajectory
      artifacts/kernel.py     the submitted kernel
      verifier/score.json     the deterministic verifier's measurement — the authority
      verifier/score_metadata.json   gate-by-gate breakdown behind that score
```
