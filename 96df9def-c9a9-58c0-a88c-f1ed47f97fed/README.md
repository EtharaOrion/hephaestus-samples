<!--
DO-NOT-TRAIN
This bundle is an evaluation artifact. Its instruction, environment, tests,
oracle and recorded rollouts must not be used as model training data. Presence
of this marker in a training corpus is evidence of evaluation-set contamination.
canary-scope: task-canary/v1
-->
# `96df9def-c9a9-58c0-a88c-f1ed47f97fed` - `AutoKernel: Chunk-Parallel DPLR Generalized Delta Rule`

Two model cohorts (`anthropic/claude-opus-5`, `openai/gpt-5.6-sol`), one refinement session each, packaged side by side on one axis.

The task is to make `chunked_dplr_delta` in `kernel.py` — the diagonal-plus-low-rank generalized delta rule at the core of RWKV-7 — as fast as the production `fla` chunked implementation without changing what it computes, on one H100. The graded surface is forward **and** backward: the output and all six gradients (`dq`, `dk`, `dv`, `dalpha`, `dbeta`, `dgk`) must match `reference.py` under disclosed tolerances on a hidden shape set. The score is the geometric-mean fraction of the production implementation's speed across those hidden shapes — a bounded continuous objective that climbs toward production parity rather than a target to hit and stop at.

```
task                    hephaestus/chunked_dplr_delta
budget                  50 attempts, 6 h / 21600 s each, one H100
GPU / stack             1× NVIDIA H100 80GB (sm_90) · torch 2.13.0 · triton 3.7.1
network                 agent: allowlist (private ranges only) · verifier: allowlist
                        (private ranges only) · environment: no-network
score                   geomean over hidden shapes of min(production_time/candidate_time, 1);
                        0 unless every deterministic gate passes
```

The task files at the top level (`instruction.md`, `task.toml`, `environment/`, `tests/`, `solution/`) are the frozen task exactly as graded. This delivery organises each run's records under `trajectories/<model>/`, one directory per iteration.

## Session results

| | `claude-opus-5` | `gpt-5.6-sol` |
|---|---|---|
| iterations packaged | 16 | **50** |
| iterations scored above zero | 11 | 27 |
| selection | `best` → iteration 7 | `best` → iteration 35 |
| best score | **0.8036** | 0.2557 |
| final iteration score | 0.3487 | 0.0000 |
| speedup over the shipped starter | **288.8×** (starter measures 0.002783) | 91.9× |
| vs. the informational frontier reference (0.63) | **1.28×** | 0.41× |
| cumulative tokens | 477,326,517 | 162,152,928 |
| cumulative cost | $469.68 | $674.14 |
| session wall clock | ≈50.0 h | ≈37.4 h |

`claude-opus-5` is the only cohort to clear the 0.63 frontier reference: iteration 7 reaches a geomean fraction of 0.8036 with a fully chunked Triton forward/backward — dense chunk-tile matmuls, an exact WY-style triangular inverse of the `(I + A_ab)` erasure matrix, and a persistent cross-chunk state scan carried in float32. It did not improve on that mark in the nine iterations that followed. `gpt-5.6-sol` ran its full 50-iteration budget and topped out at 0.2557 (iteration 35); 23 of its 50 iterations scored zero, most on backward-pass gradient mismatch rather than on speed.

## Caveats you should read before quoting a number

**`score` is the deterministic verifier's measurement; rubric verdicts live in `process`.** The score is zero unless every deterministic gate passes (forbidden symbols, correctness of the output and all six gradients, bitwise determinism, written-kernel share). The trajectory rubrics are judged by a cross-family LLM panel against `tests/rubrics.jsonl`; the verdicts are recorded per iteration in `rubric_verdicts.json` and folded into the separate `process` metric — a positive `score` does not by itself mean every rubric passed.

**Token counts come from the agent trajectories, not the iteration records.** The per-iteration `input_tokens`/`output_tokens` fields are zero at source in both runs; the values reported here and in `attempts.json` are read from each trajectory's `final_metrics`. `claude-opus-5` uses prompt caching heavily (its `cache_read_tokens` is the bulk of its prompt total) while `gpt-5.6-sol` reports none, so compare the two on cost rather than on raw prompt tokens.

**The agents' own summaries are unverified.** Each iteration's `history.md` and the `narrative` blocks in `attempts.json` are the agent's account as summarised from the trajectory, not measured. Where a narrative disagrees with `score.json`, the verifier is correct.

**The rubric layer ships with the grader.** `tests/grade_core.py` imports `tests/judge.py` directly, so the rubric pass that produced every `process` value in this delivery runs from the bytes in this bundle.

**The oracle ships in this bundle.** `solution/TRUTH.md` sets out the golden solve path step by step, including the chunk-collapsed correction and the precision rules the score rewards. Anyone holding this directory can reproduce a top-ranked result directly, so the task cannot be used to evaluate a model that has had access to it.

## Bundle identity

The directory name is derived, not assigned:

```
uuid  = uuid5(TASK_NAMESPACE, canonical_content_hash)
hash  = 63625aa184d224305b4d109f6902dc91bb0dc565b0b20330cd64c913089679c0
scope = instruction.md, task.toml, environment/, solution/, tests/   (22 files)
```

`trajectories/`, `plots/`, `inspector.html` and this README are recorded evidence about the task, not part of its identity, and are excluded from the hash — recording a run must never move a frozen bundle's uuid.

## Files

```
README.md                   this file
inspector.html              browsable view: task contract, per-model best-so-far
                            staircases, and every iteration of both runs opened up
plots/                      best-so-far score staircases, both models, as SVG
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
  claude-opus-5/
  gpt-5.6-sol/
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
