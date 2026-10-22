# `bc81a86b-7118-5bce-a3c1-fcb6f2459870` - `AutoKernel: Chunk-Parallel Gated Delta Rule`

Two model cohorts (`anthropic/claude-opus-5`, `openai/gpt-5.6-sol`), one refinement session each, packaged side by side on one axis.

The task is to make `gated_delta_rule` in `kernel.py` — the chunk-parallel gated delta rule — as fast as the production chunked implementation without changing what it computes, on one H100. The graded surface is forward **and** backward: the output and all five input gradients (`dq`, `dk`, `dv`, `dg`, `dbeta`) must match `reference.py` under disclosed tolerances on a hidden shape set. The score is the geometric-mean fraction of the production implementation's speed across those hidden shapes — a bounded continuous objective that saturates at parity rather than a target to hit and stop at.

```
task                    hephaestus/kda_chunk_gdr_fwdbwd
budget                  100 attempts, 1.5 h / 5400 s each, one H100
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
| iterations packaged | 4 | **41** |
| iterations scored above zero | 2 | 36 |
| selection | `best` → iteration 4 | `best` → iteration 21 |
| best score | **1.0000** | 0.8106 |
| final iteration score | 1.0000 | 0.0000 |
| speedup over the shipped starter | **121.4×** (starter measures 0.008235) | 98.4× |
| vs. the informational reference (0.95) | **1.05×** | 0.85× |
| cumulative tokens | 36,744,510 | 110,949,772 |
| cumulative cost | $48.65 | $463.44 |
| session wall clock | ≈8.8 h | ≈19.9 h |

`claude-opus-5` reached production parity at iteration 4: a geomean fraction of 1.0 across the hidden graded shapes, from a chunked Triton forward and a hand-written Triton backward that reverses the scan and solves the transposed intra-chunk system on the same tile. `gpt-5.6-sol` climbed to 0.8106 by iteration 21 — beating the 0.95 reference on individual shapes but not on the geomean — and did not improve on it in the twenty iterations that followed.

## Caveats you should read before quoting a number

**`score` is the deterministic verifier's measurement; rubric verdicts live in `process`.** The score is zero unless every deterministic gate passes (forbidden symbols, correctness of the output and all five gradients, bitwise determinism, written-kernel share). The trajectory rubrics are judged by a cross-family LLM panel against `tests/rubrics.jsonl`; the verdicts are recorded per iteration in `rubric_verdicts.json` and folded into the separate `process` metric — a positive `score` does not by itself mean every rubric passed.

**The target does not gate.** `TARGET_FRACTION_OF_SOTA = 0.95` is informational: the score is the raw geometric-mean fraction, not divided by the target, and only a failed gate pays zero. A score above 0.95 therefore means the candidate beat that reference point, not that it saturated the metric.

**Token counts come from the agent trajectories, not the iteration records.** The per-iteration `input_tokens`/`output_tokens` fields are zero at source, so the values here and in `attempts.json` are read from each trajectory's `final_metrics`. `claude-opus-5` uses prompt caching heavily while `gpt-5.6-sol` reports none, so compare the two on cost rather than on raw prompt tokens.

**The agents' own summaries are unverified.** Each iteration's `history.md` and the `narrative` blocks in `attempts.json` are the agent's account as summarised from the trajectory, not measured. Where a narrative disagrees with `score.json`, the verifier is correct.

**The oracle ships in this bundle.** `solution/TRUTH.md` sets out the golden solve path step by step, including the chunking, the WY-form intra-chunk solve and the hand-written backward the score rewards. Anyone holding this directory can reproduce a top-ranked result directly, so the task cannot be used to evaluate a model that has had access to it.

## Bundle identity

The directory name is derived, not assigned:

```
uuid  = uuid5(TASK_NAMESPACE, canonical_content_hash)
hash  = c5e4484f4a2cb0ba521e505527a07f97de337c77a71991ab05776361966929e2
scope = instruction.md, task.toml, environment/, solution/, tests/   (18 files)
```

`trajectories/`, `plots/`, `inspector.html` and this README are recorded evidence about the task, not part of its identity, and are excluded from the hash — recording a run must never move a frozen bundle's uuid.

## Files

```
README.md                   this file
inspector.html              browsable view: task contract, per-model best-so-far
                            staircases, and every iteration of both runs opened up
plots/                      best-so-far score staircases, both models, as SVG
  iterations.svg            compact score-per-iteration summary (dots + staircase)
  score_by_iteration.svg    verifier score vs attempt index, best-so-far staircase
  score_by_tokens.svg       verifier score vs cumulative tokens (log-scale x)
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
    attempts.json           the run ledger, one record per iteration
    rollout.json            the session record: model identity, per-iteration rewards,
                            submission digests, final selection
    final_submission.py     the submission the declared selection rule chose
    final_submission.json   which iteration it came from, and why
    iteration-NN/           one directory per iteration
      config.json           the trial as configured
      result.json           the trial as it ended
      rubric_verdicts.json  the rubric review: overall pass, per-rubric verdicts
      history.md            the record of prior iterations this one was handed
      agent/trajectory.json   the structured agent trajectory
      artifacts/kernel.py     the submitted kernel
      verifier/score.json     the deterministic verifier's measurement — the authority
      verifier/score_metadata.json   gate-by-gate breakdown behind that score
```
