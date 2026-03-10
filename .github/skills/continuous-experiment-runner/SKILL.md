---
name: continuous-experiment-runner
description: Run experiment ideas continuously with no user feedback loop until blocked, exhausted, or explicitly stopped.
---

# Continuous experiment runner

Use this skill when asked to run repo experiments continuously without waiting for user feedback between iterations.

## Mission

Run an autonomous experiment loop that keeps going for as long as there is a credible next step:

1. choose a promising next direction unless the user already constrained the search space
2. make whatever supporting analysis, ablations, or sweeps help you decide
3. put the chosen source changes on an `experiment/<name>` branch
4. run the canonical evaluation path with `training/train_main_model.py`
5. inspect the TSV and latest-metrics JSON
6. decide the next move yourself and continue without asking the user for feedback

This skill is not one idea per invocation.
It should keep iterating until:
- the user explicitly stops the process

## Required context

Read these first:

- `.github/copilot-instructions.md`
- `README.md`
- `docs/evaluation_policy.md`
- the most relevant code for the current idea

## Hard rules

- Preserve the canonical workflow unless the user explicitly asks to change it.
- Prefer editing existing canonical files over adding permanent experiment scaffolding.
- Keep the repo surface lean.
- Do not write markdown reports unless the user explicitly asks for one.
- Do not commit generated model bundles under `artifacts/models/`.
- Do not rewrite prior rows in `artifacts/experiment_metrics/result_main_runs.tsv`.
- Name experiment branches `experiment/<name>`.
- Use CV `log_loss` as the decision metric.
- Treat the fixed test season as watch-only monitoring output.
- Compare against the latest comparable row in `artifacts/experiment_metrics/result_main_runs.tsv`.
- Cheap local prescreens are allowed when they help rank nearby ideas, but they do not replace the canonical trainer and must not append to the TSV ledger.
- When ranking nearby candidates, prefer a stricter local support scorer: use the epoch-selection split only to choose `best_epoch`, then evaluate that fixed epoch count across the objective folds without appending a ledger row.
- Do not trust the epoch-selection season by itself for close calls; in this repo it is noisy enough to produce false positives.
- If a prescreened direction looks good, confirm the exact candidate with the canonical trainer before treating it as a real improvement.
- If a branch improves canonically and nearby variants still look promising, keep iterating on that same branch until the local neighborhood looks exhausted.
- If a branch regresses clearly, abandon or delete it and move on instead of carrying dead ideas forward.
- Do not pause for user confirmation between iterations
- Running out of local ideas is not enough reason to stop; if the current search space is exhausted, actively look for papers, blog posts, repo references, or other credible external sources of new experiment ideas and try to translate them into branch-local candidates.

Everything relevant to improving the canonical path is fair game.
This includes, when justified:

- model architecture
- optimizer
- hyperparameters
- training loop details
- batch size and model size
- feature engineering
- feature selection
- preprocessing choices

The list is not exhaustive; if you find a credible next step that does not fit into one of those buckets, it is still fair game.

## Choosing the next idea

If the user gives a direction, stay within it until that neighborhood is exhausted.

If not, choose the next step yourself from the code and current results.
Good choices include:

- simplifying code while plausibly preserving or improving the main metric
- removing a component that may be unnecessary
- running a sweep or ablation to decide which component to keep
- trying a reasonable model or optimizer adjustment
- trying a reasonable training-loop or batch-size adjustment
- trying a reasonable feature-engineering or feature-selection adjustment
- tightening the canonical path where the current setup looks overbuilt
- recalibrating how the model uses bookmaker information
- iterating on a promising direction that is not yet better, when nearby adjustments are still plausible
- importing a credible idea from recent papers or other technical references when repo-local ideas are thinning out

Prefer ideas that are plausible and clean.
If two directions look similar, prefer the one that simplifies the canonical path.
If one direction is already showing incremental gains, keep working that neighborhood before jumping to a new family.
If the nearby neighborhood is exhausted, switch into idea-generation mode by reading papers and related technical material until you find another credible candidate.

## Continuous run policy

After each experiment cycle, choose one of these paths yourself:

1. continue on the same branch if the line still looks live
2. abandon the branch and start a fresh branch from `main` for a new direction if the line clearly regressed
3. if repo-local ideas are weak, do literature search and paper review

Default behavior is to keep going.
Do not stop just because one canonical run completed.

## Canonical experiment path

Default commands:

1. refresh any required data inputs
2. run `uv run python training/train_main_model.py`

Before spending a full canonical run, it is reasonable to do narrow branch-local support work such as:

- a tiny prescreen script that mirrors the canonical split closely enough to rank nearby candidates
- a narrow ablation or small parameter sweep
- a smoke check for newly introduced model plumbing

For nearby architecture or hyperparameter variants, the preferred support path is:

1. choose `best_epoch` on the fixed epoch-selection season
2. retrain each candidate for that fixed epoch count on each objective fold
3. compare mean CV `log_loss` locally without writing to the TSV ledger

Keep those helpers lean and avoid turning them into permanent experiment infrastructure unless they clearly belong in the canonical path.

The trainer is the default experiment harness and writes the main outputs to:

- `artifacts/experiment_metrics/result_main_runs.tsv`
- `artifacts/models/latest_main_model_metrics.json`

Use those as the default experiment record.