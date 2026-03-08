---
name: experiment-runner
description: Pick one experiment idea, make changes on its own branch, run the canonical trainer, and use the TSV and latest-metrics JSON as the experiment record.
---

# Experiment runner

Use this skill when asked to run a repo experiment.

## Mission

Run a useful experiment loop from `main`, or keep iterating on the current best experiment branch when a successful line is still live:

1. choose a promising next direction unless the user already picked one
2. make whatever supporting analysis, ablations, or sweeps help you decide
3. put the chosen source changes on an experiment branch
4. run the canonical evaluation path with `training/train_main_model.py`
5. inspect the TSV and latest-metrics JSON
6. leave the branch in a reviewable state

## Required context

Read these first:

- `.github/copilot-instructions.md`
- `README.md`
- `docs/evaluation_policy.md`
- the most relevant code for the chosen idea

## Hard rules

- Preserve the canonical workflow unless the user explicitly asks to change it.
- Prefer editing existing canonical files over adding permanent experiment scaffolding.
- Keep the repo surface lean.
- Do not write markdown reports unless the user explicitly asks for one.
- Do not commit generated model bundles under `artifacts/models/`.
- Do not rewrite prior rows in `artifacts/experiment_metrics/result_main_runs.tsv`.
- Use CV `log_loss` as the decision metric.
- Treat the fixed test season as watch-only monitoring output.
- Compare against the latest comparable row in `artifacts/experiment_metrics/result_main_runs.tsv`.
- Cheap local prescreens are allowed when they help rank nearby ideas, but they do not replace the canonical trainer and must not append to the TSV ledger.
- If a prescreened direction looks good, confirm the exact candidate with the canonical trainer before treating it as a real improvement.
- If a branch improves canonically and nearby variants still look promising, keep iterating on that same branch until the local neighborhood looks exhausted.
- If a branch regresses clearly, prefer deleting or abandoning it rather than carrying dead ideas forward.

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

## Choosing the idea

If the user gives an idea, use it.

If not, choose one yourself from the code and current results.
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

Prefer ideas that are plausible and clean.
If two directions look similar, prefer the one that simplifies the canonical path.
If one direction is already showing incremental gains, prefer exhausting that neighborhood before jumping to a new family.

## Canonical experiment path

Default commands:

1. refresh any required data inputs
2. run `uv run python training/train_main_model.py`

Before spending a full canonical run, it is reasonable to do narrow branch-local support work such as:

- a tiny prescreen script
- a narrow ablation or small parameter sweep
- a smoke check for newly introduced model plumbing

Keep those helpers lean and avoid turning them into permanent experiment infrastructure unless they clearly belong in the canonical path.

The trainer is the default experiment harness and writes the main outputs to:

- `artifacts/experiment_metrics/result_main_runs.tsv`
- `artifacts/models/latest_main_model_metrics.json`

Use those as the default experiment record.

## What good completion looks like

A completed run should leave behind:

- code changes on the experiment branch
- one new row in `artifacts/experiment_metrics/result_main_runs.tsv`
- updated runtime candidate metrics in `artifacts/models/latest_main_model_metrics.json`
- a concise summary of what changed and whether the evidence looks stronger, weaker, or mixed
- if a successful path stayed live across multiple iterations, the branch should contain the best cumulative state found in that neighborhood
