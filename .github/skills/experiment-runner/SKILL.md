---
name: experiment-runner
description: Pick one experiment idea, make changes on its own branch, run the canonical trainer, and use the TSV and latest-metrics JSON as the experiment record.
---

# Experiment runner

Use this skill when asked to run a repo experiment.

## Mission

Complete one experiment end to end:

1. choose one idea unless the user already picked one
2. create or use a dedicated experiment branch
3. implement changes for that idea
4. run the canonical evaluation path with `training/train_main_model.py`
5. inspect the outputs written by the trainer
6. leave the branch in a reviewable state

## Required context

Read these first:

- `.github/copilot-instructions.md`
- `README.md`
- `docs/evaluation_policy.md`
- the most relevant code for the chosen idea

## Hard rules

- Work on exactly one idea per branch.
- Preserve the canonical workflow unless the user explicitly asks to change it.
- Prefer editing existing canonical files over adding new scripts.
- Keep the experiment surface small.
- Do not write markdown reports unless the user explicitly asks for one.
- Do not commit generated model bundles under `artifacts/models/`.
- Do not promote a new baseline unless the user explicitly asks.

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
- trying a reasonable model or optimizer adjustment
- trying a reasonable training-loop or batch-size adjustment
- trying a reasonable feature-engineering or feature-selection adjustment
- tightening the canonical path where the current setup looks overbuilt

Prefer ideas that are plausible and clean.
There is a slight preference for simplicity, but not a hard rule. You can always pick a more complex idea if it looks like it has a better chance of improving the main metric.
Keep the spirit: if two ideas look similar in expected benefit, prefer the simpler one.

## Canonical experiment path

Default commands:

1. refresh any required data inputs
2. run `uv run python training/train_main_model.py`

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

## Promotion

Promotion is separate.

Unless the user explicitly asks for promotion, stop after the experiment is implemented and evaluated.
