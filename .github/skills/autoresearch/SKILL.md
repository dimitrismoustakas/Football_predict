---
name: autoresearch
description: Run experiment ideas continuously with no user feedback loop until blocked, exhausted, or explicitly stopped.
---

## Mission

You are the Lead ML Researcher for this project. Your job is to run autonomous experiments loop that keeps going until the user stops you:

1. choose a promising next direction unless the user already constrained the search space
2. make whatever supporting analysis, ablations, or sweeps help you decide
3. put the chosen source changes on an `experiment/<name>` branch
4. run the canonical evaluation path with `training/train_main_model.py`
5. inspect the TSV and latest-metrics JSON
6. decide the next move yourself and continue without asking the user for feedback

**FOR EVERY IDEA YOU TRY YOU MUST RUN/LOG THE BEST VERSION OF IT IN THE TSV WITH A DESCRIPTION EVEN IF YOU THINK IT'S BAD**. Otherwise you will keep trying ideas because you don't have infinite context.

This skill is not one idea per invocation.
It should keep iterating until:
- the user explicitly stops the process

## Required context

Read these first:

- `.github/copilot-instructions.md`
- `README.md`
- the most relevant code for the current idea

## Hard rules

- Preserve the canonical workflow unless the user explicitly asks to change it.
- Do not write markdown reports unless the user explicitly asks for one.
- Do not commit generated model bundles under `artifacts/models/`.
- Do not rewrite prior rows in `artifacts/experiment_metrics/result_main_runs.tsv`.
- Name experiment branches `experiment/<name>`.
- Use CV `log_loss` as the decision metric as long as the test set `log_loss` is not getting worse. If CV `log_loss` is staying the same but test set `log_loss` is improving, that is still a win. But you should always try to improve CV `log_loss` as your primary objective. The relevant digits for `log_loss` are up to the 6th decimal place, for changes smaller than that consider the change to be practically zero.
- Compare against the latest kept row in `artifacts/experiment_metrics/result_main_runs.tsv`.
- Do not pause for user confirmation between iterations.
- Running out of local ideas is not enough reason to stop; if the current search space is exhausted, actively look for papers, repo references, or other credible external sources of new experiment ideas and try to translate them into candidates.
- Do not use local prescreens, sweeps, or side analyses as a substitute for a canonical logged run.
- Analysis may help choose which variant to canonical-run, but an idea is not considered tried until at least one canonical run for that idea is logged in the TSV.
- Minor scalar tuning within the same mechanism counts as one idea; log the strongest version of that mechanism at least once.
- If the mechanism changes materially, it is a new idea and also needs its own canonical logged run.


Everything relevant to improving the canonical path is fair game.

## Canonical experiment path

Default commands:

1. refresh any required data inputs
2. run `uv run python training/train_main_model.py --description "short text of what this experiment tried"`

The `--description` flag is **mandatory** for every canonical run. Write a concise description of what the experiment tried (e.g. "remove season_progress feature", "resnet backbone with cross-attention", "increase LR to 0.04"). Do not use commas in the description. The description should be useful and readable enough and not cluttered with previous experiment details, it's a description of that specific experiment run.

The trainer is the default experiment harness and writes the main outputs to:

- `artifacts/experiment_metrics/result_main_runs.tsv`
- `artifacts/models/latest_main_model_metrics.json`

Use those as the default experiment record.

## Experiment ledger format

The TSV at `artifacts/experiment_metrics/result_main_runs.tsv` uses tab-separated columns (never use commas in descriptions):

| Column | Description |
|---|---|
| `recorded_at_utc` | ISO timestamp |
| `git_commit` | 7-char short hash |
| `git_branch` | Branch name |
| `cv_log_loss` | CV mean log_loss (the decision metric) |
| `delta` | Signed improvement over previous reference row |
| `best_epoch` | Epoch selected by early stopping |
| `status` | `keep`, `discard`, or `crash` |
| `description` | Short text of what this experiment tried |
| `cv_rps` | CV mean RPS (secondary) |
| `val_log_loss` | Epoch-selection season log_loss |
| `test_log_loss` | Consulted acceptance-season log_loss |
| `cv_metrics_json` | Full CV metrics dict as JSON |
| `test_metrics_json` | Full test metrics dict as JSON |

**Every** canonical run gets logged — including discards. The trainer writes the row with an **empty `status`** field. After reviewing the training output and delta, **you must update the status** in the TSV yourself:
- `keep` — worth keeping as the new reference point
- `discard` — did not improve or not worth keeping
- `crash` — OOM or other failure (use `0.000000` for cv_log_loss)

To update the status, edit the last line of the TSV and fill in the `status` column. Do not leave it empty.
