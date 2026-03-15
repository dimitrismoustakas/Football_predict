---
name: continuous-experiment-runner
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
- Do not write markdown reports unless the user explicitly asks for one.
- Do not commit generated model bundles under `artifacts/models/`.
- Do not rewrite prior rows in `artifacts/experiment_metrics/result_main_runs.tsv`.
- Name experiment branches `experiment/<name>`.
- Use CV `log_loss` as the decision metric as long as the test set `log_loss` is not getting worse. If CV `log_loss` is staying the same but test set `log_loss` is improving, that is still a win. But you should always try to improve CV `log_loss` as your primary objective. The relevant digits for `log_loss` are up to the 6th decimal place, for changes smaller than that consider the change to be practically zero.
- Compare against the latest comparable row in `artifacts/experiment_metrics/result_main_runs.tsv`.
- Cheap local prescreens are allowed when they help rank nearby ideas, but they do not replace the canonical trainer and must not append to the TSV ledger.
- When ranking nearby candidates, prefer a stricter local support scorer: use the epoch-selection split only to choose `best_epoch`, then evaluate that fixed epoch count across the objective folds without appending a ledger row.
- Do not trust the epoch-selection season by itself for close calls; in this repo it is noisy enough to produce false positives.
- If a prescreened direction looks good, confirm the exact candidate with the canonical trainer before treating it as a real improvement.
- If a branch improves canonically and nearby variants still look promising, keep iterating on that same branch until the local neighborhood looks exhausted.
- Do not pause for user confirmation between iterations.
- Running out of local ideas is not enough reason to stop; if the current search space is exhausted, actively look for papers, repo references, or other credible external sources of new experiment ideas and try to translate them into candidates.

Everything relevant to improving the canonical path is fair game.

## Canonical experiment path

Default commands:

1. refresh any required data inputs
2. run `uv run python training/train_main_model.py --description "short text of what this experiment tried"`

The `--description` flag is **mandatory** for every canonical run. Write a concise description of what the experiment tried (e.g. "remove season_progress feature", "resnet backbone with cross-attention", "increase LR to 0.04"). Do not use commas in the description (the TSV uses tabs but commas in descriptions still cause readability issues).

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
| `test_log_loss` | Watch-only test season log_loss |
| `cv_metrics_json` | Full CV metrics dict as JSON |
| `test_metrics_json` | Full test metrics dict as JSON |

**Every** canonical run gets logged — including discards. The trainer writes the row with an **empty `status`** field. After reviewing the training output and delta, **you must update the status** in the TSV yourself:
- `keep` — worth keeping as the new reference point
- `discard` — did not improve or not worth keeping
- `crash` — OOM or other failure (use `0.000000` for cv_log_loss)

To update the status, edit the last line of the TSV and fill in the `status` column. Do not leave it empty.