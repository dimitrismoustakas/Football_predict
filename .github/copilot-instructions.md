# Football Prediction Pipeline - AI Agent Instructions

## Project overview
This repo trains and serves one Torch model:
- match result (`Home/Draw/Away`)

The canonical training and evaluation loop is fixed and should be the default path for any branch experiment.

## Canonical workflow

### Feature pipeline
- Training features are built by `preprocessing/build_understat_features.py`
- Shared feature logic lives in `preprocessing/feature_engineering.py`
- Production features are built by `prod_run/build_prod_features.py`

### Main training entry point
- `training/train_main_model.py` trains the canonical model
- Frozen hyperparameters live in `training/configs/main_models/`

### Evaluation protocol
Always preserve this unless the user explicitly asks to change it:
- use `training/configs/main_models/evaluation.json` as the source of truth
- rolling expanding-window CV mean `log_loss` is the single decision metric
- the last pre-test season is reserved for epoch selection
- the fixed test season is watch-only and not part of branch acceptance
- use the latest comparable row in `artifacts/experiment_metrics/result_main_runs.tsv` as the default comparison point

## Research workflow
Keep the repo surface small.
If an experiment needs custom search or analysis code, prefer a narrow branch-specific helper over permanent research scripts.
Cheap local prescreens are encouraged when they help rank nearby ideas quickly, but they are only a guide.
Any candidate that matters still needs the canonical `training/train_main_model.py` run before it counts.

## Production
- `prod_run/pipeline.py` loads the canonical model bundle from `artifacts/models/`
- `prod_run/fetch_odds.py` fetches match-result prices
- Generated model bundles under `artifacts/models/` are runtime outputs and should not be committed
- `artifacts/experiment_metrics/result_main_runs.tsv` is the single experiment ledger and should be kept append-only

## GitHub tooling
- GitHub CLI is available at `C:/Program Files/GitHub CLI/gh.exe` on this machine even if `gh` is not on `PATH`
- prefer using it for branch/PR actions when needed

## Experiment surface
Keep the experiment surface lean:
- prefer editing `training/train_main_model.py` for canonical training changes
- use `artifacts/experiment_metrics/result_main_runs.tsv` as the default experiment ledger
- do not add report workflows or extra experiment registries unless the user explicitly asks for them

When iterating:
- start from `main` for a fresh line of inquiry
- if a branch improves on the canonical objective and nearby variants still look promising, keep iterating on that branch until the neighborhood is exhausted
- if a branch clearly regresses, abandon it rather than accumulating dead changes
- if a prescreen helper suggests a good direction, make sure the exact canonical candidate matches what was prescreened before trusting the result

For experiments, everything relevant to improving the canonical path is fair game.
This includes, when justified:
- model architecture
- optimizer
- hyperparameters
- training loop details
- batch size and model size
- feature engineering
- feature selection
- preprocessing choices

Do not limit experiments to abstract ideas only.
The agent should be free to choose any concrete change it thinks can improve the main metric.
It may use ablations, sweeps, helper analysis, or targeted searches when they help choose or validate the next branch.
Promising lines should be worked, not just sampled once.
It is better to iterate a live direction through a few coherent nearby adjustments than to abandon it after the first non-winning attempt.

## Skill
If the user asks to run an experiment, use the single `experiment-runner` skill.
It should:
- branch from `main` unless the user explicitly wants another base
- run the canonical evaluation path
- rely on `artifacts/experiment_metrics/result_main_runs.tsv` and `artifacts/models/latest_main_model_metrics.json` as the handoff
- avoid markdown reports unless the user explicitly asks for one

## Betting diagnostics
Use proper scoring metrics as the primary quality gate.
Betting diagnostics are secondary.

## Coding conventions
- Keep code simple and direct
- Prefer Polars for feature engineering
- Use `uv` for Python commands
- Python files should use tabs for indentation in this repo
- Avoid defensive wrappers that add no value
- Backwards compatibility is not required during cleanup
