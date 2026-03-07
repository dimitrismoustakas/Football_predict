---
name: proposal-runner
description: Pick one improvement idea from improvement_proposals.md, create a branch, implement the idea as fully as justified, run the repo's evaluation workflow, and write a standardized experiment report without making the promotion decision.
---

# Proposal runner

Use this skill when asked to autonomously take one idea from `improvement_proposals.md` from concept to a finished, reviewable experiment branch.

## Mission

Complete one proposal end to end:

1. choose one idea from `improvement_proposals.md`
2. create a dedicated experiment branch
3. implement the idea as thoroughly as possible without drifting into a different idea
4. run the evaluation workflow defined by the repo
5. write a standardized experiment report
6. commit the implementation and report on the branch

Do **not** decide whether the experiment is accepted.
Do **not** update `training/configs/main_models/baselines.json`.
Do **not** update any champion alias/registry.
Do **not** open a PR.

## Required context to read first

Always read these before choosing or implementing an idea:

- `.github/copilot-instructions.md`
- `README.md`
- `improvement_proposals.md`
- `docs/evaluation_policy.md`
- `docs/model_acceptance_scorecard_template.md`

Also read the most relevant implementation files for the chosen idea before editing.

## Branch and scope rules

- Work on exactly one proposal per branch.
- Prefer branch name format: `exp/<idea-id>-<short-slug>-<yyyymmdd>`.
- If the working tree already contains unrelated changes, do not mix them into the experiment.
- Prefer the smallest coherent implementation that fully tests the idea.
- Keep pushing the same idea until there is no obvious high-value extension left that still belongs to that idea.
- Do not silently bundle extra proposals just because they are nearby.

## How to choose the proposal

When the user does not specify an idea, choose one yourself.

Selection order:

1. Prefer the `Suggested First Pass` section in `improvement_proposals.md`.
2. Prefer ideas that are self-contained, testable, and compatible with current repo data.
3. Prefer ideas that do not require new external paid/private data. You can try to find external free data if a proposal required however.
4. Prefer ideas that can be implemented and evaluated cleanly in the existing canonical/research workflow.
5. Skip ideas that are already clearly implemented in the codebase.

Record the chosen proposal ID and exact proposal text in the final report.

## Implementation rules

- Preserve the canonical workflow unless the proposal explicitly requires a research-only path.
- Canonical feature flow:
  - `preprocessing/build_understat_features.py`
  - `preprocessing/feature_engineering.py`
- Canonical training entry points:
  - `training/train_main_model.py`
  - `training/train_all_models.py`
- Frozen main configs live in `training/configs/main_models/`.
- Runtime model outputs under `artifacts/models/` are generated outputs and must not be committed.
- Prefer Polars for feature work.
- Keep Python indentation with tabs.
- Avoid defensive wrappers and unrelated refactors.

## Evaluation workflow

Follow `docs/evaluation_policy.md` and use `docs/model_acceptance_scorecard_template.md` as the evidence structure.

### Primary policy

- Optimize on rolling CV.
- Use the held-out season only as a promotion gate.
- Judge by proper scoring metrics first.
- Treat betting diagnostics as secondary only.
- For binary, policy primary metric is `log_loss`, secondary is `brier`.
- For multiclass, primary metric is `log_loss`, secondary is `rps`.

### Minimum evidence to gather

For the chosen idea, gather as much of this as the repo currently supports:

1. Rolling CV evidence for the candidate.
2. Weighted CV summary using weights `0.2 / 0.3 / 0.5` for oldest / middle / latest folds.
3. Latest-fold guardrail check.
4. Held-out season evidence from the canonical retrain/eval path.
5. Slice checks when feasible:
   - league breakdown
   - latest CV fold
   - draw slice if multiclass
   - market-confidence deciles
   - calibration
6. Stability check with seeds `42`, `43`, and `44` for any serious challenger.

### Practical command guidance

Use `uv` for Python commands.
Typical command families include:

- data refresh when needed:
  - `uv run python data_collection/collect_understat.py`
  - `uv run python data_collection/collect_full_schedule.py`
  - `uv run python data_collection/collect_match_history.py`
  - `uv run python data_collection/collect_elo.py`
- training feature rebuild:
  - `uv run python preprocessing/build_understat_features.py`
- canonical evaluation:
  - `uv run python training/train_main_model.py`
  - `uv run python training/train_all_models.py`
- research-only scripts when the chosen idea truly belongs there:
  - `training/fixed_arch_sweep.py`
  - `training/architecture_search.py`
  - `training/result_architecture_search.py`
  - `training/feature_selection_search.py`
  - `training/analyze_residuals_by_decile.py`

Use the least custom path that gives credible evidence.
If you need a one-off helper script/notebook for measurement, keep it narrowly scoped and commit it only if it materially improves reproducibility.

## Standardized report requirement

Write the final report to:

- `docs/experiment_reports/YYYY-MM-DD_<idea-id>_<slug>.md`

Use the structure in `report-template.md` in this skill directory.

The report must include:

- experiment metadata
- chosen idea ID and quoted proposal text
- rationale for choosing the idea
- exact files changed
- implementation summary
- commands run
- artifact/output paths used as evidence
- rolling CV summary
- held-out season summary
- seed stability summary if run
- slice checks
- betting diagnostics
- a filled scorecard section
- final status set to `pending-promotion-review`
- suggested next follow-up only if it is still the same idea family

## Finish state

Before finishing:

1. ensure code changes and the report are saved on the experiment branch
2. ensure generated model bundles are not committed
3. commit the branch with a clear message
4. leave the branch ready for the separate promotion skill

Your handoff is complete only when a reviewer can read one report file and reproduce what happened.