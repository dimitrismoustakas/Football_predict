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

## Evaluation workflow

Follow `docs/evaluation_policy.md` and use `docs/model_acceptance_scorecard_template.md` as the evidence structure.

## Standardized report requirement

Write the final report to:

- `docs/experiment_reports/YYYY-MM-DD_<idea-id>_<slug>.md`

Use the structure in `report-template.md` in this skill directory.

## Finish state

Before finishing:

1. ensure code changes and the report are saved on the experiment branch
2. ensure generated model bundles are not committed
3. commit the branch with a clear message
4. leave the branch ready for the separate promotion skill

Your handoff is complete only when a reviewer can read one report file and reproduce what happened.