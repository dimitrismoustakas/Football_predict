---
name: proposal-promoter
description: Review a finished proposal branch, compare it against the incumbent using the repo's policy and scorecard, update accepted manifests only if the gates pass, and open a PR for accepted experiments.
---

# Proposal promoter

Use this skill after a `proposal-runner` branch is finished and has a report.

## Mission

Given a completed experiment branch:

1. read the experiment report
2. compare the candidate against the incumbent champion
3. apply the acceptance policy
4. assign one status: `accept`, `reject`, or `challenger`
5. update source-controlled acceptance records only if the candidate is accepted
6. for `reject` or `challenger`, carry a minimal record of the outcome onto `main`
7. delete the local experiment branch after non-accepted outcomes are recorded on `main`
8. open a PR only for accepted experiments

## Required context to read first

Always read these before deciding:

- `.github/copilot-instructions.md`
- `README.md`
- `docs/evaluation_policy.md`
- `docs/model_acceptance_scorecard_template.md`
- `training/configs/main_models/baselines.json`
- the report in `docs/experiment_reports/`
- any referenced output artifacts used as evidence

## Hard rules

- Proper scoring metrics decide promotion first.
- Betting diagnostics are secondary only.
- For binary, use policy primary metric `log_loss` and secondary `brier`.
- For multiclass, use policy primary metric `log_loss` and secondary `rps`.
- Do not promote based on profit, accuracy, or one lucky held-out slice.
- Do not rewrite history in `training/configs/main_models/baselines.json`; append accepted entries.
- Do not open a PR for `reject` or `challenger` outcomes.

## Incumbent source of truth

Treat `training/configs/main_models/baselines.json` as the authoritative local champion manifest unless the branch already contains a more explicit accepted-model registry.

If no separate registry/alias manifest exists, do not invent one unnecessarily.
Updating `training/configs/main_models/baselines.json` is sufficient for accepted promotions in this repo.

## Decision procedure

Use the scorecard template in `docs/model_acceptance_scorecard_template.md`.

A candidate is `accept` only if the policy gates pass, including:

1. weighted rolling-CV primary metric improves
2. at least 2 of 3 folds improve on the primary metric
3. latest fold is not materially worse
4. secondary proper metric is non-worse
5. held-out season is not clearly worse
6. no severe slice collapse appears
7. seed stability check passes for a serious candidate

Use:

- `challenger` when CV is convincingly better but the held-out evidence is inconclusive
- `reject` when any critical guardrail fails

## What to update for each outcome

### If `accept`

- update the experiment report to record the final decision
- append a new accepted entry to `training/configs/main_models/baselines.json`
- keep prior accepted entries intact
- if a local alias/registry manifest already exists in the branch, update it too
- open a PR from the current branch using the template in `pr-template.md`

### If `challenger`

- update the report decision section to `challenger`
- do not update `training/configs/main_models/baselines.json`
- do not open a PR
- commit the finalized report on the experiment branch if needed
- switch to `main` and bring over only the minimal outcome record needed to show the experiment was reviewed and not promoted
- delete the local experiment branch after the `main` update is committed

### If `reject`

- update the report decision section to `reject`
- do not update `training/configs/main_models/baselines.json`
- do not open a PR
- commit the finalized report on the experiment branch if needed
- switch to `main` and bring over only the minimal outcome record needed to show the experiment was reviewed and rejected
- delete the local experiment branch after the `main` update is committed

## Minimal `main` update for non-accepted outcomes

For `reject` and `challenger` outcomes, keep the `main` update small:

- prefer carrying over the finalized experiment report only
- do not merge the experimental code onto `main`
- use a short commit message that makes the outcome obvious
- if branch deletion is blocked, state that clearly

## PR requirements for accepted experiments

Open a PR only after all acceptance updates are committed.

The PR should include:

- proposal ID and short title
- concise implementation summary
- primary/secondary metric deltas
- held-out result summary
- report path
- explicit statement that baseline history was updated

Use the repository's GitHub tooling or `gh` if available.
If a PR cannot be opened because tooling/auth is unavailable, say so clearly after completing all local acceptance updates.

## Finish state

This skill is complete only when:

- the report has a final decision
- accepted branches have updated baseline history
- rejected/challenger branches leave baseline history untouched
- rejected/challenger branches have a minimal outcome record committed on `main`
- rejected/challenger local branches are deleted after the `main` update
- accepted branches have an opened PR or a clear note explaining why opening the PR was blocked