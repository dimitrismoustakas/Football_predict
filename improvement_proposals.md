# Improvement Proposals

## Current Repo Snapshot

This repo trains one canonical model for match result (`Home/Draw/Away`).
The current production family is a gated residual neural net that starts from bookmaker implied probabilities and learns when to deviate from the market.

Relevant code anchors:
- `training/train_main_model.py`
- `training/configs/main_models/result.json`
- `training/models/neural_net.py`
- `training/train_utils.py`
- `preprocessing/build_understat_features.py`

## Proposal Backlog

### Near-term
- `P001`: Re-run feature-family search for the current canonical result baseline.
- `P002`: Re-tune architecture depth, width, and gate settings for the result model.
- `P003`: Add missing-value indicators instead of relying only on post-scaling zero fill.
- `P004`: Compare `StandardScaler` against robust or quantile scaling.
- `P005`: Add explicit home-away deltas and interaction features.
- `P006`: Add post-hoc calibration on the epoch-selection season.
- `P007`: Re-weight recent seasons more heavily in training.

### Medium-term
- `P008`: Train strong tree-model baselines such as `LightGBM`, `CatBoost`, or `XGBoost`.
- `P009`: Add league-specific heads on top of a shared trunk.
- `P010`: Add richer bookmaker decomposition such as opening/closing prices and bookmaker dispersion.
- `P011`: Extend player-derived features beyond the current aggregate block.
- `P012`: Add trend and volatility features for recent team state.

### Evaluation and process
- `P013`: Keep paired prediction archives for every serious experiment.
- `P014`: Add bootstrap confidence intervals for held-out metric deltas.
- `P015`: Standardize league-slice and class-slice evaluation output.
