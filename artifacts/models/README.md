# Model artifacts

This directory is for runtime-generated model bundles only.

Tracked source configuration now lives under `training/configs/main_models/`.

Typical generated files:
- `over_under_model.pt`
- `over_under_model_config.json`
- `over_under_model_scaler.joblib`
- `result_model.pt`
- `result_model_config.json`
- `result_model_scaler.joblib`
- `latest_main_model_metrics.json`

Do not commit generated model binaries or scalers.
`latest_main_model_metrics.json` is also runtime output and should stay untracked.
Accepted baseline/history notes now live in `training/configs/main_models/baselines.json`.
