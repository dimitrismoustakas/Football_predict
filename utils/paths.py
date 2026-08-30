from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXPERIMENT_METRICS_DIR = ARTIFACTS_DIR / "experiment_metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"
MAPPINGS_DIR = ARTIFACTS_DIR / "mappings"


def project_path(value: str | Path) -> Path:
	path = Path(value)
	return path if path.is_absolute() else PROJECT_ROOT / path
