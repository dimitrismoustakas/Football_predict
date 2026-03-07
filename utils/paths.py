from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRACKED_ASSETS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = TRACKED_ASSETS_DIR / "models"
MAPPINGS_DIR = TRACKED_ASSETS_DIR / "mappings"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
