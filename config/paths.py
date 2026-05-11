"""Central path constants for the repository (optional import from scripts)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CAPTION_DATASET_CSV = PROCESSED_DATA_DIR / "caption_dataset_final_full.csv"
SEEDS_LIST_TXT = PROCESSED_DATA_DIR / "seeds_list.txt"
RAW_IMAGE_DATASET_DIR = DATA_DIR / "raw dataset"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def experiment_metrics(name: str) -> Path:
    return EXPERIMENTS_DIR / name / "metrics"


def experiment_artifacts(name: str) -> Path:
    return EXPERIMENTS_DIR / name / "artifacts"
