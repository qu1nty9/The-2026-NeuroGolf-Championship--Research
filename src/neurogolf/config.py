from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    experiments: Path = PROJECT_ROOT / "experiments"
    reports: Path = PROJECT_ROOT / "experiments" / "reports"
    models: Path = PROJECT_ROOT / "models"
    candidate_models: Path = PROJECT_ROOT / "models" / "candidates"
    verified_models: Path = PROJECT_ROOT / "models" / "verified"
    submissions: Path = PROJECT_ROOT / "submissions"
    notebooks: Path = PROJECT_ROOT / "notebooks"


PATHS = ProjectPaths()

TASK_COUNT = 400
CHANNELS = 10
GRID_HEIGHT = 30
GRID_WIDTH = 30
GRID_SHAPE = (1, CHANNELS, GRID_HEIGHT, GRID_WIDTH)
SPLITS = ("train", "test", "arc-gen")

EXCLUDED_ONNX_OPS = (
    "Loop",
    "Scan",
    "NonZero",
    "Unique",
    "Script",
    "Function",
    "Compress",
)
ONNX_FILE_SIZE_LIMIT_BYTES = int(1.44 * 1024 * 1024)
