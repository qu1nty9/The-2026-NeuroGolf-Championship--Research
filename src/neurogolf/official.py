from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from types import ModuleType

from .config import EXCLUDED_ONNX_OPS, ONNX_FILE_SIZE_LIMIT_BYTES, PATHS


os.environ.setdefault("MPLCONFIGDIR", str(PATHS.root / ".cache" / "matplotlib"))

RUNTIME_DEPENDENCIES = {
    "IPython.display": "ipython",
    "matplotlib.pyplot": "matplotlib",
    "numpy": "numpy",
    "onnx": "onnx",
    "onnx_tool": "onnx-tool",
    "onnxruntime": "onnxruntime",
}


def missing_runtime_dependencies() -> list[str]:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    missing = []
    for import_name, package_name in RUNTIME_DEPENDENCIES.items():
        try:
            importlib.import_module(import_name)
        except ModuleNotFoundError:
            missing.append(package_name)
    return sorted(set(missing))


def load_official_utils(data_dir: str | Path = PATHS.data) -> ModuleType:
    """Import the competition utility module and point it at local data."""

    missing = missing_runtime_dependencies()
    if missing:
        packages = ", ".join(missing)
        raise RuntimeError(
            "Official NeuroGolf utilities need missing runtime dependencies: "
            f"{packages}. Install the project requirements before ONNX validation."
        )

    data_path = Path(data_dir).resolve()
    utils_path = data_path / "neurogolf_utils"
    if not utils_path.exists():
        raise FileNotFoundError(f"Official utilities not found: {utils_path}")
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    module = importlib.import_module("neurogolf_utils")
    if hasattr(module, "_NEUROGOLF_DIR"):
        module._NEUROGOLF_DIR = str(data_path) + "/"
    return module


def competition_score(memory_bytes: int | float, params: int | float) -> float:
    cost = max(1.0, float(memory_bytes) + float(params))
    return max(1.0, 25.0 - math.log(cost))


def constraints() -> dict[str, object]:
    return {
        "excluded_ops": EXCLUDED_ONNX_OPS,
        "onnx_file_size_limit_bytes": ONNX_FILE_SIZE_LIMIT_BYTES,
    }
