from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config import GRID_HEIGHT, GRID_WIDTH, PATHS, SPLITS, TASK_COUNT


TASK_RE = re.compile(r"task(\d{3})\.json$")


class TaskValidationError(ValueError):
    """Raised when a task JSON file does not match the expected ARC schema."""


def normalize_task_id(task_id: int | str) -> int:
    if isinstance(task_id, int):
        value = task_id
    else:
        match = re.search(r"\d+", str(task_id))
        if not match:
            raise ValueError(f"Could not parse task id from {task_id!r}")
        value = int(match.group(0))
    if value < 1 or value > TASK_COUNT:
        raise ValueError(f"Task id must be between 1 and {TASK_COUNT}: {value}")
    return value


def task_filename(task_id: int | str) -> str:
    return f"task{normalize_task_id(task_id):03d}.json"


def task_id_from_path(path: str | Path) -> int:
    match = TASK_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Not a task filename: {path}")
    return int(match.group(1))


def task_path(task_id: int | str, data_dir: str | Path = PATHS.data) -> Path:
    return Path(data_dir) / task_filename(task_id)


def iter_task_paths(data_dir: str | Path = PATHS.data) -> list[Path]:
    paths = [p for p in Path(data_dir).glob("task*.json") if TASK_RE.search(p.name)]
    return sorted(paths, key=task_id_from_path)


def load_task(task_id: int | str, data_dir: str | Path = PATHS.data) -> dict[str, Any]:
    path = task_path(task_id, data_dir)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def validate_grid(
    grid: Any,
    *,
    task_id: int,
    split: str,
    pair_index: int,
    side: str,
) -> tuple[int, int, Counter[int]]:
    context = f"task{task_id:03d} {split}[{pair_index}].{side}"
    if not isinstance(grid, list) or not grid:
        raise TaskValidationError(f"{context} must be a non-empty list of rows")
    if not all(isinstance(row, list) for row in grid):
        raise TaskValidationError(f"{context} rows must be lists")
    width = len(grid[0])
    if width == 0:
        raise TaskValidationError(f"{context} must have at least one column")
    colors: Counter[int] = Counter()
    for row_index, row in enumerate(grid):
        if len(row) != width:
            raise TaskValidationError(f"{context} row {row_index} is not rectangular")
        for color in row:
            if not isinstance(color, int) or color < 0 or color > 9:
                raise TaskValidationError(f"{context} has invalid color {color!r}")
            colors[color] += 1
    return len(grid), width, colors


def iter_pairs(task: dict[str, Any]) -> Iterable[tuple[str, int, dict[str, Any]]]:
    for split in SPLITS:
        for index, pair in enumerate(task.get(split, [])):
            yield split, index, pair


def validate_task(task: dict[str, Any], task_id: int) -> None:
    missing = [split for split in SPLITS if split not in task]
    if missing:
        raise TaskValidationError(f"task{task_id:03d} missing splits: {missing}")
    for split in SPLITS:
        if not isinstance(task[split], list):
            raise TaskValidationError(f"task{task_id:03d} {split} must be a list")
    for split, index, pair in iter_pairs(task):
        if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
            raise TaskValidationError(f"task{task_id:03d} {split}[{index}] is invalid")
        validate_grid(pair["input"], task_id=task_id, split=split, pair_index=index, side="input")
        validate_grid(pair["output"], task_id=task_id, split=split, pair_index=index, side="output")


def _shape_key(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def summarize_task(task_id: int | str, data_dir: str | Path = PATHS.data) -> dict[str, Any]:
    normalized = normalize_task_id(task_id)
    task = load_task(normalized, data_dir)
    validate_task(task, normalized)

    pair_counts = {split: len(task[split]) for split in SPLITS}
    input_shapes: Counter[str] = Counter()
    output_shapes: Counter[str] = Counter()
    colors: Counter[int] = Counter()
    size_changes = 0
    oversize_pairs = 0
    max_height = 0
    max_width = 0

    for split, index, pair in iter_pairs(task):
        input_height, input_width, input_colors = validate_grid(
            pair["input"], task_id=normalized, split=split, pair_index=index, side="input"
        )
        output_height, output_width, output_colors = validate_grid(
            pair["output"], task_id=normalized, split=split, pair_index=index, side="output"
        )
        input_shape = (input_height, input_width)
        output_shape = (output_height, output_width)
        input_shapes[_shape_key(input_shape)] += 1
        output_shapes[_shape_key(output_shape)] += 1
        colors.update(input_colors)
        colors.update(output_colors)
        max_height = max(max_height, input_height, output_height)
        max_width = max(max_width, input_width, output_width)
        if input_shape != output_shape:
            size_changes += 1
        if (
            input_height > GRID_HEIGHT
            or input_width > GRID_WIDTH
            or output_height > GRID_HEIGHT
            or output_width > GRID_WIDTH
        ):
            oversize_pairs += 1

    return {
        "task_id": normalized,
        "path": str(task_path(normalized, data_dir)),
        "pair_counts": pair_counts,
        "total_pairs": sum(pair_counts.values()),
        "input_shapes": dict(sorted(input_shapes.items())),
        "output_shapes": dict(sorted(output_shapes.items())),
        "colors": {str(k): v for k, v in sorted(colors.items())},
        "max_height": max_height,
        "max_width": max_width,
        "oversize_pairs": oversize_pairs,
        "size_change_pairs": size_changes,
    }


def summarize_dataset(
    data_dir: str | Path = PATHS.data,
    *,
    include_tasks: bool = False,
) -> dict[str, Any]:
    paths = iter_task_paths(data_dir)
    found_ids = [task_id_from_path(path) for path in paths]
    expected_ids = set(range(1, TASK_COUNT + 1))
    missing_ids = sorted(expected_ids.difference(found_ids))
    extra_ids = sorted(set(found_ids).difference(expected_ids))

    split_counts: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    invalid: list[dict[str, str]] = []
    task_summaries: list[dict[str, Any]] = []
    size_change_tasks = 0
    oversize_pairs = 0
    oversize_tasks = 0
    max_height = 0
    max_width = 0

    for task_id in found_ids:
        try:
            summary = summarize_task(task_id, data_dir)
        except Exception as exc:  # noqa: BLE001 - report all validation failures.
            invalid.append({"task_id": f"{task_id:03d}", "error": str(exc)})
            continue
        split_counts.update(summary["pair_counts"])
        colors.update(summary["colors"])
        size_change_tasks += int(summary["size_change_pairs"] > 0)
        oversize_pairs += summary["oversize_pairs"]
        oversize_tasks += int(summary["oversize_pairs"] > 0)
        max_height = max(max_height, summary["max_height"])
        max_width = max(max_width, summary["max_width"])
        if include_tasks:
            task_summaries.append(summary)

    result: dict[str, Any] = {
        "task_files": len(paths),
        "valid_tasks": len(paths) - len(invalid),
        "missing_task_ids": missing_ids,
        "extra_task_ids": extra_ids,
        "invalid_tasks": invalid,
        "pair_counts": dict(split_counts),
        "total_pairs": sum(split_counts.values()),
        "colors": dict(sorted(colors.items(), key=lambda item: int(item[0]))),
        "tasks_with_size_changes": size_change_tasks,
        "oversize_pairs": oversize_pairs,
        "tasks_with_oversize_pairs": oversize_tasks,
        "max_height": max_height,
        "max_width": max_width,
    }
    if include_tasks:
        result["tasks"] = task_summaries
    return result
