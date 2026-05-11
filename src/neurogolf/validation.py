from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
import onnxruntime

from .config import PATHS
from .official import competition_score, load_official_utils
from .research_log import append_experiment
from .tasks import normalize_task_id


def validate_model(
    model_path: str | Path,
    task_id: int | str,
    *,
    data_dir: str | Path = PATHS.data,
    report_path: str | Path | None = None,
    log_event: bool = True,
) -> dict[str, Any]:
    task_num = normalize_task_id(task_id)
    model_file = Path(model_path)
    utils = load_official_utils(data_dir)
    examples = utils.load_examples(task_num)

    if not utils.check_network(model_file):
        result = {
            "task_id": task_num,
            "model_path": str(model_file),
            "status": "invalid_file",
        }
        _finish_validation(result, report_path=report_path, log_event=log_event)
        return result

    sanitized = onnx.load(model_file)
    for node in sanitized.graph.node:
        node.name = node.output[0]
        if "kernel_time" in node.name:
            raise ValueError(f"Node name contains disallowed profiler token: {node.name}")

    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(PATHS.root / ".cache" / f"profile_task{task_num:03d}")
    session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)

    arc_agi_right, arc_agi_wrong, _ = utils.verify_subset(
        session, examples["train"] + examples["test"]
    )
    arc_gen_right, arc_gen_wrong, _ = utils.verify_subset(session, examples["arc-gen"])
    trace_path = session.end_profiling()
    memory, params = utils.score_network(sanitized, trace_path)

    status = "passed" if arc_agi_wrong + arc_gen_wrong == 0 else "failed"
    result = {
        "task_id": task_num,
        "model_path": str(model_file),
        "status": status,
        "arc_agi": {
            "pass": arc_agi_right,
            "fail": arc_agi_wrong,
        },
        "arc_gen": {
            "pass": arc_gen_right,
            "fail": arc_gen_wrong,
        },
        "memory_bytes": memory,
        "params": params,
        "estimated_score": competition_score(memory, params)
        if memory is not None and params is not None
        else None,
        "profile_path": trace_path,
    }
    _finish_validation(result, report_path=report_path, log_event=log_event)
    return result


def _finish_validation(
    result: dict[str, Any],
    *,
    report_path: str | Path | None,
    log_event: bool,
) -> None:
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
    if log_event:
        append_experiment(
            {
                "kind": "model_validation",
                "task_id": f"{result['task_id']:03d}",
                "status": result["status"],
                "model_path": result["model_path"],
                "params": result.get("params"),
                "memory_bytes": result.get("memory_bytes"),
                "estimated_score": result.get("estimated_score"),
                "arc_agi_fail": result.get("arc_agi", {}).get("fail"),
                "arc_gen_fail": result.get("arc_gen", {}).get("fail"),
            }
        )

