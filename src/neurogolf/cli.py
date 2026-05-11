from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PATHS
from .official import constraints, missing_runtime_dependencies
from .research_log import append_experiment
from .tasks import summarize_dataset, summarize_task


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def cmd_dataset_summary(args: argparse.Namespace) -> int:
    summary = summarize_dataset(args.data_dir, include_tasks=args.include_tasks)
    print(f"Task files: {summary['task_files']}")
    print(f"Valid tasks: {summary['valid_tasks']}")
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"Pair counts: {summary['pair_counts']}")
    print(f"Max grid: {summary['max_height']}x{summary['max_width']}")
    print(f"Tasks with size changes: {summary['tasks_with_size_changes']}")
    print(f"Oversize pairs ignored by official validation: {summary['oversize_pairs']}")
    if summary["missing_task_ids"]:
        print(f"Missing task ids: {summary['missing_task_ids']}")
    if summary["invalid_tasks"]:
        print(f"Invalid tasks: {summary['invalid_tasks']}")
    if args.write:
        _write_json(args.write, summary)
        print(f"Wrote {args.write}")
    return 1 if summary["missing_task_ids"] or summary["invalid_tasks"] else 0


def cmd_inspect_task(args: argparse.Namespace) -> int:
    summary = summarize_task(args.task_id, args.data_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_constraints(args: argparse.Namespace) -> int:
    print(json.dumps(constraints(), indent=2, sort_keys=True))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    missing = missing_runtime_dependencies()
    if missing:
        print("Missing runtime dependencies:")
        for package in missing:
            print(f"- {package}")
        print()
        print("Install them with: python -m pip install -r requirements.txt")
        return 1
    print("Runtime dependencies are available.")
    return 0


def cmd_log_event(args: argparse.Namespace) -> int:
    append_experiment(
        {
            "kind": args.kind,
            "task_id": args.task_id,
            "status": args.status,
            "notes": args.notes,
        },
        log_dir=args.log_dir,
    )
    print(f"Logged event to {Path(args.log_dir) / 'experiments.jsonl'}")
    return 0


def cmd_build_task001_baseline(args: argparse.Namespace) -> int:
    from .baselines import save_task001_self_tiling_model
    from .validation import validate_model

    output = save_task001_self_tiling_model(args.output)
    print(f"Wrote {output}")
    if args.validate:
        report_path = args.report or str(PATHS.reports / "task001_baseline_validation.json")
        result = validate_model(output, 1, data_dir=args.data_dir, report_path=report_path)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_validate_model(args: argparse.Namespace) -> int:
    from .validation import validate_model

    report_path = args.report
    if report_path is None:
        report_path = str(PATHS.reports / f"task{int(args.task_id):03d}_validation.json")
    result = validate_model(
        args.model,
        args.task_id,
        data_dir=args.data_dir,
        report_path=report_path,
        log_event=not args.no_log,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurogolf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset-summary")
    dataset.add_argument("--data-dir", default=str(PATHS.data))
    dataset.add_argument("--write")
    dataset.add_argument("--include-tasks", action="store_true")
    dataset.set_defaults(func=cmd_dataset_summary)

    inspect = subparsers.add_parser("inspect-task")
    inspect.add_argument("task_id")
    inspect.add_argument("--data-dir", default=str(PATHS.data))
    inspect.set_defaults(func=cmd_inspect_task)

    constraint_parser = subparsers.add_parser("constraints")
    constraint_parser.set_defaults(func=cmd_constraints)

    doctor = subparsers.add_parser("doctor")
    doctor.set_defaults(func=cmd_doctor)

    log_event = subparsers.add_parser("log-event")
    log_event.add_argument("--kind", default="manual")
    log_event.add_argument("--task-id")
    log_event.add_argument("--status", default="noted")
    log_event.add_argument("--notes", default="")
    log_event.add_argument("--log-dir", default=str(PATHS.experiments))
    log_event.set_defaults(func=cmd_log_event)

    build_task001 = subparsers.add_parser("build-task001-baseline")
    build_task001.add_argument("--output", default=str(PATHS.candidate_models / "task001.onnx"))
    build_task001.add_argument("--data-dir", default=str(PATHS.data))
    build_task001.add_argument("--report")
    build_task001.add_argument("--no-validate", dest="validate", action="store_false")
    build_task001.set_defaults(func=cmd_build_task001_baseline, validate=True)

    validate = subparsers.add_parser("validate-model")
    validate.add_argument("task_id")
    validate.add_argument("model")
    validate.add_argument("--data-dir", default=str(PATHS.data))
    validate.add_argument("--report")
    validate.add_argument("--no-log", action="store_true")
    validate.set_defaults(func=cmd_validate_model)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
