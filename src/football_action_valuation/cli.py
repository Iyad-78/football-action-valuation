from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from football_action_valuation.analysis.final import analyze_world_cup_final
from football_action_valuation.constants import (
    WORLD_CUP_2022_COMPETITION_ID,
    WORLD_CUP_2022_SEASON_ID,
    find_project_root,
)
from football_action_valuation.data.statsbomb import download_competition_events
from football_action_valuation.data.xg_dataset import prepare_xg_dataset
from football_action_valuation.evaluation.comparison import compare_xg_models
from football_action_valuation.models.xg import train_xg_model


def _print_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _paths(root: Path) -> dict[str, Path]:
    return {
        "raw": root / "data" / "raw" / "statsbomb",
        "processed": root / "data" / "processed" / "shots_world_cup_2022.csv",
        "models": root / "artifacts" / "models",
        "reports": root / "artifacts" / "reports",
    }


def _add_common_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--competition-id", type=int, default=WORLD_CUP_2022_COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=WORLD_CUP_2022_SEASON_ID)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fav",
        description="Football Action Valuation — reproducible ML pipeline",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root (auto-detected by default)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    data_parser = commands.add_parser("data", help="Download or prepare data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    download = data_commands.add_parser("download", help="Download StatsBomb event files")
    _add_common_dataset_arguments(download)
    download.add_argument("--limit", type=int, default=None)
    download.add_argument("--workers", type=int, default=4)
    download.add_argument("--force", action="store_true")

    prepare = data_commands.add_parser("prepare-xg", help="Build the shot-level dataset")
    _add_common_dataset_arguments(prepare)
    prepare.add_argument("--include-penalties", action="store_true")

    train = commands.add_parser("train", help="Train a model")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    train_xg = train_commands.add_parser("xg", help="Train an xG baseline")
    train_xg.add_argument("--model", choices=["logistic", "lightgbm"], default="logistic")
    train_xg.add_argument("--random-state", type=int, default=42)

    analyze = commands.add_parser("analyze", help="Generate match reports")
    analyze_commands = analyze.add_subparsers(dest="analyze_command", required=True)
    final = analyze_commands.add_parser("final", help="Analyze the 2022 World Cup final")
    final.add_argument("--model", choices=["logistic", "lightgbm"], default="logistic")

    compare = commands.add_parser("compare", help="Compare trained models")
    compare_commands = compare.add_subparsers(dest="compare_command", required=True)
    compare_xg = compare_commands.add_parser("xg", help="Compare xG probability metrics")
    compare_xg.add_argument(
        "--models",
        nargs="+",
        choices=["logistic", "lightgbm"],
        default=["logistic", "lightgbm"],
    )

    baseline = commands.add_parser(
        "run-baseline", help="Run download, preparation, training and final analysis"
    )
    baseline.add_argument("--model", choices=["logistic", "lightgbm"], default="logistic")
    baseline.add_argument("--workers", type=int, default=4)

    actions = commands.add_parser("run-actions", help="Train action models and rate final players")
    actions.add_argument(
        "--models",
        nargs="+",
        choices=["logistic", "lightgbm", "mlp", "gru", "transformer"],
        default=["logistic", "lightgbm"],
    )
    actions.add_argument("--window", type=int, default=6)
    actions.add_argument("--horizon", type=int, default=10)
    actions.add_argument("--epochs", type=int, default=6)
    actions.add_argument("--seed", type=int, default=42)
    actions.add_argument("--output", type=Path, default=None)
    inference = commands.add_parser(
        "predict-actions", help="Reload a saved model for batch inference"
    )
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--model-dir", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    root = (arguments.project_root or find_project_root()).resolve()
    paths = _paths(root)

    if arguments.command == "run-actions":
        from football_action_valuation.analysis.actions import report_actions
        from football_action_valuation.data.actions import prepare_actions
        from football_action_valuation.evaluation.metrics import save_json
        from football_action_valuation.models.actions import train_actions

        output = arguments.output or root / "outputs" / "v0.2"
        processed = root / "data" / "processed" / "actions"
        model_dir = root / "artifacts" / "models" / "actions"
        dataset = prepare_actions(paths["raw"], processed, horizon=arguments.horizon)
        print(json.dumps(dataset, indent=2), flush=True)
        selected = train_actions(
            processed / "actions.csv",
            model_dir,
            arguments.models,
            window=arguments.window,
            epochs=arguments.epochs,
            seed=arguments.seed,
        )
        result = report_actions(model_dir, processed / "players.csv", output)
        save_json(dict(dataset=dataset, selection=selected, report=result), output / "run.json")
    elif arguments.command == "predict-actions":
        import pandas as pd

        from football_action_valuation.models.actions import predict_saved

        frame = pd.read_csv(arguments.input)
        frame[["p_score", "p_concede"]] = predict_saved(frame, arguments.model_dir)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(arguments.output, index=False)
        result = dict(rows=len(frame), output=str(arguments.output))
    elif arguments.command == "data" and arguments.data_command == "download":
        result = download_competition_events(
            paths["raw"],
            competition_id=arguments.competition_id,
            season_id=arguments.season_id,
            limit=arguments.limit,
            workers=arguments.workers,
            force=arguments.force,
        )
    elif arguments.command == "data" and arguments.data_command == "prepare-xg":
        result = prepare_xg_dataset(
            paths["raw"],
            paths["processed"],
            competition_id=arguments.competition_id,
            season_id=arguments.season_id,
            include_penalties=arguments.include_penalties,
        )
    elif arguments.command == "train" and arguments.train_command == "xg":
        result = train_xg_model(
            paths["processed"],
            paths["models"],
            model_name=arguments.model,
            random_state=arguments.random_state,
        )
    elif arguments.command == "analyze" and arguments.analyze_command == "final":
        result = analyze_world_cup_final(
            paths["models"] / f"xg_{arguments.model}_predictions.csv",
            paths["reports"],
            model_name=arguments.model,
        )
    elif arguments.command == "compare" and arguments.compare_command == "xg":
        result = compare_xg_models(
            paths["models"],
            root / "artifacts" / "evaluation",
            model_names=arguments.models,
        )
    elif arguments.command == "run-baseline":
        download_result = download_competition_events(paths["raw"], workers=arguments.workers)
        dataset_result = prepare_xg_dataset(paths["raw"], paths["processed"])
        model_result = train_xg_model(
            paths["processed"], paths["models"], model_name=arguments.model
        )
        report_result = analyze_world_cup_final(
            paths["models"] / f"xg_{arguments.model}_predictions.csv",
            paths["reports"],
            model_name=arguments.model,
        )
        result = {
            "download": download_result,
            "dataset": dataset_result,
            "model": model_result,
            "final_report": report_result,
        }
    else:  # pragma: no cover - argparse prevents this branch
        raise RuntimeError("Unsupported command")

    _print_result(result)
