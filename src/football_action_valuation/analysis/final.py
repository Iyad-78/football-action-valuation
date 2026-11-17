from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "football-action-valuation-matplotlib")
)

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Rectangle  # noqa: E402


def _draw_attacking_half(axis: Any) -> None:
    axis.add_patch(Rectangle((60, 0), 60, 80, fill=False, linewidth=1.5, color="#334155"))
    axis.add_patch(Rectangle((102, 18), 18, 44, fill=False, linewidth=1.2, color="#334155"))
    axis.add_patch(Rectangle((114, 30), 6, 20, fill=False, linewidth=1.2, color="#334155"))
    axis.add_patch(Arc((108, 40), 20, 20, theta1=126, theta2=234, color="#334155"))
    axis.scatter([108], [40], s=12, color="#334155")
    axis.set(xlim=(60, 121), ylim=(81, -1), aspect="equal")
    axis.axis("off")


def _save_shot_map(shots: pd.DataFrame, path: Path) -> None:
    teams = list(shots["team"].dropna().unique())
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    color_by_team = {team: colors[index % len(colors)] for index, team in enumerate(teams)}

    figure, axis = plt.subplots(figsize=(10, 7))
    figure.patch.set_facecolor("#f8fafc")
    axis.set_facecolor("#f8fafc")
    _draw_attacking_half(axis)

    for team in teams:
        subset = shots.loc[shots["team"] == team]
        axis.scatter(
            subset["x"],
            subset["y"],
            s=50 + 700 * subset["predicted_xg"],
            alpha=0.68,
            edgecolors="white",
            linewidths=1,
            color=color_by_team[team],
            label=team,
        )
        goals = subset.loc[subset["goal"] == 1]
        axis.scatter(
            goals["x"],
            goals["y"],
            marker="*",
            s=260,
            color="#facc15",
            edgecolors="#0f172a",
            linewidths=1,
            zorder=5,
        )

    axis.set_title(
        "2022 World Cup final — non-penalty xG\nMarker size = model xG; star = goal",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
    )
    axis.legend(loc="lower left", frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_world_cup_final(
    predictions_path: Path,
    report_dir: Path,
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    final = predictions.loc[predictions["split"] == "showcase"].copy()
    if final.empty:
        raise ValueError(
            "No showcase match found. Download and prepare the complete 2022 World Cup dataset."
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    model_slug = model_name or predictions_path.stem.removeprefix("xg_").removesuffix(
        "_predictions"
    )
    team_summary = (
        final.groupby("team", as_index=False)
        .agg(
            shots=("event_id", "count"),
            non_penalty_goals=("goal", "sum"),
            model_xg=("predicted_xg", "sum"),
            statsbomb_xg=("statsbomb_xg", "sum"),
        )
        .sort_values("model_xg", ascending=False)
    )
    player_summary = (
        final.groupby(["team", "player"], as_index=False)
        .agg(
            shots=("event_id", "count"),
            non_penalty_goals=("goal", "sum"),
            model_xg=("predicted_xg", "sum"),
            statsbomb_xg=("statsbomb_xg", "sum"),
        )
        .sort_values(["model_xg", "shots"], ascending=False)
    )
    team_summary[["model_xg", "statsbomb_xg"]] = team_summary[
        ["model_xg", "statsbomb_xg"]
    ].round(3)
    player_summary[["model_xg", "statsbomb_xg"]] = player_summary[
        ["model_xg", "statsbomb_xg"]
    ].round(3)

    team_path = report_dir / f"world_cup_2022_final_teams_{model_slug}.csv"
    player_path = report_dir / f"world_cup_2022_final_players_{model_slug}.csv"
    shot_map_path = report_dir / f"world_cup_2022_final_shot_map_{model_slug}.png"
    report_path = report_dir / f"world_cup_2022_final_report_{model_slug}.md"
    team_summary.to_csv(team_path, index=False)
    player_summary.to_csv(player_path, index=False)
    _save_shot_map(final, shot_map_path)

    lines = [
        f"# 2022 World Cup final — {model_slug} xG baseline",
        "",
        "This first report evaluates non-penalty shots only. It is an xG baseline,",
        "not yet the final player-action rating system.",
        "",
        "## Team summary",
        "",
        team_summary.to_markdown(index=False),
        "",
        "## Most threatening shooters",
        "",
        player_summary.head(10).to_markdown(index=False),
        "",
        "## Methodological note",
        "",
        "The final is isolated from training and model selection. StatsBomb xG is",
        "retained only as an external benchmark and is never used as a model feature.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "match_id": int(final["match_id"].iloc[0]),
        "model": model_slug,
        "shots": int(len(final)),
        "team_summary": str(team_path),
        "player_summary": str(player_path),
        "shot_map": str(shot_map_path),
        "report": str(report_path),
    }
