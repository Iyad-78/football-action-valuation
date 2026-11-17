"""Observable action contributions and an explicitly experimental rating scale."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from football_action_valuation.evaluation.metrics import save_json


def value_actions(
    frame: pd.DataFrame, priors: list[float], penalty_conversion: float | None = None
) -> pd.DataFrame:
    result = frame.copy()
    offense, defense, transitions = [], [], []
    previous = None
    for row in frame.to_dict("records"):
        same_segment = previous is not None and (
            row["match_id"],
            row["period"],
            row["segment"],
        ) == (previous["match_id"], previous["period"], previous["segment"])
        if same_segment:
            before_for, before_against = previous["p_score"], previous["p_concede"]
            if row["team_id"] != previous["team_id"]:
                before_for, before_against = before_against, before_for
        else:
            before_for, before_against = priors
        after_for, after_against = row["p_score"], row["p_concede"]
        if (
            penalty_conversion is not None
            and row.get("action") == "Foul Committed"
            and row.get("penalty") == 1
        ):
            # Known restart: empirical training-only penalty conversion, Laplace-smoothed.
            after_for, after_against = 0.0, penalty_conversion
        if row["reward"]:
            after_for, after_against = float(row["reward"] > 0), float(row["reward"] < 0)
        offense.append(after_for - before_for)
        defense.append(before_against - after_against)
        transitions.append((before_for, before_against, after_for, after_against))
        previous = {**row, "p_score": after_for, "p_concede": after_against}
    result["offensive_value"] = offense
    result["defensive_value"] = defense
    result["action_value"] = result.offensive_value + result.defensive_value
    result[["before_score", "before_concede", "after_score", "after_concede"]] = transitions
    return result


def role(position: str) -> str:
    if position == "Goalkeeper":
        return "GK"
    if "Back" in position:
        return "DEF"
    if "Midfield" in position:
        return "MID"
    if "Wing" in position or "Forward" in position or "Striker" in position:
        return "ATT"
    return "UNKNOWN"


def player_ratings(actions: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    contributions = actions.groupby(["match_id", "player_id"], as_index=False).agg(
        actions=("event_id", "count"),
        offensive_value=("offensive_value", "sum"),
        defensive_value=("defensive_value", "sum"),
        total_value=("action_value", "sum"),
    )
    result = roster.merge(contributions, on=["match_id", "player_id"], how="left")
    columns = ["actions", "offensive_value", "defensive_value", "total_value"]
    result[columns] = result[columns].fillna(0)
    splits = actions.groupby("match_id").split.first()
    result["split"] = result.match_id.map(splits)
    result["role"] = result.position.map(role)
    # Regularization limits explosive per-90 values for short appearances.
    result["regularized_per90"] = 90 * result.total_value / (result.minutes + 60)
    result["experimental_rating"] = np.nan
    result["reference_count"] = 0
    result["rating_status"] = "insufficient_minutes"
    result.loc[result.role.eq("GK"), "rating_status"] = "goalkeeper_not_rated"
    result.loc[result.role.eq("UNKNOWN"), "rating_status"] = "unknown_role"
    reference = result.loc[
        result.split.eq("validation") & result.minutes.ge(30) & ~result.role.isin(["GK", "UNKNOWN"])
    ]
    for index, player in result.iterrows():
        if player.minutes < 30 or player.role in {"GK", "UNKNOWN"}:
            continue
        peers = reference.loc[reference.role.eq(player.role), "regularized_per90"]
        if len(peers) < 10:
            result.loc[index, "rating_status"] = "insufficient_reference"
            continue
        percentile = (
            float((peers < player.regularized_per90).sum())
            + 0.5 * float((peers == player.regularized_per90).sum())
        ) / len(peers)
        result.loc[index, "experimental_rating"] = round(1 + 9 * percentile, 1)
        result.loc[index, "reference_count"] = len(peers)
        result.loc[index, "rating_status"] = "experimental"
    return result


def report_actions(action_dir: Path, roster_path: Path, output_dir: Path) -> dict:
    selected = json.loads((action_dir / "selection.json").read_text(encoding="utf-8"))
    kind = selected["selected_model"]
    model_dir = action_dir / kind
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(model_dir / "predictions.csv")
    valued = value_actions(frame, metadata["train_priors"], metadata["penalty_conversion"])
    ratings = player_ratings(valued, pd.read_csv(roster_path))
    final = ratings.loc[ratings.split.eq("showcase")].sort_values(
        ["experimental_rating", "total_value"], ascending=False, na_position="last"
    )
    comparison = pd.read_csv(action_dir / "comparison.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    valued.to_csv(output_dir / "action_values.csv", index=False)
    ratings.to_csv(output_dir / "player_profiles.csv", index=False)
    final.to_csv(output_dir / "final_ratings.csv", index=False)
    final_actions = valued.loc[valued.split.eq("showcase")]
    evidence = pd.concat(
        [
            group.sort_values("action_value", ascending=False).head(3)
            for _, group in final_actions.groupby("player_id")
        ]
        + [
            group.sort_values("action_value").head(3)
            for _, group in final_actions.groupby("player_id")
        ]
    ).drop_duplicates("event_id")
    evidence.to_csv(output_dir / "final_key_actions.csv", index=False)
    summary = dict(
        model=kind,
        actions=len(valued),
        final_players=len(final),
        rated_players=int(final.experimental_rating.notna().sum()),
        reference="Validation player-matches, same broad role, >=30 elapsed minutes",
        formula="1 + 9 * percentile(90 * total_value / (minutes + 60))",
        limitations=[
            "event-only",
            "no causal interpretation",
            "one tournament",
            "no goalkeeper rating",
            "experimental scale",
            "penalties included",
        ],
    )
    save_json(summary, output_dir / "summary.json")
    cols = [
        "team",
        "player",
        "role",
        "minutes",
        "actions",
        "total_value",
        "experimental_rating",
        "rating_status",
    ]
    table = final[cols].copy()
    table[["minutes", "total_value"]] = table[["minutes", "total_value"]].round(3)
    test = comparison.loc[
        comparison.split.eq("test"),
        ["model", "target", "samples", "brier_score", "log_loss", "average_precision"],
    ].round(5)
    report = [
        "# Football Action Valuation — v0.2",
        "",
        f"Modèle retenu sur le Brier moyen de validation : **{kind}**.",
        "",
        "## Notes expérimentales — finale 2022",
        "",
        "Contributions des événements observés ; échelle relative aux joueurs de validation "
        "du même grand rôle. Ce ne sont pas des notes expertes ni une vérité sur la performance.",
        "Les penalties pendant le match sont inclus ; la séance de tirs au but est exclue.",
        "Un penalty concédé utilise un risque empirique calculé sur les penalties de train ; "
        "cette convention explicite pénalise le fautif et ajuste le crédit du tireur.",
        "Le temps de jeu inclut les arrêts de jeu et les prolongations. Gardien non noté ; "
        "moins de 30 minutes : contribution affichée mais pas de note.",
        "",
        table.to_markdown(index=False),
        "",
        "## Évaluation sur les matchs de test",
        "",
        test.to_markdown(index=False),
        "",
        "## Exemples de contributions sur la finale",
        "",
        "Plus fortes contributions positives :",
        "",
        final_actions.nlargest(6, "action_value")[
            ["player", "minute", "action", "penalty", "action_value"]
        ]
        .round(4)
        .to_markdown(index=False),
        "",
        "Plus fortes contributions négatives :",
        "",
        final_actions.nsmallest(6, "action_value")[
            ["player", "minute", "action", "penalty", "action_value"]
        ]
        .round(4)
        .to_markdown(index=False),
        "",
        "## Comment expliquer une contribution",
        "",
        "Valeur offensive = P(marquer après) − P(marquer avant). Valeur défensive = "
        "P(encaisser avant) − P(encaisser après). Les perspectives sont inversées quand "
        "l'équipe qui agit change. Un but est traité comme une récompense observée ; "
        "l'état est réinitialisé après le but et au changement de période.",
        "",
        "Les probabilités portent sur le premier but dans les dix prochaines actions retenues. "
        "Les fins de période sans fenêtre complète sont exclues de la perte d'entraînement. "
        "Cette implémentation est inspirée de VAEP, "
        "avec un vocabulaire et des conventions propres.",
        "",
        "## Limites à présenter",
        "",
        "- Seulement 64 matchs : les objectifs rares et le test sont de petite taille.",
        "- Séquences voisines corrélées : les lignes ne sont pas des essais indépendants.",
        "- Pas de preuve causale, pas de tracking, pas de contrôle exhaustif des biais.",
        "- Note relative : pondération par minutes et rôles sont des choix de conception.",
        "- La référence validation sert aussi à sélectionner le modèle ; pas de vérité terrain "
        "pour les notes ni d'intervalle de confiance prétendument calibré.",
        "- Les contributions défensives mesurent une baisse de risque prédite, "
        "pas toute la défense.",
        "- Tests et entraînements CPU vérifiés ; qualité sportive à valider avec des analystes.",
    ]
    (output_dir / "RAPPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary
