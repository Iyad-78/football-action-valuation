"""Train, compare and reload two-output action-state probability models."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from football_action_valuation.evaluation.metrics import probability_metrics, save_json
from football_action_valuation.features.actions import TYPE_MAP, encode_sequences, tabular_view


def neural_predict(model, numeric, categorical, lengths, batch_size=512):
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for first in range(0, len(numeric), batch_size):
            sl = slice(first, first + batch_size)
            logits = model(
                torch.from_numpy(numeric[sl]),
                torch.from_numpy(categorical[sl]),
                torch.from_numpy(lengths[sl]),
            )
            chunks.append(logits.sigmoid().numpy())
    return np.concatenate(chunks)


def fit_neural(kind, arrays, targets, train_ids, valid_ids, epochs, seed):
    import torch
    from torch import nn

    from football_action_valuation.models.action_networks import ActionNetwork

    numeric, categorical, lengths = arrays
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ActionNetwork(kind, numeric.shape[-1], len(TYPE_MAP) + 1, numeric.shape[1])
    # Bias initialization avoids starting rare-event probabilities at 50%.
    prior = targets[train_ids].mean(axis=0).clip(1e-5, 1 - 1e-5)
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor(np.log(prior / (1 - prior))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, patience, history = float("inf"), 0, []
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_ids)
        train_loss = 0.0
        for first in range(0, len(order), 256):
            ids = order[first : first + 256]
            logits = model(
                torch.from_numpy(numeric[ids]),
                torch.from_numpy(categorical[ids]),
                torch.from_numpy(lengths[ids]),
            )
            loss = loss_fn(logits, torch.from_numpy(targets[ids]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(ids)
        p = neural_predict(model, numeric[valid_ids], categorical[valid_ids], lengths[valid_ids])
        p = p.clip(1e-7, 1 - 1e-7)
        y = targets[valid_ids]
        val_loss = float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())
        history.append(
            dict(epoch=epoch + 1, train_loss=train_loss / len(order), validation_loss=val_loss)
        )
        print(f"{kind}: epoch {epoch + 1}, validation loss={val_loss:.5f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, patience = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 2:
                break
    model.load_state_dict(best_state)
    return model, history


def predict_saved(frame: pd.DataFrame, model_dir: Path) -> np.ndarray:
    """Batch inference from a saved artifact; takes ordered action history."""
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    codec = joblib.load(model_dir / "codec.joblib")
    numeric, categorical, lengths, _ = encode_sequences(frame, codec["window"], codec)
    kind = metadata["model"]
    if kind in {"logistic", "lightgbm"}:
        estimators = joblib.load(model_dir / "model.joblib")
        x = tabular_view(numeric, categorical, lengths)
        with threadpool_limits(limits=2):
            return np.column_stack([est.predict_proba(x)[:, 1] for est in estimators])
    import torch

    from football_action_valuation.models.action_networks import ActionNetwork

    torch.set_num_threads(2)
    model = ActionNetwork(kind, numeric.shape[-1], len(TYPE_MAP) + 1, codec["window"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    return neural_predict(model, numeric, categorical, lengths)


def train_actions(
    dataset: Path,
    artifact_dir: Path,
    model_names: list[str],
    window: int = 6,
    epochs: int = 6,
    seed: int = 42,
) -> dict:
    if epochs < 1 or not model_names:
        raise ValueError("At least one model and one epoch are required")
    frame = pd.read_csv(dataset)
    numeric, categorical, lengths, codec = encode_sequences(frame, window)
    labels = frame[["scores", "concedes"]].to_numpy(dtype=np.float32)
    train_ids = np.flatnonzero(frame.split.eq("train") & frame.label_valid.eq(1))
    valid_ids = np.flatnonzero(frame.split.eq("validation") & frame.label_valid.eq(1))
    if not len(valid_ids) or not len(train_ids):
        raise ValueError("Both training and validation matches are required")
    if any(len(np.unique(labels[train_ids, j])) != 2 for j in (0, 1)):
        raise ValueError("Each training target needs both classes")
    rates = labels[train_ids].mean(axis=0)
    penalties = frame.loc[frame.split.eq("train") & frame.action.eq("Shot") & frame.penalty.eq(1)]
    penalty_rate = float((penalties.reward.eq(1).sum() + 1) / (len(penalties) + 2))
    records = []
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    for kind in model_names:
        start = time.perf_counter()
        destination = artifact_dir / kind
        destination.mkdir(parents=True, exist_ok=True)
        history = []
        parameters = None
        if kind in {"logistic", "lightgbm"}:
            x = tabular_view(numeric, categorical, lengths)
            estimators = []
            for target in range(2):
                if kind == "logistic":
                    estimator = LogisticRegression(C=0.1, max_iter=500, random_state=seed)
                else:
                    from lightgbm import LGBMClassifier

                    estimator = LGBMClassifier(
                        n_estimators=100,
                        max_depth=4,
                        num_leaves=15,
                        min_child_samples=100,
                        learning_rate=0.04,
                        reg_lambda=5,
                        n_jobs=2,
                        verbosity=-1,
                        random_state=seed,
                    )
                with threadpool_limits(limits=2):
                    estimator.fit(x[train_ids], labels[train_ids, target])
                estimators.append(estimator)
            with threadpool_limits(limits=2):
                predictions = np.column_stack([e.predict_proba(x)[:, 1] for e in estimators])
            joblib.dump(estimators, destination / "model.joblib")
            del x
        else:
            import torch

            model, history = fit_neural(
                kind, (numeric, categorical, lengths), labels, train_ids, valid_ids, epochs, seed
            )
            parameters = sum(p.numel() for p in model.parameters())
            predictions = neural_predict(model, numeric, categorical, lengths)
            torch.save(model.state_dict(), destination / "model.pt")
        duration = time.perf_counter() - start
        joblib.dump(codec, destination / "codec.joblib")
        save_json(
            dict(
                model=kind,
                window=window,
                seed=seed,
                epochs_requested=epochs,
                parameters=parameters,
                runtime_seconds=duration,
                dataset_sha256=digest,
                train_rows=len(train_ids),
                validation_rows=len(valid_ids),
                train_priors=rates.tolist(),
                penalty_conversion=penalty_rate,
                training_penalties=len(penalties),
                history=history,
            ),
            destination / "metadata.json",
        )
        output = frame.copy()
        output["p_score"], output["p_concede"] = predictions.T
        output.to_csv(destination / "predictions.csv", index=False)
        for split in ("train", "validation", "test", "showcase"):
            ids = np.flatnonzero(frame.split.eq(split) & frame.label_valid.eq(1))
            for j, target in enumerate(("scores", "concedes")):
                measures = probability_metrics(labels[ids, j], predictions[ids, j])
                records.append(dict(model=kind, split=split, target=target, **measures))
        print(f"Saved {kind} ({duration:.1f}s)", flush=True)
    for split in ("train", "validation", "test", "showcase"):
        ids = np.flatnonzero(frame.split.eq(split) & frame.label_valid.eq(1))
        for j, target in enumerate(("scores", "concedes")):
            records.append(
                dict(
                    model="constant",
                    split=split,
                    target=target,
                    **probability_metrics(labels[ids, j], np.full(len(ids), rates[j])),
                )
            )
    comparison = pd.DataFrame(records)
    comparison.to_csv(artifact_dir / "comparison.csv", index=False)
    selection = (
        comparison.loc[comparison.split.eq("validation") & comparison.model.ne("constant")]
        .groupby("model")
        .brier_score.mean()
    )
    best = str(selection.idxmin())
    summary = dict(
        selected_model=best,
        selection="Mean validation Brier, both targets",
        models=model_names,
        window=window,
        seed=seed,
        dataset_sha256=digest,
    )
    save_json(summary, artifact_dir / "selection.json")
    return summary
