# Football Action Valuation

## v0.2 — contributions des actions et notes expérimentales

La nouvelle tranche du projet va des événements bruts aux contributions par
joueur. Elle compare **régression logistique, LightGBM, MLP, GRU et Transformer**
sur deux objectifs supervisés : marquer ou encaisser le prochain but dans les
10 actions suivantes. Les cinq modèles reçoivent les mêmes six dernières actions.

Installation CPU (Python **3.11+**, 3.12 recommandé) :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,boosting]"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fav data download
fav run-actions --models logistic lightgbm mlp gru transformer --window 6 --epochs 6
```

Si l'environnement du projet est déjà installé, exécuter simplement la dernière
commande. Une version tabulaire fonctionne avec `fav run-actions` (sans PyTorch).
Sur Colab, utiliser le PyTorch fourni, puis `pip install -e ".[dev,boosting]"`.
Le programme entraîne actuellement sur CPU, même si Colab dispose d'un GPU.

Résultats locaux :

- `outputs/v0.2/RAPPORT.md` : comparaison et notes de la finale ;
- `outputs/v0.2/final_key_actions.csv` : contributions positives/négatives par joueur ;
- `outputs/v0.2/final_ratings.csv` : notes, minutes, rôles et statut ;
- `outputs/v0.2/action_values.csv` : toutes les contributions ;
- `artifacts/models/actions/comparison.csv` : métriques par modèle, split et cible ;
- `artifacts/models/actions/selection.json` : modèle choisi sur validation ;
- `artifacts/models/actions/<modèle>/` : poids, prétraitement, historique et paramètres.

Réutiliser un modèle sans réentraînement :

```powershell
fav predict-actions --input data/processed/actions/actions.csv --model-dir artifacts/models/actions/gru --output outputs/gru_predictions.csv
```

Cette commande attend les actions **préparées et ordonnées**, avec leur historique.
Ne pas lui passer directement un JSON StatsBomb. Le format est celui du dataset
produit par `run-actions`.

Les notes sont des percentiles de contributions, ramenés sur 10, par rapport aux
joueurs de validation du même grand rôle. Elles sont **expérimentales**, sans
vérité terrain experte. Les gardiens et apparitions de moins de 30 minutes ne sont
pas notés. Les minutes incluent le temps additionnel et les prolongations.
Les penalties en match sont inclus ; les tirs au but sont exclus. Une convention
empirique de risque après penalty concédé est documentée dans la méthode.

Lire [la méthode détaillée](docs/action_methodology.md) et
[la fiche de préparation à l'entretien](docs/ENTRETIEN.md).
Une [synthèse des résultats vérifiés](docs/RESULTATS_V02.md) est conservée avec
le code : 142 550 actions, cinq modèles entraînés et 20 tests réussis.
API HTTP, dashboard, tracking, préentraînement auto-supervisé et validation externe
ne sont pas implémentés. Dockerfile/CI sont fournis ; leur exécution distante
n'est pas une validation effectuée sur cette machine.

---

## Socle xG v0.1

An explainable machine-learning project for valuing football actions and, in
later milestones, producing context-aware player ratings from event sequences.

The repository is built as a progressive ML study rather than a single opaque
model. This section documents the original shot-level baseline. The completed
action-level and neural comparison is documented in the v0.2 section above.

> **Original milestone — v0.1:** the xG baseline is implemented. It downloads
> StatsBomb Open Data, builds a leakage-safe shot dataset, creates chronological
> match-level splits, trains a probabilistic classifier and generates an
> isolated report for the 2022 World Cup final.

## Research question

Can the context surrounding an action be learned well enough to estimate how it
changes a team's probability of scoring or conceding, and can those changes be
aggregated into transparent player ratings?

The final target is not to imitate a proprietary rating. It is to expose the
reason for every contribution:

```text
event sequence → game-state probabilities → action value → player profile
```

## Why this is an ML and deep-learning project

The project compares models under one experimental protocol:

| Stage | Model | Purpose |
|---|---|---|
| Implemented | Logistic regression | Interpretable probabilistic xG baseline |
| Implemented | LightGBM (optional) | Strong non-linear tabular baseline |
| Implemented (action pipeline) | MLP | Neural baseline on event history |
| Implemented (action pipeline) | GRU | Compact sequential model |
| Implemented (action pipeline) | Transformer | Learned event embeddings and temporal attention |

Deep learning is not assumed to be better. It must improve probability quality,
calibration or context sensitivity enough to justify its additional cost.

## Original xG pipeline

```text
StatsBomb Open Data (JSON)
          ↓ streamed one match at a time
Shot-level feature table
          ↓ split chronologically by complete match
Logistic regression / LightGBM
          ↓
Brier score, log-loss, ROC-AUC, PR-AUC, calibration
          ↓
2022 World Cup final report (never seen during training)
```

Safeguards already implemented:

- the final is isolated as a `showcase` split;
- no event from one match can appear in multiple splits;
- post-shot information and StatsBomb's xG are excluded from model inputs;
- penalties are excluded by default and clearly labelled in reports;
- raw data and generated model artifacts are not committed to Git;
- unit tests use synthetic fixtures and require no external data.

## Quick start

Python 3.11 or newer is required. On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,boosting]"
```

Run the full baseline:

```powershell
fav run-baseline --model logistic
```

Or execute each reproducible stage independently:

```powershell
fav data download --competition-id 43 --season-id 106
fav data prepare-xg --competition-id 43 --season-id 106
fav train xg --model logistic
fav analyze final --model logistic
fav compare xg
```

LightGBM is available after installing the `boosting` extra:

```powershell
fav train xg --model lightgbm
fav analyze final --model lightgbm
```

For a fast pipeline check, download a small sample first:

```powershell
fav data download --limit 8
fav data prepare-xg
fav train xg --model logistic
```

The final-report command requires the complete competition because the final may
not be part of a limited sample.

## Verified v0.1 results

The pipeline was executed end-to-end on all 64 matches of the 2022 World Cup:

- 1,430 non-penalty shots and 152 goals;
- 953 training, 222 validation and 228 test shots;
- 27 final-match shots isolated as showcase data.

Held-out results are deliberately reported without claiming that the first
models are production-ready:

| Model | Split | Brier ↓ | Log-loss ↓ | ROC-AUC ↑ | PR-AUC ↑ |
|---|---|---:|---:|---:|---:|
| Logistic regression | Validation | 0.0973 | 0.3225 | 0.7409 | 0.2967 |
| LightGBM | Validation | 0.1035 | 0.3761 | 0.7019 | 0.2443 |
| Logistic regression | Test | 0.0873 | 0.3169 | 0.7121 | 0.4015 |
| LightGBM | Test | 0.0895 | 0.3375 | 0.7466 | 0.3561 |
| StatsBomb external benchmark | Test | 0.0772 | 0.2814 | 0.7896 | 0.5454 |

In the original xG experiment, LightGBM reached a train ROC-AUC of 0.9998 but did
not improve validation or test probability scores. These historical results
concern shots, not the separate action-level task introduced in v0.2. Extending
the training corpus and checking probability calibration remain future work.

## Generated outputs

```text
data/processed/shots_world_cup_2022.csv
artifacts/models/xg_logistic.joblib
artifacts/models/xg_logistic_metrics.json
artifacts/models/xg_logistic_calibration.png
artifacts/models/xg_logistic_predictions.csv
artifacts/reports/world_cup_2022_final_report_logistic.md
artifacts/reports/world_cup_2022_final_shot_map_logistic.png
```

These paths are ignored by Git because they are reproducible. Selected figures
and summarized metrics can later be copied into a versioned portfolio report.

## Repository structure

```text
├── src/football_action_valuation/
│   ├── data/          # acquisition and dataset construction
│   ├── features/      # leakage-safe feature definitions
│   ├── models/        # trainable pipelines
│   ├── evaluation/    # probabilistic metrics and calibration
│   └── analysis/      # match/player reports
├── tests/             # offline synthetic tests
├── docs/              # architecture and Colab workflow
├── notebooks/         # learning and exploration only
├── data/              # local data (ignored except documentation)
└── artifacts/         # generated models and reports
```

## Experimental protocol

The model predicts whether a shot becomes a goal. The first baseline uses only
information known before contact with the ball: location, distance, visible
angle, body part, technique, game period, current score and play context.

Models are evaluated using metrics appropriate for probabilities and imbalanced
targets:

- **Brier score** and **log-loss** for probability quality;
- **PR-AUC** for the rare positive class;
- **ROC-AUC** for ranking;
- calibration curves to compare predicted and observed frequencies.

See [`docs/architecture.md`](docs/architecture.md) for the complete protocol.

## Roadmap

- [x] Reproducible StatsBomb download and memory-conscious preprocessing
- [x] Leakage-safe xG dataset and match-level temporal split
- [x] Logistic-regression and optional LightGBM baselines
- [x] Probability metrics, model artifacts and final-match report
- [x] Offline unit tests, GitHub Actions and Docker CLI image
- [x] Action-level scoring/conceding targets inspired by VAEP
- [x] MLP baseline on event histories
- [x] Event sequences and learned action-type embeddings
- [x] GRU and compact Transformer implementations
- [x] Broad-role player profiles and experimental ratings
- [ ] Statistical uncertainty estimates and external validation
- [ ] Lightweight inference API and interactive dashboard

## Data attribution

This project uses [StatsBomb Open Data](https://github.com/hudl/open-data).
StatsBomb/Hudl must be credited when publishing analysis derived from these data;
consult the upstream repository and its license before redistribution. Raw data
are deliberately not included here.

## License

The source code in this repository is released under the MIT License. This does
not change or replace the terms that apply to third-party football data.
