# Résultats vérifiés — v0.2

Exécution complète sur CPU avec seed 42, historique de six actions et au plus
six époques pour les réseaux. Les fichiers générés restent locaux ; cette
synthèse peut être versionnée sur GitHub avec le code.

## Périmètre effectivement exécuté

- 64 matchs de la Coupe du monde 2022, 142 550 actions retenues.
- 44 matchs d'entraînement, 9 de validation, 10 de test et la finale à part.
- 95 065 exemples d'entraînement, 19 639 de validation et 23 696 de test
  après exclusion des buts courants et des fenêtres futures incomplètes.
- Régression logistique, LightGBM, MLP, GRU et Transformer entraînés et sauvegardés.
- Finale : 2 649 actions, 35 profils de joueurs, dont 27 avec une note
  expérimentale. Pas de note pour les gardiens ou les apparitions trop courtes.

Le nombre de lignes n'est pas un nombre d'observations indépendantes : des
fenêtres successives se recouvrent et les joueurs apparaissent dans plusieurs matchs.

## Comparaison des probabilités

Le Brier de validation ci-dessous est la moyenne des deux cibles, marquer et
encaisser. Plus il est bas, mieux c'est. La sélection exclut le test et la finale.
L'average precision (AP) de test concerne ici seulement la cible « marquer ».

| Modèle | Brier moyen validation ↓ | AP test, marquer ↑ |
|---|---:|---:|
| Constante issue de train | 0,006134 | 0,01034 |
| Régression logistique | 0,006090 | 0,04622 |
| **LightGBM — retenu** | **0,006026** | **0,08499** |
| MLP | 0,006078 | 0,04803 |
| GRU | 0,006047 | 0,06349 |
| Transformer | 0,006043 | 0,05533 |

Interprétation : les réseaux ne gagnent pas sur le critère de sélection dans
cette exécution. L'écart est faible ; une seule seed et un seul tournoi ne
démontrent pas une supériorité robuste. Le faible Brier brut s'explique aussi par
la rareté des buts : la comparaison à la constante est indispensable.
Une AP de 0,08499 n'est **pas** une précision globale de 8,5 %.

## Vérifications logicielles

- `pytest -p no:cacheprovider` : **20 tests réussis**.
- `ruff check src tests` : aucun problème détecté.
- Les cinq modèles rechargés reproduisent les probabilités sauvegardées de la
  finale avec une différence absolue maximale inférieure à `1e-6`.
- Aucun match partagé entre splits ; tirs au but exclus ; six buts et trois
  penalties concédés identifiés dans la finale.

Dix avertissements de dépréciation NumPy/joblib sont apparus pendant les tests,
sans échec. Docker et la CI distante n'ont pas été exécutés ici.

## Ce que les notes ne prouvent pas

Les notes sont des percentiles de contributions estimées, avec des conventions
explicites pour les minutes, les rôles et les penalties. Elles ne sont ni des
notes expertes validées, ni une preuve d'absence de biais. Le jeu sans ballon et
les arrêts des gardiens ne sont pas représentés.

Pour les chiffres détaillés : `outputs/v0.2/comparison.csv`, `RAPPORT.md`,
`run.json` et `verification.json`. Pour relancer : voir le README.
