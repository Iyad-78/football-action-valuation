# Méthode v0.2 — valoriser des actions observées

## Ce qui est appris

Après une action, deux classifieurs (ou deux sorties d'un réseau) estiment la
probabilité que **le premier but dans les 10 actions suivantes** soit marqué par
l'équipe qui vient d'agir ou par son adversaire. Les labels sont construits depuis
les événements futurs ; ces événements futurs ne sont jamais des entrées du modèle.

La méthode s'inspire de [VAEP](https://socceraction.readthedocs.io/en/latest/documentation/valuing_actions/vaep.html).
Ce projet n'est pas une reproduction officielle : vocabulaire réduit, futur strict
après l'action, buts traités en récompenses terminales et censure de fin de période.

Actions retenues : passes, conduites, dribbles, tirs, interceptions, récupérations,
dégagements, duels de type tacle, contrôles manqués, dépossessions, fautes commises et buts contre
son camp. Réceptions et pressions séparées ne sont pas valorisées ; les doubles
événements de but contre son camp sont comptés une seule fois. Un duel offensif
peut avoir une observation défensive correspondante : ces contributions sont
des attributions par acteur, pas une décomposition causale indépendante.

## Informations disponibles au modèle

Les six dernières actions observées, sans traverser match, période ni but :
type d'action, coordonnées initiales/finales, réussite, pression, penalty,
écart au score, minute, équipe relative et ancienneté de l'action.

Les coordonnées et scores passés sont remis dans la perspective de l'équipe de
l'action courante. Les réseaux lisent un historique se terminant à l'instant
présent, avec masque de padding. Le Transformer peut regarder cet historique en
entier : aucun événement futur n'y figure. Les identités/noms des joueurs et des
équipes, le xG du fournisseur, les récompenses et les labels sont exclus des entrées.

Toutes les architectures voient le même historique. Logistique et LightGBM
utilisent les types one-hot et l'historique aplati ; les réseaux apprennent de
petits embeddings de types d'action. La normalisation numérique est ajustée
sur les matchs d'entraînement uniquement.

## Entraînement et contrôle

- 44 matchs train, 9 validation, 10 test et la finale de démonstration.
- Tri par date, coup d'envoi puis identifiant : aucune fenêtre entre deux matchs.
- Les derniers états sans 10 actions futures dans la même période sont censurés
  (exclus de l'apprentissage et des métriques, mais prédits pour le rapport).
- Les buts courants ne servent pas d'exemples d'apprentissage des probabilités.
- Logistique régularisée ; LightGBM peu profond et feuilles de taille minimale 100.
- MLP, GRU et Transformer compacts ; CPU, deux threads, mini-batches, AdamW,
  gradient clipping, early stopping sur la log-loss de validation.
- Deux pertes binaires, sans surpondération des buts : préserver la fréquence
  des classes. Pas de rééquilibrage qui fausserait les probabilités.
- Référence constante = fréquence des deux cibles dans le train.
- Brier, log-loss, average precision et ROC-AUC par cible et par split.
- Choix du modèle via le Brier moyen de validation. Le test et la finale ne
  choisissent ni le modèle ni le nombre d'époques.
- Pas de recalibration post-hoc entraînée dans cette version ; diagnostic de
  calibration et qualité probabiliste uniquement.
- Les dépendances entre fenêtres rendent inappropriés des intervalles calculés
  comme si toutes les lignes étaient indépendantes. Aucun intervalle de ce type
  n'est fourni. Un bootstrap par matchs et plusieurs seeds restent à faire.

## Passer des probabilités à une contribution

```
attaque = P(marquer après) - P(marquer avant)
défense = P(encaisser avant) - P(encaisser après)
contribution = attaque + défense
```

Au changement d'équipe, on échange les deux probabilités du précédent état.
Un but fournit la récompense observée `(1, 0)` ; un CSC `(0, 1)`. Le prochain
état repart des taux de base d'entraînement. On procède aussi à une remise à
zéro à chaque période. Ce crédit n'est pas une estimation causale de ce qui
se serait passé sans l'action. L'horizon glissant et les frontières de jeu
peuvent aussi modifier les probabilités : limite explicite de l'attribution.

Exception de valorisation pour un penalty concédé : `(P_pour, P_contre) = (0, q)`
où `q = (buts sur penalty du train + 1) / (penalties du train + 2)`.
Ce risque empirique est reporté dans l'état précédent du tireur, ce qui évite
de créditer un but sur penalty comme s'il était créé depuis une situation neutre.
Il s'agit d'une règle métier explicitement séparée des probabilités brutes
évaluées dans les métriques. Les fautes ordinaires restent évaluées par le modèle.

## Échelle de note expérimentale

1. Somme des contributions par joueur et match.
2. Régularisation : `90 * total / (minutes + 60)` ; limite les taux extrêmes des
   courtes apparitions. Le paramètre 60 est un choix de conception documenté.
3. Rang percentile parmi les joueurs de **validation**, dans le même grand rôle
   DEF/MID/ATT et ayant au moins 30 minutes. La finale ne définit pas cette échelle.
4. Note = `1 + 9 * percentile`, arrondie à un chiffre.

Ni cette échelle, ni les contributions ne sont des notes expertes. Les rôles sont
grossiers (position initiale ou première position observée). La validation sert
aussi au choix de modèle : une étude sérieuse des notes devra avoir une référence
plus vaste et indépendante. GK : non noté, car les arrêts ne sont pas modélisés.
Moins de 30 minutes : contribution affichée sans note. Pas de note si trop peu
de références pour le rôle. Les minutes incluent arrêts de jeu/prolongations ;
les sorties temporaires ne sont pas soustraites. Les penalties en match sont
inclus ; la séance de tirs au but est exclue. Une analyse avec/sans penalties
reste une extension utile.

## Reproductibilité

`metadata.json` enregistre seed, fenêtre, taille des données, empreinte SHA-256 du
dataset, paramètres du réseau, durée et pertes par époque. `codec.joblib` et
`model.joblib`/`model.pt` permettent l'inférence sans réentraîner. Ne charger que
les artefacts de confiance : joblib est un format de sérialisation Python.

Ce prototype fonctionne sur un tournoi ; il n'établit pas une généralisation
aux autres ligues ou saisons. La prochaine vraie amélioration consiste à étendre
le corpus et mesurer la stabilité des performances et des notes.
