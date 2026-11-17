# Présenter le projet en entretien

## Présentation en 45 secondes

« Je développe un prototype explicable pour évaluer les contributions de joueurs
de football à partir des événements StatsBomb. J'ai construit un pipeline qui
transforme les données brutes en exemples d'apprentissage. Après une première
baseline xG sur les tirs, j'ai étendu le modèle aux actions : il prédit les
probabilités de marquer ou d'encaisser prochainement, puis mesure leur variation.
Je compare régression logistique, LightGBM et petits réseaux MLP, GRU et Transformer
sur les mêmes historiques. Les matchs d'entraînement, de validation, de test et
la finale de démonstration sont séparés. Le résultat est un rapport par joueur
avec les actions qui expliquent sa contribution. Les notes restent expérimentales,
car je n'observe pas tout le jeu sans ballon. »

Cette présentation doit correspondre aux exécutions effectivement terminées dans
`outputs/v0.2/run.json`. Lire le rapport avant de citer des résultats.

La v0.2 a effectivement été exécutée : 142 550 actions, cinq modèles entraînés,
20 tests réussis. LightGBM est retenu sur la validation ; les réseaux ne font pas
mieux sur ce critère dans cette exécution. Voir `docs/RESULTATS_V02.md` pour les
chiffres et leurs limites. Les notes de joueurs restent expérimentales.

## Les questions à savoir traiter

**Où sont les labels si personne ne donne la bonne note ?**
Les labels sont observables : premier but dans les dix prochaines actions, pour
ou contre l'équipe qui agit. La note vient ensuite d'une règle de restitution.
Je n'entraîne pas un réseau à reproduire SofaScore.

**Pourquoi un Transformer ?**
Pour tester si la représentation de l'historique améliore les probabilités.
Je le compare à un modèle tabulaire qui reçoit le même historique. Utiliser un
réseau plus complexe n'implique pas qu'il sera meilleur.

**Qu'est-ce qu'un embedding ici ?**
Un vecteur appris pour chaque type d'action (passe, tir, etc.). Sa valeur évolue
avec l'entraînement pour aider les prédictions.

**Comment éviter les fuites ?**
Séparation des matchs entiers, normalisation sur train, fenêtres arrêtées au
présent, pas d'identité ni de label futur en entrée. La fin de période ne devient
pas artificiellement une longue série de labels négatifs.

**Comment sont comparés les modèles ?**
Même cible, mêmes matchs et même fenêtre. Brier/log-loss pour les probabilités,
average precision pour les événements rares. Une constante issue de train donne
une référence minimale. Le choix final utilise seulement la validation.

**Que signifient les notes ?**
Un rang relatif aux contributions des joueurs de validation du même grand rôle,
après régularisation par temps de jeu. La formule est publique. Une note élevée
n'établit pas que le joueur a été meilleur dans tous les aspects du jeu.

**Quelle difficulté d'ingénierie ?**
La cohérence temporelle, la perspective lors d'un changement d'équipe et les
frontières de match/but. Les tests vérifient ces invariants et le rechargement
des modèles. Le pipeline tourne sur CPU et produit des artefacts traçables.

**Quelles limites ?**
Un seul tournoi, objectifs rares, choix d'échelle discutables, faible couverture
du jeu sans ballon, gardiens non notés, absence de validation humaine indépendante.
Les tests logiciels ne prouvent pas la justesse sportive des notes.

**Quelle est ta contribution personnelle ?**
Expliquer honnêtement le périmètre que tu as conçu, vérifié et compris, ainsi que
l'aide reçue pour le code. Ne pas prétendre avoir implémenté sans assistance un
modèle dont tu ne peux pas expliquer la cible, les entrées ou l'évaluation.

## Démo courte

1. Montrer `outputs/v0.2/RAPPORT.md` et une contribution détaillée dans
   `final_key_actions.csv`.
2. Montrer le choix de modèle dans `selection.json`, décidé avant le test.
3. Expliquer les deux probabilités puis leur différence avant/après l'action.
4. Montrer les tests de frontière temporelle dans `tests/test_actions.py`.
5. Donner une limite et une amélioration concrète, sans promettre une note objective.
