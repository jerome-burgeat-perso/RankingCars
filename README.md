# RankingCars

# Système de Support à la Décision pour la Visualisation

## Installation

### Prérequis
Python 3.6 ou supérieur est requis, avec les paquets suivants :
- `Pandas` : Utilisé pour lire les fichiers CSV et gérer les données au format DataFrame, facilitant ainsi la manipulation des données de manière efficace.
- `NumPy` : Utilisé pour la gestion de grands tableaux et matrices multidimensionnels, ainsi qu'une collection de fonctions mathématiques pour opérer sur ces tableaux.
- `Matplotlib` : Employé pour créer des visualisations statiques, interactives et animées en Python.
- `NetworkX` : Appliqué pour la création, la manipulation et l'étude de la structure, de la dynamique et des fonctions des réseaux complexes.


### Installation
1. Assurez-vous que Python 3.6+ est installé sur votre système.
2. Installez les bibliothèques Python requises avec pip :
   ```
   pip install pandas numpy matplotlib networkx
   ```
3. Clonez ce dépôt ou téléchargez le script sur votre machine locale.

## Vue d'ensemble
Ce script Python est conçu pour aider dans les processus de prise de décision en visualisant divers modèles de décision tels que la frontière de Pareto, Promethee I et II, et Electre IV et IS. Il offre des fonctionnalités pour analyser et classer les alternatives basées sur plusieurs critères et aide à comprendre les forces et faiblesses de chaque alternative.

## Fonctionnalités
- **Visualisation de données** : Génère des représentations visuelles pour l'analyse décisionnelle, y compris les meilleurs N, les pires N et les diagrammes de Pareto.
- **Modèles de décision multiples** : Prend en charge les méthodes Promethee I et II, Electre IV et IS.
- **Sélection interactive** : Les utilisateurs peuvent sélectionner les colonnes d'intérêt et choisir le type de diagramme à visualiser.

### Exécution du script
Pour exécuter le script, naviguez jusqu'au répertoire du script dans votre terminal et exécutez :
```bash
python main.py <chemin_vers_csv_entree>
```
Remplacez `<chemin_vers_csv_entree>` par le chemin vers votre fichier CSV contenant les données de décision.

## Utilisation
Après avoir lancé le script, suivez les invites à l'écran pour :
1. Sélectionner un type de diagramme parmi les options disponibles.
2. Choisir une colonne spécifique (si nécessaire selon le type de diagramme sélectionné).
3. Voir les résultats et décider de continuer avec une autre visualisation.

## Contribution
Les contributions à ce projet sont bienvenues. Veuillez fork le dépôt et soumettre une pull request avec vos améliorations.