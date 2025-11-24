# TP – Clustering

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Organisation

```
src/
    agglomerative.py
    dbscan.py
    hdbscan.py
    kmeans.py
    mini_batch.py
dataset/artificial/
```

Tous les scripts doivent se lancer depuis la racine du projet.

## Usage

### K-means

```bash
python src/kmeans.py "dataset.arff"
```

Recherche du meilleur `k` (méthode du coude + métriques internes).

### MiniBatch K-means

Script dédié (à implémentation similaire à kmeans.py).

### Agglomerative clustering

```bash
python src/agglomerative.py "dataset.arff" "metric" "linkage"
```

`linkage ∈ {average, complete, single, ward}`
`metric ∈ {silhouette, davies, calinski}`

### DBSCAN

```bash
python src/dbscan.py "dataset.arff" "metric"
```


Deux modes :

* `SHOW_EXECUTION = False` : epsilon automatique
* `SHOW_EXECUTION = True` : exploration de plusieurs epsilon - Tracé de graphes associé
* `SHOW_NEIGHBORS = True` : Visualisation de la distance aux points voisins

### HDBSCAN

```bash
python src/hdbscan.py "dataset.arff" "metric"
```

Exploration automatique de `min_cluster_size` et `min_samples` selon la métrique choisie.


## Données

Les fichiers `.arff` doivent être placés dans :

```
dataset/artificial/
```

## Options d’affichage

`SHOW_EXECUTION` permet d’activer ou non :

* l'exploration des hyperparamètres
* les visualisations associées
