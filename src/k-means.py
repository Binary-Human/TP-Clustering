import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

import csv
import sys

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("TkAgg")
from kneed import KneeLocator


def best_params(dataset, is_plot_graph):
    # TODO : test on samples (expected - found)
    # Evaluate metrics for different parameters and plot

    n_samples = dataset.shape[0]
    k_max = min(50, max(2, int(n_samples / 50)))  # dynamic upper bound for k
    all_metrics = []
    best_perf = {
            'k': -1,
            'inertia': -1,
            'iterations': -1,
            'runtime': -1
        }
    
    print(f"Recherche du meilleur k entre 2 et {k_max-1}...")

    for k in range(2,k_max) :
        
        # TODO Test with different hyperparameters ?
        tps1 = time.time()
        model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
        model.fit(dataset)
        tps2 = time.time()
        runtime = round((tps2 - tps1)*1000,2)

        # Silhouette (non calculable pour k=1)
        sil = metrics.silhouette_score(dataset, model.labels_)
        davies = metrics.davies_bouldin_score(dataset, model.labels_)
        calinski = metrics.calinski_harabasz_score(dataset, model.labels_)

        # Collect all_metrics
        all_metrics.append({
            'k': k,
            'inertia': model.inertia_,
            'silhouette': sil,
            'davies': davies,
            'calinski': calinski,
            'iterations': model.n_iter_,
            'runtime': runtime,
            'centroids': model.cluster_centers_,
            'labels': model.labels_
        })

    df = pd.DataFrame(all_metrics)
    print(df.head())

    # Méthode du coude
    ks = df['k'].values
    inertias = df['inertia'].values

    # Utilisation de KneeLocator
    kl = KneeLocator(ks, inertias, curve='convex', direction='decreasing')
    best_k = kl.knee or int(ks[np.argmin(np.gradient(inertias))])

    print(f"→ k optimal détecté automatiquement : {best_k} (KneeLocator)")

    # Récupération de la ligne correspondante dans all_metrics
    best_perf = next(item for item in all_metrics if item['k'] == best_k) # HERE TO SET K

    if is_plot_graph:

        plt.plot(df['k'], df['inertia'], marker='o')
        plt.title("Inertia vs k")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.show()

        plt.plot(df['k'], df['runtime'], marker='o')
        plt.title("Runtime vs k")
        plt.xlabel("k")
        plt.ylabel("Runtime")
        plt.show()

        plt.plot(df['k'], df['silhouette'], marker='o')
        plt.title("Silhouette vs k")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.show()

        plt.plot(df['k'], df['davies'], marker='o')
        plt.title("Davies-Bouldin vs k")
        plt.xlabel("k")
        plt.ylabel("Davies-Bouldin")
        plt.show()

        plt.plot(df['k'], df['calinski'], marker='o')
        plt.title("Calinski vs k")
        plt.xlabel("k")
        plt.ylabel("Calinski")
        plt.show()

    return best_perf

# name="square1.arff"

### Prepare data
path = './dataset/artificial/'
dataset_name = str(sys.argv[1])

databrut = arff.loadarff(open(path+str(dataset_name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])


# TODO : Besoin de prétraitement des données ?
# Prétraitement
scaler = StandardScaler()
scaled_datanp = scaler.fit_transform(datanp)
# TODO : Use scaled data

print("---------------------------------------")
print("Affichage données initiales            "+ str(dataset_name))
f0 = scaled_datanp[:,0]
f1 = scaled_datanp[:,1]

# Run best param selection 
results = best_params(scaled_datanp, 0)

# Show plot - make optional
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(dataset_name))
plt.show()

plt.scatter(f0, f1, c=results['labels'], s=8)
plt.scatter(results['centroids'][:, 0],results['centroids'][:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(dataset_name) + " - Nb clusters ="+ str(results['k']))
plt.show()

# Fetch the metrics from best params
print(
    f"nb clusters = {results['k']}, "
    f"nb iter = {results['iterations']}, "
    f"inertie = {results['inertia']:.2f}, "
    f"silhouette = {results['silhouette']:.4f}, "
    f"runtime = {results['runtime']} ms"
)
