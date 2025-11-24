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

from sklearn.cluster import MiniBatchKMeans

SHOW_EXECUTION = True

def best_params(dataset, is_plot_graph, batch_size=128):
    # Evaluate for different k values

    n_samples = dataset.shape[0]
    k_max = min(50, max(2, int(n_samples / 20)))
    all_metrics = []

    print(f"Recherche du meilleur k entre 2 et {k_max-1}...")

    for k in range(2, k_max):

        t0 = time.time()
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=batch_size,
            init='k-means++',
            n_init='auto'
        )
        model.fit(dataset)
        t1 = time.time()

        runtime = round((t1 - t0) * 1000, 2)

        # metrics
        sil = metrics.silhouette_score(dataset, model.labels_)
        davies = metrics.davies_bouldin_score(dataset, model.labels_)
        calinski = metrics.calinski_harabasz_score(dataset, model.labels_)

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
    kl = KneeLocator(ks, inertias, curve='convex', direction='decreasing', S=3)
    best_k = kl.knee or int(ks[np.argmin(np.gradient(inertias))])

    print(f"→ k optimal détecté automatiquement : {best_k}")

    # Récupération de la ligne correspondante dans all_metrics
    best_perf = next(item for item in all_metrics if item['k'] == best_k)

    if is_plot_graph:

        plt.plot(df['k'], df['inertia'], marker='o')
        plt.title("Inertia vs k")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.show()

        plt.plot(df['k'], df['runtime'], marker='o')
        plt.title("Runtime vs k")
        plt.xlabel("k")
        plt.ylabel("Runtime (ms)")
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
        plt.title("Calinski-Harabasz vs k")
        plt.xlabel("k")
        plt.ylabel("Calinski-Harabasz")
        plt.show()

    return best_perf

### Prepare data
path = './dataset/artificial/'
dataset_name = str(sys.argv[1])

databrut = arff.loadarff(open(path+str(dataset_name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# Prétraitement
scaler = StandardScaler()
scaled_datanp = scaler.fit_transform(datanp)

print("---------------------------------------")
print("Affichage données initiales            "+ str(dataset_name))
f0 = scaled_datanp[:,0]
f1 = scaled_datanp[:,1]

# Run best param selection 
results = best_params(scaled_datanp, SHOW_EXECUTION, batch_size=128)

# Show plot - make optional
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(dataset_name))
plt.show()

plt.scatter(f0, f1, c=results['labels'], s=8)
plt.scatter(results['centroids'][:, 0],results['centroids'][:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(dataset_name) + " - Nb clusters ="+ str(results['k']))
plt.show()

print(
    f"nb clusters = {results['k']}, "
    f"nb iter = {results['iterations']}, "
    f"inertie = {results['inertia']:.2f}, "
    f"silhouette = {results['silhouette']:.4f}, "
    f"runtime = {results['runtime']} ms"
)
