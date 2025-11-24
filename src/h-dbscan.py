import hdbscan

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from scipy.io import arff
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import sys

import matplotlib
matplotlib.use("TkAgg")

SHOW_EXECUTION = True

from sklearn.neighbors import NearestNeighbors
import numpy as np

# The objective is to determine suitable min_cluster_size values
def numberOfNeighbors(dataset, k=5, n_values=6):
    N = len(dataset)

    nn = NearestNeighbors(n_neighbors=k).fit(dataset)
    dists, _ = nn.kneighbors(dataset)
    knn_d = np.sort(dists[:, -1])

    percentiles = np.linspace(10, 90, n_values)
    scales = np.percentile(knn_d, percentiles)

    # min_cluster_size candidates
    mcs = (scales.max() / scales) * (0.02 * N)
    mcs = np.clip(mcs, 5, 0.2 * N).astype(int)

    return sorted(set(mcs))


def best_params(dataset, is_plot_graph, metric="stability"):

    # Data-adaptive search space for HDBSCAN
    N = dataset.shape[0]

    min_cluster_size_list = numberOfNeighbors(dataset, k=5, n_values=6)
    print("Adaptive min_cluster_size candidates:", min_cluster_size_list)

    all_metrics = []
    best_perf = None

    print(f"Recherche de la meilleure combinaison min_cluster_size × min_samples basée sur la stabilité...")

    for mcs in min_cluster_size_list:
        
        tps1 = time.time()
        model = hdbscan.HDBSCAN(min_cluster_size=mcs, metric='euclidean', cluster_selection_method='eom', gen_min_span_tree=True)
        model.fit(dataset)
        labels = model.labels_
        tps2 = time.time()
        runtime = round((tps2 - tps1) * 1000, 2)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        if n_clusters > 1:
            silhouette = metrics.silhouette_score(dataset, labels)
            davies = metrics.davies_bouldin_score(dataset, labels)
            calinski = metrics.calinski_harabasz_score(dataset, labels)
        
        all_metrics.append({
            "min_cluster_size": mcs,
            "n_clusters": n_clusters,
            "min_samples": model.min_samples,
            "silhouette": silhouette,
            "davies": davies,
            "calinski": calinski,
            "noise_points": n_noise,
            "runtime": runtime,
            "labels": labels,
            "model": model
        })

    df = pd.DataFrame(all_metrics)
    print(df.head())
    print(f"Recherche du meilleur modèle selon la métrique spécifiée...")

    # Now safely get best by defined metric
    best_perf = df.loc[df[metric].idxmax()]

    print(best_perf)

    if is_plot_graph:

        print("Affichage du minimum spanning tree du meilleur modèle HDBSCAN...")
        best_perf['model'].minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=50, edge_linewidth=2)
        plt.show()
    
        # Runtime vs min_cluster_size
        plt.plot(df['min_cluster_size'], df['runtime'], marker='o')
        plt.title("Runtime vs min_cluster_size")
        plt.xlabel("min_cluster_size")
        plt.ylabel("Runtime (ms)")
        plt.show()

        # davies
        plt.plot(df['min_cluster_size'], df['davies'], marker='o')
        plt.title("Davies-Bouldin vs min_cluster_size")
        plt.xlabel("min_cluster_size")
        plt.ylabel("Davies-Bouldin")
        plt.show()

        # silhouette
        plt.plot(df['min_cluster_size'], df['silhouette'], marker='o')
        plt.title("Silhouette vs min_cluster_size")
        plt.xlabel("min_cluster_size")
        plt.ylabel("Silhouette")
        plt.show()

        # calinski
        plt.plot(df['min_cluster_size'], df['calinski'], marker='o')
        plt.title("Calisnki-Harabasz vs min_cluster_size")
        plt.xlabel("min_cluster_size")
        plt.ylabel("Calinski-Harabasz")
        plt.show()

        print("Affichage de l'arbre condensé du meilleur modèle HDBSCAN...")
        best_perf['model'].condensed_tree_.plot(select_clusters=True, label_clusters=True)
        plt.show()

    return best_perf.to_dict()

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
results = best_params(scaled_datanp, SHOW_EXECUTION, metric = str(sys.argv[2])) # use stability

# Show plot - make optional
plt.scatter(f0, f1, s=8)
plt.title("Données initiales : "+ str(dataset_name))
plt.show()

# After clustering
plt.scatter(f0, f1, c=results['labels'], s=8)
plt.title("Données après clustering : "+ str(dataset_name) +
          f" - Nb clusters = {results['n_clusters']} - Min size = {results['min_cluster_size']} - MinPts = {results['min_samples']}")
plt.show()

# Fetch the metrics from best params
print(
    f"nb clusters = {results['n_clusters']}, "
    f"noise points = {results['noise_points']}, "
    f"runtime = {results['runtime']} ms",
    f"silhouette = {results['silhouette']:.4f}, "
    f"davies = {results['davies']:.4f}, "
    f"calinski = {results['calinski']:.4f}"
)