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

def numberOfNeighbors(dataset, k=5, n_values=6, min_frac=0.01, max_frac=0.3):

    N = len(dataset)
    k = min(max(2, k), N - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(dataset)
    dists, _ = nn.kneighbors(dataset)
    knn_d = np.sort(dists[:, -1])

    percentiles = np.linspace(10, 90, n_values)
    scales = np.percentile(knn_d, percentiles)

    raw = (scales.max() / scales)
    min_size_low = max(5, int(min_frac * N))
    min_size_high = max(min_size_low + 1, int(max_frac * N))

    # distribute to integer interval
    mcs = np.linspace(min_size_low, min_size_high, n_values)
    mcs = np.unique(np.clip(np.round(mcs).astype(int), min_size_low, min_size_high))
    return sorted(mcs)


def best_params(dataset, is_plot_graph, metric="stability", min_cluster_size_list=None, min_samples_list=None):

    N = dataset.shape[0]

    if min_cluster_size_list is None:
        min_cluster_size_list = numberOfNeighbors(dataset, k=2, n_values=6)
    if min_samples_list is None:
        min_samples_list = [1, 5, 10]

    print("Adaptive min_cluster_size candidates:", min_cluster_size_list)
    print("min_samples candidates:", min_samples_list)

    all_metrics = []
    best_perf = None

    print(f"Recherche de la meilleure combinaison min_cluster_size × min_samples basée sur la métrique: {metric}")

    for mcs in min_cluster_size_list:
        for ms in min_samples_list:
            tps1 = time.time()
            model = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs),
                min_samples=int(ms),
                metric='euclidean',
                cluster_selection_method='eom',
                gen_min_span_tree=True
            )
            model.fit(dataset)
            tps2 = time.time()
            runtime = round((tps2 - tps1) * 1000, 2)

            labels = model.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            try:
                stability = float(np.sum(model.cluster_persistence_)) if hasattr(model, 'cluster_persistence_') else float('nan')
            except Exception:
                stability = float('nan')

            silhouette = float('nan')
            davies = float('nan')
            calinski = float('nan')

            masked_idx = labels != -1

            if n_clusters > 1:
                silhouette = metrics.silhouette_score(dataset[masked_idx], labels[masked_idx])
                davies = metrics.davies_bouldin_score(dataset[masked_idx], labels[masked_idx])
                calinski = metrics.calinski_harabasz_score(dataset[masked_idx], labels[masked_idx])

            all_metrics.append({
                "min_cluster_size": mcs,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "stability": stability,
                "silhouette": silhouette,
                "davies": davies,
                "calinski": calinski,
                "runtime": runtime,
                "labels": labels,
                "model": model
            })

    df = pd.DataFrame(all_metrics)

    df = df.sort_values(['min_cluster_size', 'min_samples']).reset_index(drop=True)
    pd.set_option('display.max_rows', 200)
    print(df.head())

    if metric == 'davies':
        best_idx = df[metric].dropna().idxmin() if df[metric].dropna().size > 0 else None
    else:
        best_idx = df[metric].dropna().idxmax() if df[metric].dropna().size > 0 else None

    best_perf = df.loc[best_idx].copy()
    print(best_perf)

    if is_plot_graph:
        print("Affichage du minimum spanning tree du meilleur modèle HDBSCAN...")
        mst = best_perf['model'].minimum_spanning_tree_
        mst.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=50, edge_linewidth=2)
        plt.show()

        # Runtime vs min_cluster_size (aggregate: show mean runtime per min_cluster_size)
        df_plot = df.groupby('min_cluster_size').agg({'runtime': 'mean'}).reset_index()
        plt.plot(df_plot['min_cluster_size'], df_plot['runtime'], marker='o')
        plt.title("Runtime vs min_cluster_size (mean over min_samples)")
        plt.xlabel("min_cluster_size")
        plt.ylabel("Runtime (ms)")
        plt.show()

        for metric_name, label in [('stability', 'Stability'), ('silhouette', 'Silhouette'),
                                   ('davies', 'Davies-Bouldin'), ('calinski', 'Calinski-Harabasz')]:
            try:
                dfm = df.groupby('min_cluster_size')[metric_name].mean().reset_index()
                plt.plot(dfm['min_cluster_size'], dfm[metric_name], marker='o')
                plt.title(f"{label} vs min_cluster_size (mean over min_samples)")
                plt.xlabel("min_cluster_size")
                plt.ylabel(label)
                plt.show()
            except Exception as e:
                print(f"Unable to plot {metric_name}:", e)
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
    f"noise points = {results['n_noise']}, "
    f"stability = {results['stability']}, "
    f"silhouette = {results['silhouette']:.4f}, "
    f"davies = {results['davies']:.4f}, "
    f"calinski = {results['calinski']:.4f}, "
    f"runtime = {results['runtime']} ms"
)