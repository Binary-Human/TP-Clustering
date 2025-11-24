import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

import sys

from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("TkAgg")

from scipy.cluster.hierarchy import dendrogram

SHOW_EXECUTION = True
SHOW_DENDROGRAM = True

def plot_dendrogram(model):

    children = model.children_
    distances = model.distances_
    n_samples = len(model.labels_)

    # Count samples under each merge
    counts = np.zeros(children.shape[0], dtype=int)
    for i, merge in enumerate(children):
        count = 0
        for child in merge:
            if child < n_samples:
                count += 1
            else:
                count += counts[child - n_samples]
        counts[i] = count

    linkage_matrix = np.column_stack([children, distances, counts]).astype(float)

    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix)
    plt.title("Agglomerative Clustering Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()


def best_params(dataset, is_plot_graph, link='average', n_clusters = None, dist = None, metric='silhouette'):

    n_samples = dataset.shape[0]
    k_values = range(2,  min(50, max(2, int(n_samples / 15))))  # dynamic upper bound for k

    silhouette_scores = []

    all_metrics = []
    best_perf = {
            'n_clusters': -1,
            'silhouette': -1,
            'davies': -1,
            'calinski': -1,
            'runtime': -1,
            'labels': -1
        }

    print(f"Recherche du meilleur k entre 2 et {k_values.stop - 1}...")
    
    for k in k_values:

        # runtime
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(linkage=link, n_clusters=k, compute_distances=True)
        model.fit(dataset)
        tps2 = time.time()
        runtime = round((tps2 - tps1)*1000,2)

        # Silhouette score
        silhouette = metrics.silhouette_score(dataset, model.labels_)
        davies = metrics.davies_bouldin_score(dataset, model.labels_)
        calinski = metrics.calinski_harabasz_score(dataset, model.labels_)

        silhouette_scores.append(silhouette)

        # Collect all_metrics
        all_metrics.append({
            'n_clusters': model.n_clusters_,
            'model': model,
            'silhouette': silhouette,
            'davies': davies,
            'calinski': calinski,
            'runtime': runtime,
            'labels': model.labels_
        })

    df = pd.DataFrame(all_metrics)

    if metric == 'silhouette':
        best_k = df.loc[df['silhouette'].idxmax(), 'n_clusters']
    elif metric == 'calinski':
        best_k = df.loc[df['calinski'].idxmax(), 'n_clusters']
    elif metric == 'davies':
        best_k = df.loc[df['davies'].idxmin(), 'n_clusters']
    else:
        raise ValueError("Metric must be: silhouette, calinski, or davies")
    # Récupération de la ligne correspondante dans all_metrics
    best_perf = next(item for item in all_metrics if item['n_clusters'] == best_k)

    print(df.head())

    if is_plot_graph:

        plt.plot(df['n_clusters'], df['runtime'], marker='o')
        plt.title("Runtime vs k")
        plt.xlabel("k")
        plt.ylabel("Runtime")
        plt.show()

        plt.plot(df['n_clusters'], df['davies'], marker='o')
        plt.title("Davies-Bouldin vs k")
        plt.xlabel("k")
        plt.ylabel("Davies-Bouldin")
        plt.show()

        plt.plot(df['n_clusters'], df['calinski'], marker='o')
        plt.title("Calinski-Harabasz vs k")
        plt.xlabel("k")
        plt.ylabel("Calinski-Harabasz")
        plt.show()

        plt.plot(df['n_clusters'], df['silhouette'], marker='o')
        plt.title("Silhouette vs k")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.show()
    
    if SHOW_DENDROGRAM:
        print("Affichage du dendrogramme...")
        plot_dendrogram(best_perf['model'])
        
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

linkage = str(sys.argv[3])                                                                          # Update linkage here
# Run best param selection 
results = best_params(scaled_datanp, SHOW_EXECUTION, link=linkage, metric = str(sys.argv[2]))

# Show plot - make optional
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(dataset_name))
plt.show()

# After clustering
plt.scatter(f0, f1, c=results['labels'], s=8)
plt.title("Données après clustering : "+ str(dataset_name) + " - Nb clusters ="+ str(results['n_clusters']))
plt.show()

# Fetch the metrics from best params
print(
    f"nb clusters = {results['n_clusters']}, "
    f"silhouette = {results['silhouette']:.4f}, "
    f"davies = {results['davies']:.4f}, "
    f"calinski = {results['calinski']:.4f}, "
    f"runtime = {results['runtime']} ms"
)

# TODO : Need to use
# plot_dendrogram(scaled_datanp)