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

def plot_dendrogram(data):
    # Create linkage matrix and then plot the dendrogram
 
    # setting distance_threshold=0 ensures we compute the full tree.
    model = cluster.AgglomerativeClustering(distance_threshold=0, linkage='average', n_clusters=None)
    model = model.fit(data)

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    plt.figure(figsize=(12, 12))
    plt.title("Hierarchical Clustering Dendrogram")

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix) #, **kwargs)

    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


def best_params(dataset, is_plot_graph, link='average', n_clusters = None, dist = None):

    # TODO : Tester sur linkage différents ? -> Which one gives un k qui se démarque

    n_samples = dataset.shape[0]
    k_values = range(2,  min(50, max(2, int(n_samples / 15))))  # dynamic upper bound for k

    silhouette_scores = []

    all_metrics = []
    best_perf = {
            'n_clusters': -1,
            'silhouette': -1,
            'runtime': -1,
            'labels': -1
        }

    print(f"Recherche du meilleur k entre 2 et {k_values.stop - 1}...")
    
    for k in k_values:

        # runtime
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(linkage=link, n_clusters=k)
        labels = model.fit_predict(dataset)
        tps2 = time.time()
        runtime = round((tps2 - tps1)*1000,2)

        # Silhouette score
        score = metrics.silhouette_score(dataset, labels)
        silhouette_scores.append(score)

        # TODO : Rajouter d'autres indicateurs ?

        # Collect all_metrics
        all_metrics.append({
            'n_clusters': model.n_clusters_,
            'silhouette': score,
            'runtime': runtime,
            'labels': model.labels_
        })

    # TODO : Réflechir a dautre facon de prioriser
    # TODO : Illustrate caveats in report
    best_k = k_values[np.argmax(silhouette_scores)]

    # Récupération de la ligne correspondante dans all_metrics
    best_perf = next(item for item in all_metrics if item['n_clusters'] == best_k)

    df = pd.DataFrame(all_metrics)
    print(df.head())

    if is_plot_graph:
        # TODO : add other indicators
        # Davies Bouldin
        # Calinski

        plt.plot(df['n_clusters'], df['runtime'], marker='o')
        plt.title("Runtime vs k")
        plt.xlabel("k")
        plt.ylabel("Runtime")
        plt.show()

        plt.plot(df['n_clusters'], df['silhouette'], marker='o')
        plt.title("Silhouette vs k")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.show()
        
    return best_perf

### Prepare data
path = './dataset/artificial/'
dataset_name = str(sys.argv[1])

databrut = arff.loadarff(open(path+str(dataset_name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# TODO : Besoin ?
# Prétraitement
scaler = StandardScaler()
scaled_datanp = scaler.fit_transform(datanp)

print("---------------------------------------")
print("Affichage données initiales            "+ str(dataset_name))
f0 = scaled_datanp[:,0]
f1 = scaled_datanp[:,1]

# Run best param selection 
results = best_params(scaled_datanp, 1)

# Show plot - make optional
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(dataset_name))
plt.show()

# After clustering
plt.scatter(f0, f1, c=results['labels'], s=8)
plt.title("Données après clustering : "+ str(dataset_name) + " - Nb clusters ="+ str(results['n_clusters']))
plt.show()

# Fetch the metrics from best params
# TODO : complement metrics
print(
    f"nb clusters = {results['n_clusters']}, "
    f"silhouette = {results['silhouette']:.4f}, "
    f"runtime = {results['runtime']} ms"
)

# plot_dendrogram(scaled_datanp)