import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import pandas as pd

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

import sys

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("TkAgg")

def best_params(dataset, is_plot_graph, eps = 0.5, min_samples = 5):

    # TODO : define a heuristic for optimization

    best_perf = {
            'n_clusters': -1,
            'epsilon': -1,
            'silhouette': -1,
            'min_samples': -1,
            'noise_points': -1,
            'runtime': -1,
            'labels': -1
        }

    tps1 = time.time()
    model = cluster.DBSCAN(eps=eps, min_sampless=min_samples)
    model.fit(dataset) # TODO : fit_predict ?
    tps2 = time.time()
    labels = model.labels_
    runtime = round((tps2 - tps1)*1000,2)

    # Silhouette score
    silhouette_score = metrics.silhouette_score(dataset, labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    all_metrics = []

    # Define best parameters after search TODO : no loop ?
    best_perf = {
            'n_clusters': n_clusters,
            'epsilon': eps,
            'silhouette': silhouette_score, # When error null ? TODO
            'min_samples': min_samples,
            'noise_points': n_noise,
            'runtime': runtime,
            'labels': model.labels_
        }
    
    # Just for code consistency
    all_metrics.append(best_perf)

    df = pd.DataFrame(all_metrics)
    print(df.head())

    if is_plot_graph:
        # TODO : Des choses a imprimer uniquement si loop

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


##################################################################

### Prepare data
path = './dataset/artificial/'
dataset_name = str(sys.argv[1])

databrut = arff.loadarff(open(path+str(dataset_name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# TODO : sur demande ?
# Prétraitement
scaler = StandardScaler()
scaled_datanp = scaler.fit_transform(datanp)
# TODO : Use scaled data

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
plt.title("Données après clustering : "+ str(dataset_name) + " - Nb clusters ="+ str(results['n_clusters']) + "- Epislon= "+str(results['epsilon'])+" MinPts= "+str(results['min_samples']))
plt.show()

# Fetch the metrics from best params
# TODO : complement metrics
print(
    f"nb clusters = {results['n_clusters']}, "
    f"noise points = {results['silhouette']:.4f}, "
    f"silhouette = {results['silhouette']:.4f}, "
    f"runtime = {results['runtime']} ms"
)
