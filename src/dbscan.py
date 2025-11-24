import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import pandas as pd
import math

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

import sys

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("TkAgg")

SHOW_NEIGHBORS = True
SHOW_EXECUTION = False

def distanceToNeighbors(dataset, v, showplot):

    print("---------------------------------------")
    print("Calculating distance to neighbors")

    # Distances to the k nearest neighbors
    v = 5
    neigh = NearestNeighbors(n_neighbors=v)
    neigh.fit(dataset)
    distances, indices = neigh.kneighbors(dataset)

    # Average these distances
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0, distances.shape[0])])
    # Sort ascending
    distancetrie = np.sort(newDistances)

    if showplot:
        plt.title(f"Distances to the {str(v)} nearest neighbors for all points")
        plt.xlabel('"id" of the point in the model')
        plt.ylabel("distance")
        plt.plot(distancetrie)
        plt.show()

    return np.percentile(distancetrie, 99)

def best_params(dataset, is_plot_graph, metric, eps = 0.5, min_samples = 3 ):

    epsilon = distanceToNeighbors(dataset, min_samples, SHOW_NEIGHBORS)
    # To go through values in the loop
    step = epsilon/10
    e = 0

    all_metrics = []
    best_perf = {
            'n_clusters': -1,
            'epsilon': -1,
            'min_samples': -1,
            'silhouette': 1,
            'davies': -1,
            'calinski': -1,
            'noise_points': -1,
            'runtime': -1,
            'labels': -1
            }
    
    print(f"Recherche du meilleur espsilon entre 0 et {epsilon}...")
    
    if is_plot_graph:
        # We only run the comparative if graphs are prompted

        while (e<epsilon*2.0):
            e += step 

            tps1 = time.time()
            model = cluster.DBSCAN(eps=e, min_samples=min_samples)
            model.fit(dataset)
            tps2 = time.time()
            labels = model.labels_
            runtime = round((tps2 - tps1)*1000,2)

            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters < 2: # only one cluster, invalid
                silhouette_score = -1
                davies = np.inf
                calinski = -1
            else:
                # Silhouette score
                silhouette_score = metrics.silhouette_score(dataset, labels)
                davies = metrics.davies_bouldin_score(dataset, model.labels_)
                calinski = metrics.calinski_harabasz_score(dataset, model.labels_)
            
            # Just for code consistency
            all_metrics.append({
                    'n_clusters': n_clusters,
                    'epsilon': e,
                    'min_samples': min_samples,
                    'silhouette': silhouette_score,
                    'davies': davies,
                    'calinski': calinski,
                    'noise_points': n_noise,
                    'runtime': runtime,
                    'labels': model.labels_
                })
            
        df = pd.DataFrame(all_metrics)
        print(df.head())

        print(f"Recherche best performance by {metric} score...")

        # Récupération de la ligne correspondante dans all_metrics
        best_perf = df.loc[df[metric].idxmax()]
        print(best_perf)
        
        plt.plot(df['epsilon'], df['runtime'], marker='o')
        plt.title("Runtime vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Runtime")
        plt.show()

        plt.plot(df['epsilon'], df['silhouette'], marker='o')
        plt.title("Silhouette vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Silhouette")
        plt.show()

        plt.plot(df['epsilon'], df['davies'], marker='o')
        plt.title("Davies-Bouldin vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Davies-Bouldin")
        plt.show()

        plt.plot(df['epsilon'], df['calinski'], marker='o')
        plt.title("Calinski vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Calinski")
        plt.show()
        
    else :

        print(f"Testing for epsilon = {epsilon}")
    
        tps1 = time.time()
        model = cluster.DBSCAN(eps=epsilon, min_samples=min_samples)
        model.fit(dataset)
        tps2 = time.time()
        labels = model.labels_
        runtime = round((tps2 - tps1)*1000,2)

        # Silhouette score
        silhouette_score = metrics.silhouette_score(dataset, labels)
        davies = metrics.davies_bouldin_score(dataset, model.labels_)
        calinski = metrics.calinski_harabasz_score(dataset, model.labels_)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # For printing performance
        best_perf = {
                'n_clusters': n_clusters,
                'epsilon': epsilon,
                'min_samples': min_samples,
                'silhouette': silhouette_score,
                'davies': davies,
                'calinski': calinski,
                'noise_points': n_noise,
                'runtime': runtime,
                'labels': model.labels_
            }
            
    return best_perf

##################################################################

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
results = best_params(scaled_datanp, SHOW_EXECUTION, metric = str(sys.argv[2])) # if exist, silhouette by default

# Show plot - make optional
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(dataset_name))
plt.show()

# After clustering
plt.scatter(f0, f1, c=results['labels'], s=8)
plt.title("Données après clustering : "+ str(dataset_name) + " - Nb clusters ="+ str(results['n_clusters']) + "- Epsilon= "+str(results['epsilon'])+" MinPts= "+str(results['min_samples']))
plt.show()

# Fetch the metrics from best params
print(
    f"nb clusters = {results['n_clusters']}, "
    f"noise points = {results['noise_points']:.4f}, "
    f"silhouette = {results['silhouette']:.4f}, "
    f"davies = {results['davies']:.4f}, "
    f"calinski = {results['calinski']:.4f}, "
    f"runtime = {results['runtime']} ms"
)
