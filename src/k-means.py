# Load data
# Make some clusters
# algorithms to automatize 
    # Show knee method

# Final parameters
# Final visualization 

# Metrics 
    # Scores
    # Execution

import numpy as np
import matplotlib.pyplot as plt
import time

import csv
import sys

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics



import matplotlib
matplotlib.use("TkAgg")

path = './dataset/artificial/'
dataset_name = str(sys.argv[1])

databrut = arff.loadarff(open(path+str(dataset_name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

k = 4

print("---------------------------------------")
print("Affichage données initiales            "+ str(dataset_name))
f0 = datanp[:,0]
f1 = datanp[:,1]

# TODO : Besoin de prétraitement des données ?

# Show plot - make optional
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(dataset_name))
plt.show()

# Run Kmeans to find k clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
tps1 = time.time()
model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# Clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_
runtime = round((tps2 - tps1)*1000,2)

plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(dataset_name) + " - Nb clusters ="+ str(k))
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ", inertie, ", runtime = ", runtime,"ms")
