import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.io import arff

import matplotlib
matplotlib.use("TkAgg")

path = './dataset/artificial/'
dataset_name = str(sys.argv[1])

databrut = arff.loadarff(open(path+str(dataset_name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

print("---------------------------------------")
print("Récupérer les données initiales            "+ str(dataset_name))
f0 = datanp[:,0]
f1 = datanp[:,1]

plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=10)
plt.title("Donnees initiales : "+ str(dataset_name))
plt.show()