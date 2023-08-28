import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay, voronoi_plot_2d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Constrained_Voronoi_relaxation import voronoi_relaxation
from utils import mean_absorb_graph

#
# import and transform dataset
#

#import data
df = pd.read_csv('Absorption_database.csv', index_col='Material').drop_duplicates()
X = df.copy()

# train PCA and tranform set
pca = PCA(n_components = 2)
X_r = pca.fit(X).transform(X)

# voronoi relaxation to obtain a more equally filled space
vor = voronoi_relaxation(X_r, 50)
X_r = vor.points

#
# Trim unnecessary points from graph
#  

tri = Delaunay(X_r)
interp_error = []

# for each point : build Delunay without it, try to interpolate to it's value and store the error produced
for i in range(0, X_r.shape[0]):
    temp_X = np.delete(X, i, 0)
    temp_tri = Delaunay(np.delete(X_r, i, 0))
    simp = temp_tri.find_simplex(X_r[i])
    if(simp != -1):
        b = temp_tri.transform[simp,:2].dot(np.transpose(X_r[i: i+1] - temp_tri.transform[simp,2]))
        weights = np.c_[np.transpose(b), 1 - b.sum(axis=0)]
        newFilter = (temp_X[temp_tri.simplices[simp]] * np.transpose(weights)).sum(axis=0)
        #interp_error.append(mean_squared_error(X[i:i+1], [newFilter], squared=False))
        interp_error.append(mean_absolute_error(X[i:i+1], [newFilter]))
    else:
        interp_error.append(10)

# find all errors below a threshold
min_error = min(interp_error)
arr = np.array(interp_error)
values_to_delete = arr[np.where(arr<0.01)]
values_to_delete = np.sort(values_to_delete)
index_to_delete = np.zeros(values_to_delete.size, np.int64)

#print(len([i for i in interp_error if ((i != 10) & (i >= 0.2))]))
print("Mean absolute error: " + str(np.mean([i for i in interp_error if i != 10])))

# find index of points with minimal error
for i in range(0, values_to_delete.size):
    index_to_delete[i] = np.where(arr==values_to_delete[i])[0][0]

X_trimmed = X.copy()
X_r_trimmed =  X_r.copy()

# delete all points selected
for i in index_to_delete:
    trimmedIndex = np.where(X_r_trimmed == X_r[i])[0][0]
    X_trimmed = X_trimmed.drop(X_trimmed.index[trimmedIndex], axis=0)
    X_r_trimmed = np.delete(X_r_trimmed, trimmedIndex, axis=0)


tri_trimmed = Delaunay(X_r_trimmed)

#print (index_to_delete.size)

#
# polts
#

# pre-trimming result
mean_absorb_graph(X.copy(), X, X_r)

# voronoi relaxation result
voronoi_plot_2d(vor)
plt.show()

# pre-trimming delunay
plt.triplot(X_r[:,0], X_r[:,1], tri.simplices)
plt.plot(X_r[:,0], X_r[:,1], 'o')
plt.show()

# post-trimming delunay
plt.triplot(X_r_trimmed[:,0], X_r_trimmed[:,1], tri_trimmed.simplices)
plt.plot(X_r_trimmed[:,0], X_r_trimmed[:,1], 'o')
plt.show()

# result
mean_absorb_graph(X_trimmed.copy(), X_trimmed, X_r_trimmed)

pass