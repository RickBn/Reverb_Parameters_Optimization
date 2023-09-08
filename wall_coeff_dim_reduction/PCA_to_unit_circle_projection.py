import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import Polygon, LineString

#
# import and transform dataset
#

def get_filter_absorption(x, y, tri, x_original, ply=None, unit_circle=False):

    # line = LineString([[0, 0], [x_in, y_in]])
    #
    # line = line.intersection(ply)
    #
    # x = 0
    # y = 0
    #
    # if(not x_in == 0 or not y_in == 0):
    #     x=line.coords[1][0]
    #     y=line.coords[1][1]

    interp_filter = np.zeros(6)

    simplex = tri.find_simplex([x,y])

    if(simplex != -1):
        b = tri.transform[simplex,:2].dot(np.transpose([[x,y]] - tri.transform[simplex,2]))
        weights = np.c_[np.transpose(b), 1 - b.sum(axis=0)]
        interp_filter = (x_original[tri.simplices[simplex]] * np.transpose(weights)).sum(axis=0)
        return interp_filter
    else:
        # CASA DA GESTIRE
        interp_filter = interp_filter - 1
        return interp_filter

#import data
df = pd.read_csv('Absorption_database.csv', index_col='Material').drop_duplicates()
X = df.copy().__array__()

unit_circle = True

# train PCA and tranform set
pca = PCA(n_components = 2, whiten=True)
X_r = pca.fit(X).transform(X)

x_in = X_r[0,0]
y_in = X_r[0,1]

if unit_circle:
    # create polygon of the convex hull of the PCA points
    hull = ConvexHull(X_r)
    ply = Polygon(X_r[hull.vertices])

    #for each point map it to a unit circle with scaling relative to the convex hull
    for idx, point in enumerate(X_r):
        line = LineString([[0, 0], point * 100]) #extend line from [0,0] to point
        if line.intersects(ply) and not line.within(ply):
            intersection = line.intersection(ply)
            X_r[idx] = (1 / intersection.length) * point

#
# example of filter extraction with interpolation
#

hull = ConvexHull(X_r)
ply = Polygon(X_r[hull.vertices])
tri = Delaunay(X_r)

coeff = get_filter_absorption(x_in, y_in, tri, X, ply=ply, unit_circle=unit_circle)

pass
