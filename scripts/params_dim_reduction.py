import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from wall_coeff_dim_reduction.Constrained_Voronoi_relaxation import voronoi_relaxation
from scipy.spatial import Delaunay

def get_dim_red_model(dim_red_alg: str = 'pca', voronoi: bool = False):
    df = pd.read_csv('wall_coeff_dim_reduction/Absorption_database.csv', index_col='Material').drop_duplicates()

    # df['8000 Hz'] = df['4000 Hz']
    # df['16000 Hz'] = df['4000 Hz']

    # PCA
    if dim_red_alg == 'pca':
        # train PCA and transform set
        dim_red_mdl = PCA(n_components=2)

        dim_red_mdl.fit(df)

        x_dimred = dim_red_mdl.transform(df)

        dim_red_mdl.type = 'pca'

    else:
        x_dimred = df.values

    dim_red_mdl.pts_pca = x_dimred

    if voronoi:

        # voronoi relaxation to obtain a more equally filled space
        vor = voronoi_relaxation(x_dimred, 50)
        x_dimred = vor.points

        tri = Delaunay(x_dimred)

        dim_red_mdl.tri = tri

    dim_red_mdl.original_pts = df.values

    x_min = np.min(x_dimred, axis=0)
    x_max = np.max(x_dimred, axis=0)

    dim_red_mdl.x_min = x_min
    dim_red_mdl.x_max = x_max

    dim_red_mdl.voronoi = voronoi

    return dim_red_mdl

def pca_inverse_interp(dim_red_mdl, x, k_neighbor: int = 2):
    # Compute distances between the new point and all the points in the PCA space
    d = cdist(dim_red_mdl.pts_pca, np.expand_dims(np.array(x), axis=0), 'euclidean')

    # Sort the distances to get the closest points
    sorted_idx = np.argsort(d, axis=0)

    # Get the first neighbors
    neighbors_idx = np.squeeze(sorted_idx[:k_neighbor])

    weights = 1 / d[neighbors_idx]

    x_origdim = np.average(dim_red_mdl.original_pts[neighbors_idx], axis=0, weights=np.squeeze(weights))

    return x_origdim

def reconstruct_original_params(dim_red_mdl, params_dim_red, min_rec_val=0, max_rec_val=1):
    params = []
    for n in range(0, len(params_dim_red), dim_red_mdl.n_components):
        if dim_red_mdl.voronoi:
            params_cur = np.array(params_dim_red[n:n + dim_red_mdl.n_components])

            simplex = dim_red_mdl.tri.find_simplex(params_cur)
            b = dim_red_mdl.tri.transform[simplex, :2].dot(np.transpose(np.expand_dims(params_cur, axis=0) - dim_red_mdl.tri.transform[simplex, 2]))
            weights = np.c_[np.transpose(b), 1 - b.sum(axis=0)]
            reconstr_params = (dim_red_mdl.original_pts[dim_red_mdl.tri.simplices[simplex]] * np.transpose(weights)).sum(axis=0)

        else:
            if dim_red_mdl.type == 'pca':
                # # PCA inverse through sklearn method
                # reconstr_params = dim_red_mdl.inverse_transform(params_dim_red[n:n + dim_red_mdl.n_components])
                # PCA inverse through interpolation
                reconstr_params = pca_inverse_interp(dim_red_mdl, params_dim_red[n:n + dim_red_mdl.n_components])

        reconstr_params = np.clip(reconstr_params, min_rec_val, max_rec_val)

        # # Force last 3 bands equal
        # reconstr_params = np.hstack([reconstr_params, reconstr_params[-1], reconstr_params[-1]])

        params = params + list(reconstr_params)
        # params = params + [params[-1]] * (n_wall_bands - dim_red_mdl.n_features_)

    return params
