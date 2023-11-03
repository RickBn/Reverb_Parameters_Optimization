import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from wall_coeff_dim_reduction.Constrained_Voronoi_relaxation import voronoi_relaxation
from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import nearest_points


def get_dim_red_model(dim_red_alg: str = 'pca', voronoi: bool = False, inv_interp: bool = False,
                      unit_circle: bool = False, materials_to_exclude: list = [], path=None):
    # df['8000 Hz'] = df['4000 Hz']
    # df['16000 Hz'] = df['4000 Hz']

    if isinstance(path, dict):
        dim_red_mdl = PCA(n_components=2)

        dim_red_mdl.pts_pca = pd.read_csv(os.path.join(os.getcwd(), path['pts_2d']), header=None).values

        dim_red_mdl.original_pts = pd.read_csv(path['pts_original'], header=None).values

        dim_red_mdl.n_components = dim_red_mdl.pts_pca.shape[1]
        dim_red_mdl.n_features_ = dim_red_mdl.original_pts.shape[1]

        dim_red_mdl.type = 'pca'

    else:
        df = pd.read_csv('wall_coeff_dim_reduction/Absorption_database.csv', index_col='Material').drop_duplicates()

        for m in materials_to_exclude:
            df.drop(m, inplace=True)

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

        dim_red_mdl.original_pts = df.values

        if voronoi:

            # voronoi relaxation to obtain a more equally filled space
            vor = voronoi_relaxation(dim_red_mdl.pts_pca, 50)
            dim_red_mdl.pts_pca = vor.points

            tri = Delaunay(dim_red_mdl.pts_pca)

            dim_red_mdl.tri = tri


    if inv_interp or unit_circle:
        # create polygon of the convex hull of the PCA points
        hull = ConvexHull(dim_red_mdl.pts_pca)
        dim_red_mdl.ply = Polygon(dim_red_mdl.pts_pca[hull.vertices])

        if unit_circle:
            # for each point map it to a unit circle with scaling relative to the convex hull
            for idx, point in enumerate(dim_red_mdl.pts_pca):
                line = LineString([[0, 0], point * 100])  # extend line from [0,0] to point
                if line.intersects(dim_red_mdl.ply) and not line.within(dim_red_mdl.ply):
                    intersection = line.intersection(dim_red_mdl.ply)
                    dim_red_mdl.pts_pca[idx] = (1 / intersection.length) * point
                else:
                    pass

            hull = ConvexHull(dim_red_mdl.pts_pca)
            dim_red_mdl.ply = Polygon(dim_red_mdl.pts_pca[hull.vertices])

        dim_red_mdl.tri = Delaunay(dim_red_mdl.pts_pca)

    dim_red_mdl.inv_interp = inv_interp
    dim_red_mdl.unit_circle = unit_circle

    dim_red_mdl.voronoi = voronoi

    # TODO: capire cosa fare con unit circe, se mettere i minimi per restare nel cerchio con coord polari
    x_min = np.min(dim_red_mdl.pts_pca, axis=0)
    x_max = np.max(dim_red_mdl.pts_pca, axis=0)

    dim_red_mdl.x_min = x_min
    dim_red_mdl.x_max = x_max



    return dim_red_mdl


def pca_inverse_interp(dim_red_mdl, pts_dim_red, k_neighbor: int = 2):
    # # Compute distances between the new point and all the points in the PCA space
    # d = cdist(dim_red_mdl.pts_pca, np.expand_dims(np.array(pts_dim_red), axis=0), 'euclidean')
    #
    # # Sort the distances to get the closest points
    # sorted_idx = np.argsort(d, axis=0)
    #
    # # Get the first neighbors
    # neighbors_idx = np.squeeze(sorted_idx[:k_neighbor])
    #
    # weights = 1 / d[neighbors_idx]
    #
    # x_origdim = np.average(dim_red_mdl.original_pts[neighbors_idx], axis=0, weights=np.squeeze(weights))

    x_origdim = np.zeros(dim_red_mdl.n_features_)

    x = pts_dim_red[0]
    y = pts_dim_red[1]

    simplex = dim_red_mdl.tri.find_simplex([x, y])

    # If the point is outside the polygon, then replace it with the nearest point on the polygon
    if simplex == -1:
        point_in_polygon = nearest_points(dim_red_mdl.ply, Point(x, y))[0]
        simplex = dim_red_mdl.tri.find_simplex([point_in_polygon.x, point_in_polygon.y])

    b = dim_red_mdl.tri.transform[simplex, :2].dot(np.transpose([[x, y]] - dim_red_mdl.tri.transform[simplex, 2]))
    weights = np.c_[np.transpose(b), 1 - b.sum(axis=0)]
    x_origdim = (dim_red_mdl.original_pts[dim_red_mdl.tri.simplices[simplex]] * np.transpose(weights)).sum(axis=0)

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
                if dim_red_mdl.inv_interp:
                    # PCA inverse through interpolation
                    reconstr_params = pca_inverse_interp(dim_red_mdl, params_dim_red[n:n + dim_red_mdl.n_components])

                else:
                    # PCA inverse through sklearn method
                    reconstr_params = dim_red_mdl.inverse_transform(params_dim_red[n:n + dim_red_mdl.n_components])

        reconstr_params = np.clip(reconstr_params, min_rec_val, max_rec_val)

        # # Force last 3 bands equal
        # reconstr_params = np.hstack([reconstr_params, reconstr_params[-1], reconstr_params[-1]])

        params = params + list(reconstr_params)
        # params = params + [params[-1]] * (n_wall_bands - dim_red_mdl.n_features_)

    return params
