from scipy.spatial import Voronoi
import numpy as np
from shapely.geometry import Polygon, box

def voronoi_relaxation(points, iter = 50):

    voronoi = Voronoi(points, qhull_options='Qc Qx', incremental= True)
    domain = get_domain(points)
    boundary = box(domain['x']['min'], domain['y']['min'], domain['x']['max'], domain['y']['max'])

    for i in range(0, iter):
        voronoi = relax(voronoi, boundary)

    return voronoi


def relax(voronoi, boundary):

    centroids = []

    for idx in voronoi.point_region:

        region = [i for i in voronoi.regions[idx] if i != -1]

        if len(region) > 2:
            polygon = get_intersecting_area(voronoi.vertices[region], boundary)
            centroids.append([polygon.centroid.x, polygon.centroid.y])
        else:
            centroids.append(voronoi.points[np.where(voronoi.point_region == idx)][0])
        
    return Voronoi(centroids, qhull_options='Qbb Qc Qx')



def get_domain(arr):

    x = arr[:, 0]
    y = arr[:, 1]
    return {
        'x': {
            'min': min(x),
            'max': max(x),
        },
        'y': {
            'min': min(y),
            'max': max(y),
        }
    }

def get_intersecting_area(verts, boundary):

    polygon = Polygon(verts)

    if(polygon.intersects(boundary)):
        return polygon.intersection(boundary)

    return polygon