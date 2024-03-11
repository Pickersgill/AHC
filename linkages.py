import numpy as np
def distance(p1, p2):
    """
    Calculate euclidian distance between two points
    """
    return np.linalg.norm(p1 - p2)

def centroid_dist(c1, c2):
    """
    Calculate centroid linkage
    """
    return distance(c1.centroid(), c2.centroid())

def average_dist(c1, c2):
    """
    Calculate average linkage
    """
    total = 0
    
    for a in c1.obs:
        for b in c2.obs:
            total += distance(a, b)
    
    return (1 / (len(c1.obs) * len(c2.obs))) * total

def complete_linkage_dist(c1, c2):
    """
    Calculate complete linkage
    """
    max_dist = None
    for a in c1.obs:
        for b in c2.obs:
            d = distance(a, b)
            if max_dist is None or d > max_dist:
                max_dist = d
    return max_dist

# single linkage
def single_linkage_dist(c1, c2):
    """
    Calculate single linkage
    """
    min_dist = None
    for a in c1.obs:
        for b in c2.obs:
            d = distance(a, b)
            if min_dist is None or d < min_dist:
                min_dist = d
    return min_dist