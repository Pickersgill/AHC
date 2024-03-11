import numpy as np
import sys
from linkages import *

# A "cluster" class, forms the basis of dendograms
class Cluster:
    """
    Cluster forming the basis of a dendogram. Each cluster is a collection of datapoints.
    """
    def __init__(self, height, obs, right=None, left=None, is_leaf=False):
        self.height = height # "split" height
        self.obs = obs # constituent datapoints
        self.right = right
        self.left = left
        self.is_leaf = is_leaf
    
    def centroid(self):
        """
        Calculate cluster centroid, average datapoint
        """
        return np.mean(self.obs, axis=0)
    
    def quality(self):
        """ 
        Measure quality of this cluster as the mean distance from centre
        """
        
        c = self.centroid()
        total = 0
        for o in self.obs:
            total += np.linalg.norm(c - o)

        return (1 / len(self.obs)) * total
    
    def get_all_heights(self):
        """
        Get a list of heights for all direct children of this cluster
        Used to determine possible split heights
        """
        heights = [self.height]
        if self.left:
            heights += self.left.get_all_heights()
        if self.right:
            heights += self.right.get_all_heights()
        return heights
        
    def split(self, h):
        """
        Recursively gather clusters to return for a given split height
        """
        if self.height <= h:
            return [self]
        clusts = []

        if self.left:
            l_s = self.left.split(h)
            for c in l_s:
                clusts.append(c)
        if self.right:
            r_s = self.right.split(h)
            for c in r_s:
                clusts.append(c)

        return clusts

    def __repr__(self):
        s = ""
        if self.left and self.right:
            s += "\n" + repr(self.left)
            s += "\n" + repr(self.right)
        s += "\n" + str(self)

    def __str__(self):
        return f"<nObs: {len(self.obs)}, height: {self.height}, left: {bool(self.left)}, right: {bool(self.right)}>"


def do_clustering(data, dist_func):
    """
    Construct initial unit clusters and start the process of merging clusters based on the given distance function
    """
    clusters = np.array([Cluster(0, np.array([r]), is_leaf=True) for r in data])
    root = do_merges(clusters, dist_func)[0]
    return root

def merge(clusters, i, j, dist):
    """
    Take a set of clusters and produce the new cluster set after enacting merge of cluster i and j
    """
    c1 = clusters[i]
    c2 = clusters[j]
    mergd = np.concatenate([c1.obs, c2.obs])
    clusters = np.delete(clusters, [i, j])
    new_clust = Cluster(dist, mergd, right=c2, left=c1)
    return np.append(clusters, (new_clust))

def do_merges(clusters, dist_func, comp=0):
    """
    Recursively perform cluster merges based on minimum distance until root node is constructed (Dendogram is complete)
    """
    msg = f"Completed {comp} merges..."
    sys.stdout.write(msg + "\b" * len(msg)) # pretty progress readout
    sys.stdout.flush()
    
    if len(clusters) == 1: # base case, we have constructed the root
        return clusters

    size = len(clusters)    

    closest_dist = None
    c1_ind = None
    c2_ind = None

    for i in range(size):
        c1 = clusters[i]
        for j in range(size):
            c2 = clusters[j]
            if i != j:
                dist = dist_func(c1, c2)
                if closest_dist is None or dist < closest_dist: # d(i,j) - d(j, i) = 0 may not be the case for all distance functions
                    closest_dist = dist
                    c1_ind = i
                    c2_ind = j
    
                
    
    new_clusters = merge(clusters, c1_ind, c2_ind, closest_dist) # use merge function to update data structure based on discovered best merge
    
    return do_merges(new_clusters, dist_func, comp+1)
    

