import numpy as np
import cluster
from matplotlib import pyplot as plt

# default linkage type
LINK_TYPE = cluster.complete_linkage_dist

def load_data(source):
    """
    Build data from source
    """
    rows = []
    with open(source, newline="") as data:
        for r in data:
            rows.append(np.array([float(x) for x in r.split()]))
    return np.array(rows)

def get_clusters(dendo, k):
    """
    Search for and take a cut from the dendogram which produces k clusters
    """
    hs = np.sort(np.unique(dendo.get_all_heights()))[::-1] # retrieve and (descending) sort discrete list of possible splits
    
    for h in hs:
        clusts = dendo.split(h)
        if len(clusts) == k: # try each potential split and if split found produces k clusters, return it
            print("Found cut for %d clusters at h=%d" % (k, h))
            return clusts

    # Some values of k will not have a valid split
    # To circumvent this (without causing strange dilations in the output graph) the "next best" cluster number is returned
    print("NO SPLIT FOUND, TRYING K+1")
    
    return get_clusters(dendo, k+1)

def make_series(dist_func, ks, data):
    """
    Builds a series of quality scores for different cluster numbers based on given dist_funct
    """
    print("Computing series based on distance function: ", dist_func.__name__)
    dendo = cluster.do_clustering(data, dist_func)
    s = []
    for k in ks:
        splits = get_clusters(dendo, k)
        qual = np.mean([s.quality() for s in splits])
        s.append(qual)

    return s
        

if __name__ == "__main__":
    data = load_data("./ncidata.txt").transpose()   
    print("Loaded %d rows..." % len(data))

    k_range = list(range(1, 16))

    comp_series = make_series(cluster.complete_linkage_dist, k_range, data)
    single_series = make_series(cluster.single_linkage_dist, k_range, data)
    avg_series = make_series(cluster.average_dist, k_range, data)
    cent_series = make_series(cluster.centroid_dist, k_range, data)

    plt.plot(k_range, comp_series, label="Complete Link.")
    plt.plot(k_range, single_series, label="Single Link.")
    plt.plot(k_range, avg_series, label="Average Link.")
    plt.plot(k_range, cent_series, label="Centroid Link.")

    plt.xlabel("k (Number of Clusters)")
    plt.ylabel("Quality Score")

    plt.legend()
    plt.show()
    print(comp_series)
        

