# Agglomerative Heirarchical Clustering in Python

[AHC](https://en.wikipedia.org/wiki/Hierarchical_clustering) is a machine learning technique for finding clusters in a dataset.

## Briefly, what is AHC
I think AHC is best explained by breaking down the individual terms back to front:
+ Clustering - the objective is to take a dataset and break it into disjoint subsets such that elements of the same set are "similar"
+ Hierarchical - the resulting clusters exists in a tree-like structure, where a cluster may be made up of many other clusters
+ Agglomerative - we begin by assuming that every individual datapoint is one cluster, and then merge them one by one until we make one super-cluster

> [!NOTE]
> The inverse approach, divisive clustering, begins with one super-cluster and decomposes it into smaller and smaller clusters. e.g. [Entropy Discretization](https://github.com/Pickersgill/EntropyDiscretization)

At each step the two "nearest" clusters are merged. Nearness is determined by a distance or linkage function. In this implementation the following linkage functions are demonstrated:
+ Average Linkage, the mean distance between points in each cluster
+ Centroid Linkage, the distance between the mean of each cluster (subtly different to Average Linkage)
+ Complete Linkage, the maximum distance between points in each cluster
+ Single Linkage, the minimum distance between points in each cluster