# K-Means Clustering

This repository is dedicated to exploring the K-Means clustering algorithm, a popular unsupervised machine learning technique used to identify clusters within an unlabeled dataset. 

## Introduction to K-Means Clustering

K-Means clustering aims to partition `n` observations into `k` clusters in which each observation belongs to the cluster with the nearest mean. This results in a partitioning of the data space into Voronoi cells. K-Means is best suited for situations where the clusters have a spherical-like distribution.

### How It Works

1. **Initialization**: Start by selecting `k` initial centroids randomly.
2. **Assignment**: Assign each data point to the nearest centroid.
3. **Update**: Recalculate the centroids as the mean of the assigned points.
4. **Repeat**: Repeat the assignment and update steps until convergence (i.e., when assignments no longer change).

### Mathematical Background

The objective of K-Means is to minimize the within-cluster sum of squares (WCSS), which is given by:

$$
\text{WCSS} = \sum_{i=1}^k \sum_{x \in S_i} ||x - \mu_i||^2
$$

where $\mu_i$ is the mean of points in $S_i$.

## Repository Structure

- `data/`: Directory for datasets used in clustering examples.
- `src/`: Contains the source code for the K-Means clustering implementations.
- `notebooks/`: Jupyter notebooks with examples and visualizations of K-Means clustering.
- `docs/`: Additional documentation and references related to the K-Means algorithm.