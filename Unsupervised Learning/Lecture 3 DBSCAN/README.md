# DBSCAN Clustering Project

Welcome to the DBSCAN Clustering Project repository. This repository is focused on the implementation and exploration of DBSCAN, a popular density-based clustering algorithm that is especially suited to discovering clusters of arbitrary shapes in spatial data with noise.

## Introduction to DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised machine learning algorithm that identifies regions of high density and separates them from regions of low density. Unlike k-means, DBSCAN does not require the user to specify the number of clusters in advance.

### How It Works

DBSCAN groups together closely packed points by identifying core points, border points, and noise:
- **Core Points**: A point with at least `MinPts` within a radius `ε` (epsilon).
- **Border Points**: Less than `MinPts` within `ε`, but in the neighborhood of a core point.
- **Noise**: Points that are neither core nor border points.

### Steps Involved

1. **Identify Core Points**: For each point, if it has more than `MinPts` within `ε`, label it as a core point.
2. **Expand Clusters**: For each core point, if it is not already assigned to a cluster, create a new cluster, then recursively add all directly density-reachable points to this cluster.
3. **Label Noise**: Mark all non-core and non-border points as noise.

### Mathematical Background

The DBSCAN algorithm relies on a density-based notion of clusters which is designed to discover clusters of arbitrary shape. Mathematically, it can be defined as follows:

$$
\text{For each point } P \text{ in dataset } D:
\begin{cases}
\text{if } |N_{\epsilon}(P)| \geq \text{MinPts} & \text{label } P \text{ as core point} \\
\text{if } |N_{\epsilon}(P)| < \text{MinPts} & \text{check if } P \text{ is border point} \\
\text{else} & \text{label } P \text{ as noise}
\end{cases}
$$

where $N_{\epsilon}(P)$ denotes the ε-neighborhood of P.

## Repository Structure

- `src/`: Contains all source code for implementing the DBSCAN algorithm.
- `data/`: Sample datasets on which the DBSCAN algorithm can be applied.
- `examples/`: Jupyter notebooks that demonstrate the application of DBSCAN to different datasets.
- `docs/`: Additional documentation related to DBSCAN.
