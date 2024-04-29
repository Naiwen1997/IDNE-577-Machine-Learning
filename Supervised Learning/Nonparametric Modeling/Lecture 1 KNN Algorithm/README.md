# K-Nearest Neighbors (KNN)

![Linear Regression Model](https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/KNN.png)

## Overview
The [k-nearest neighbors (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) algorithm is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.

## How K-Nearest Neighbors (KNN) Works

The K-Nearest Neighbors algorithm (KNN) is a simple, yet powerful machine learning technique used for both classification and regression. Below is a breakdown of how it operates:

### Steps to Implement KNN

1. **Data Preparation**
   - Clean the dataset by removing missing values.
   - Scale the features to ensure they are on the same scale which is crucial for distance calculation.

2. **Choose the Value of K**
   - Select the number of neighbors k. A smaller k makes the model sensitive to noise, whereas a larger k makes it computationally expensive and may include features that are irrelevant.

3. **Calculate Distances**
   - Compute the distance between the query instance and all the training samples using a suitable distance metric.

4. **Find K-Nearest Neighbors**
   - Identify the k training samples closest to the query instance based on the computed distances.

5. **Decision Rule**
   - **Classification**: Determine the output class by majority vote among the k neighbors.
   - **Regression**: Predict the output by averaging the values of the k neighbors.

6. **Evaluate the Model**
   - Assess the model’s performance using appropriate metrics such as accuracy for classification or mean squared error for regression.

### Distance Metrics

Distance calculation plays a crucial role in the KNN algorithm. Below are the formulas for the most commonly used distance metrics:

### Euclidean Distance: 

$$
\text{Euclidean}(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
$$

### Manhattan Distance

Used for calculating the distance in grid-like paths:

$$
\text{Manhattan}(x, y) = \sum_{i=1}^n |x_i - y_i|
$$

### Minkowski Distance

A generalized metric that includes others as special cases:

$$
\text{Minkowski}(x, y, p) = \left(\sum_{i=1}^n |x_i - y_i|^p\right)^{1/p}
$$

Where `p` is the order parameter: `p=1` yields Manhattan distance, and `p=2` yields Euclidean distance.

## Application
- Real Estate for property price prediction
- Healthcare for Medical Diagnosis
- Agriculture for Crop Yield Prediction

## Dataset
We use the [Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) dataset, which includes a total of 3810 rice grain's images were taken for the two species, processed and feature inferences were made. 7 morphological features were obtained for each grain of rice.