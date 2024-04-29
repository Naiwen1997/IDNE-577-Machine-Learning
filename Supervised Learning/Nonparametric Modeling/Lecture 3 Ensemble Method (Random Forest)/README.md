# Ensemble Method

<p align="center">
  <img src="https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/random_forest.png" alt="Ensemble Method" width="800" height="600">
</p>


## Overview
In statistics and machine learning, [ensemble methods](https://en.wikipedia.org/wiki/Ensemble_learning) use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone. Unlike a statistical ensemble in statistical mechanics, which is usually infinite, a machine learning ensemble consists of only a concrete finite set of alternative models, but typically allows for much more flexible structure to exist among those alternatives. 

Ensemble methods help improve machine learning results by combining several models. This approach helps in reducing variance, bias, or improving predictions. Ensemble methods can be broadly divided into two categories:
- **Bagging**: Helps reduce variance and helps avoid overfitting.
- **Boosting**: Reduces both bias and variance by building a series of models that build on each other to improve predictive performance.

## Random Forest

Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random Forests perform a type of ensemble learning called bagging, which involves training each tree on a different sample of the data.

### Random Forest Algorithm

In Random Forest, each tree is trained on a subset of data and features, which is described by the following formula:
- Given a training set $X = x_1, ..., x_n$ with responses $Y = y_1, ..., y_n$, a number of trees in the forest `T`, and the number of features sampled for each split `m`, Random Forest algorithm follows every tree from 1 to T:
    1. Sample `n` examples randomly with replacement from X, Y.
    2. Train a decision tree on this sample (typically with a maximum allowed depth).
    3. At each node:
        - Randomly sample `m` features from all features.
        - Split the node using the feature that provides the best split according to the objective function, typically using Gini impurity or entropy in classification.

## Bagging

Bagging, or Bootstrap Aggregating, is an ensemble technique primarily used to reduce variance in a noisy dataset. Bagging involves creating multiple copies of the original training dataset using random sampling with replacement, training a model on each copy, and then averaging the outputs.

### Bagging Algorithm

The typical steps in a bagging algorithm include:
- Given a dataset with `n` instances:
  1. Randomly select a sample of `n` instances with replacement.
  2. Train a model on these samples.
  3. Repeat the above two steps for `k` times creating `k` models.
  4. Aggregate the results of these models into a single result.

## Conclusion

By combining multiple models, ensemble methods, such as Random Forest and Bagging, are able to achieve higher accuracy than any of the individual models contributing to the ensemble. This repository aims to explore these methods in depth, providing both theoretical explanations and practical implementations.

## Application
- Customer Relationship Management
- Credit Scoring in Financial Institution
- Fraud Detection in Insurance

## Dataset
We use the [Wine dataset](https://archive.ics.uci.edu/dataset/109/wine) which  are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.