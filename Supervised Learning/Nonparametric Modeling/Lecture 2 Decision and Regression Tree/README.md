# Decision Tree Algorithm Overview

![Decision_Tree]((https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/KNN.png)

## Dataset
We use the [Wine dataset](https://archive.ics.uci.edu/dataset/109/wine) which  are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.

A Decision Tree is a versatile supervised learning algorithm used for both classification and regression tasks. It constructs a tree-like model of decisions and their possible consequences, including chances of event outcomes, resource costs, and utility.

## Concept

The [decision tree algorithm](https://scikit-learn.org/stable/modules/tree.html) involves splitting data into subsets based on feature values, continuously partitioning until a stopping criterion is met. It uses a measure known as information gain to determine which feature to split on at each step, seeking to maximize the predictability and minimize entropy.

## How It Works

1. **Start at the root node**: Include all samples.
2. **Feature selection**: Choose the feature that provides the highest information gain.
3. **Node splitting**: Create child nodes and distribute the data based on feature values.
4. **Recursion**: Repeat the process for each child until a leaf node is reached.
5. **Stopping criteria**: Stop when maximum depth is reached or minimum data at a node is too small.

## Tree Structure

- **Internal nodes**: Each represents a decision rule on a feature.
- **Branches**: Correspond to the outcome of the test.
- **Leaf nodes**: Represent the prediction outcome (a class label or a regression value).

## Mathematical Formulation

### Decision Tree Algorithm - Gini Impurity and Entropy

Decision Trees often use criteria like Gini impurity and entropy to determine the best points to split the data. These measures help assess the quality of a potential split, with the goal of maximizing the homogeneity of the resulting subsets.

### Gini Impurity

[Gini](https://en.wikipedia.org/wiki/Gini_coefficient) impurity measures the impurity of a dataset; it's a metric that quantifies how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. The Gini Impurity of a dataset can be calculated using the following equation:

$$
\text{Gini}(D) = 1 - \sum_{i=1}^k p_i^2
$$

Where:
- D is the dataset,
- $p_i$ is the probability of an object being classified to a particular class.

A Gini impurity of 0 indicates perfect homogeneity, where all elements belong to a single class.

## Entropy

[Entropy](https://en.wikipedia.org/wiki/Entropy), a concept borrowed from information theory, measures the level of uncertainty or impurity in a group of examples. The entropy of a dataset is defined as:

$$
\text{Entropy}(D) = -\sum_{i=1}^k p_i \log_2 p_i
$$

Where:
- D is the dataset,
- $p_i$ is the proportion of the elements that belong to class i in the dataset.

Entropy will be zero when all cases in the node fall into a single target category, indicating no uncertainty or impurity.

### Choosing Between Gini Impurity and Entropy

Both Gini impurity and entropy are used to compute the homogeneity of the labels in the dataset. While they are very similar, they differ slightly in how they penalize changes in probability:
- Entropy tends to be more computationally intensive because it involves logarithmic calculations.
- Gini impurity is generally faster to compute, so it is the default choice in many implementations of decision trees, like Scikit-learn's `DecisionTreeClassifier`.

Understanding these metrics is crucial for tuning decision trees and understanding their decision-making process.

## Advantages and Disadvantages

- **Pros**: Intuitive, handles both numerical and categorical data, models non-linear relationships.
- **Cons**: Prone to overfitting, sensitive to noisy data, may not handle correlated features well.

## Application
- Customer Relationship Management
- Credit Scoring in Financial Institution
- Fraud Detection in Insurance

## Dataset
We use the [Wine dataset](https://archive.ics.uci.edu/dataset/109/wine) which  are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.