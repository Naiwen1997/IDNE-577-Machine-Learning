# Decision Tree Algorithm Overview

A Decision Tree is a versatile supervised learning algorithm used for both classification and regression tasks. It constructs a tree-like model of decisions and their possible consequences, including chances of event outcomes, resource costs, and utility.

## Concept

The [decision tree algorithm](https://scikit-learn.org/stable/modules/tree.html) involves splitting data into subsets based on feature values, continuously partitioning until a stopping criterion is met. It uses a measure known as information gain to determine which feature to split on at each step, seeking to maximize the predictability and minimize entropy.

### How It Works

1. **Start at the root node**: Include all samples.
2. **Feature selection**: Choose the feature that provides the highest information gain.
3. **Node splitting**: Create child nodes and distribute the data based on feature values.
4. **Recursion**: Repeat the process for each child until a leaf node is reached.
5. **Stopping criteria**: Stop when maximum depth is reached or minimum data at a node is too small.

### Tree Structure

- **Internal nodes**: Each represents a decision rule on a feature.
- **Branches**: Correspond to the outcome of the test.
- **Leaf nodes**: Represent the prediction outcome (a class label or a regression value).

### Mathematical Formulation

The decision-making at each node is driven by the criterion of maximizing the information gain, computed as:

$$
\text{Gain}(D, f_i) = \text{Entropy}(D) - \sum_{j=1}^{k} \frac{|D_j|}{|D|} \times \text{Entropy}(D_j)
$$

Where:
- \(D\) is the dataset at the node,
- \(f_i\) is a feature,
- \(D_j\) is the subset of \(D\) for each value \(v_j\) of \(f_i\),
- \(\text{Entropy}(D)\) is given by:
  
  $$
  \text{Entropy}(D) = - \sum_{i=1}^{c} p_i \log_2(p_i)
  $$

  \(p_i\) is the proportion of class \(i\) samples at node \(D\).

## Advantages and Disadvantages

- **Pros**: Intuitive, handles both numerical and categorical data, models non-linear relationships.
- **Cons**: Prone to overfitting, sensitive to noisy data, may not handle correlated features well.

## Application
- Customer Relationship Management
- Credit Scoring in Financial Institution
- Fraud Detection in Insurance

## Dataset
We use the [Wine dataset](https://archive.ics.uci.edu/dataset/109/wine) which  are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.