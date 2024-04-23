# Logistic Regression: Single Neuron Model

![image](https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/logistic.jpg)

## Overview
[Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression#:~:text=Logistic%20regression%20is%20a%20supervised,based%20on%20patient%20test%20results.) is a fundamental statistical and machine learning technique used for binary classification tasks. Unlike linear regression, logistic regression outputs probabilities, thanks to the logistic (sigmoid) function, which maps predictions to a probability curve. This repository provides a custom implementation of logistic regression modeled as a single neuron in a neural network, often referred to as a logistic neuron.

## Key Concepts

### The Sigmoid Activation Function
The sigmoid function, denoted as \(\sigma(z)\), is crucial for logistic regression as it maps any real-valued number into the (0, 1) range, which can be interpreted as a probability:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

This function is smooth and differentiable, which makes it suitable for gradient-based optimization methods.

### Binary Cross Entropy Loss
The loss function used in logistic regression is the [Binary Cross Entropy Loss](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a), which measures the "distance" between the model's predicted probabilities and the actual class outputs (0 or 1):

$$L(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \Big[y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)}) \log (1 - \hat{y}^{(i)})\Big]$$

The goal is to minimize this loss function through the training process, refining the model's weights and bias toward better accuracy.

### Gradient Calculation
Optimization is performed via stochastic gradient descent (SGD), where gradients of the loss function with respect to model parameters are computed to update the parameters:

$$\frac{\partial C}{\partial w_1} = (\hat{y}^{(i)} - y^{(i)}) x^{(i)}$$

$$\frac{\partial C}{\partial b} = (\hat{y}^{(i)} - y^{(i)})$$

These gradients help in nudging the weights and bias of the neuron in the direction that reduces the loss, gradually improving model performance on the training data.

## Applications
- Customer Churn Prediction
- Targeted Advertising
- Predicting Patient Outcomes

## Dataset
We use the [Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) dataset, which includes a total of 3810 rice grain's images were taken for the two species, processed and feature inferences were made. 7 morphological features were obtained for each grain of rice.

