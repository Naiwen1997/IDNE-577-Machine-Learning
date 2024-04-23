# Single Neuron Logistic Regression

![image](https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/logistic.jpg)

# Logistic Regression: Single Neuron Model

## Overview
Logistic Regression is a fundamental statistical and machine learning technique used for binary classification tasks. Unlike linear regression, logistic regression outputs probabilities, thanks to the logistic (sigmoid) function, which maps predictions to a probability curve. This repository provides a custom implementation of logistic regression modeled as a single neuron in a neural network, often referred to as a logistic neuron.

## Key Concepts

### The Sigmoid Activation Function
<p align="center">
    <img src="https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/general_single_neuron.png" width="500">
</p>
The sigmoid function, denoted as \(\sigma(z)\), is crucial for logistic regression as it maps any real-valued number into the (0, 1) range, which can be interpreted as a probability:

$ \sigma(z) = \frac{1}{1 + e^{-z}} $

This function is smooth and differentiable, which makes it suitable for gradient-based optimization methods.

### Binary Cross Entropy Loss
The loss function used in logistic regression is the Binary Cross Entropy Loss, which measures the "distance" between the model's predicted probabilities and the actual class outputs (0 or 1):

\[ L(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \Big[y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)}) \log (1 - \hat{y}^{(i)})\Big] \]

The goal is to minimize this loss function through the training process, refining the model's weights and bias toward better accuracy.

### Gradient Calculation
Optimization is performed via stochastic gradient descent (SGD), where gradients of the loss function with respect to model parameters are computed to update the parameters:

\[ \frac{\partial C}{\partial w_1} = (\hat{y}^{(i)} - y^{(i)}) x^{(i)} \]

\[ \frac{\partial C}{\partial b} = (\hat{y}^{(i)} - y^{(i)}) \]

These gradients help in nudging the weights and bias of the neuron in the direction that reduces the loss, gradually improving model performance on the training data.

## Implementation
This repository includes Python code for building and training a logistic regression model using the single neuron abstraction. The model is implemented from scratch, giving detailed insights into the workings of logistic regression and gradient-based learning.

### Usage
Instructions to run the model training and evaluation are provided in separate files, along with examples and explanations.

### Contributing
Contributions to improve the model or extend its functionality are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

