# Deep Neural Network

## Overview
This repository contains two main tasks showcasing the implementation of a Deep Neural Network. Neural networks were a major area of research in both neuroscience and computer science until 1969, when, according to computer science lore, they were killed off by the MIT mathematicians Marvin Minsky and Seymour Papert, who a year later would become co-directors of the new MIT Artificial Intelligence Laboratory.

The technique then enjoyed a resurgence in the 1980s, fell into eclipse again in the first decade of the new century, and has returned like gangbusters in the second, fueled largely by the increased processing power of graphics chips.

## Multilayer Perceptron (MLP) - Mathematical Overview

## Math Foundation
### Network Architecture
An MLP consists of multiple layers, each composed of neurons. The input to the network is denoted by \( X \), which passes through several hidden layers before producing an output \( Y \). The network's goal is to approximate the true function mapping inputs to outputs.

### Forward Propagation
The forward pass computes the output of each neuron using:

1. **Linear Combination**: $z_i^j = \sum_{k=1}^{n_{i-1}} w_{i-1}^{k,j} h_{i-1}^k + b_i^j$, where \( w \) are weights, \( b \) are biases, and \( h \) are outputs from the previous layer.
2. **Activation Function**: $h_i^j = g_i(z_i^j)$, applying a non-linear transformation.

### Loss Function
To train the network, a loss function \( L \) quantifies the error between the predicted outputs $\hat{Y}$ and the actual labels \( Y \). Common choices are Mean Squared Error for regression and Cross-Entropy for classification.

### Backpropagation
Backpropagation helps optimize the weights and biases by computing gradients:

1. **Gradient Calculation**: For each weight and bias, compute the gradient of the loss function using the chain rule.
2. **Parameter Update**: Adjust the weights and biases by a small step $\alpha$ in the direction that minimally decreases the loss,$w_{i}^{j,k} \leftarrow w_{i}^{j,k} - \alpha \frac{\partial L}{\partial w_{i}^{j,k}}$.

### Conclusion
By iteratively updating its parameters through backpropagation, an MLP learns to reduce the error between its predicted outputs and the true data. The architecture's depth and the choice of activation function can significantly influence its learning capacity and performance on various tasks.



## Tasks
1. **Implementing MLP from Scratch**: Learn how to build a multilayer feedforward neural network from the ground up using only basic libraries like NumPy. Dive into the mathematical fundamentals and understand the workings of neural networks in detail.
   
2. **MLP with TensorFlow and Keras**: Utilize high-level frameworks like TensorFlow and Keras to efficiently implement a multilayer perceptron. This task focuses on leveraging modern deep learning libraries to simplify network implementation and training.

## Applications
- 
- 
- 

## Dataset
We use the [Fashion MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist) dataset. Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
