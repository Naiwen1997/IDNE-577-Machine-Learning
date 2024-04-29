# Deep Neural Network

![image](https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/MLP.png)

## Overview
This repository contains two main tasks showcasing the implementation of a [Deep Neural Network](https://www.tutorialspoint.com/python_deep_learning/python_deep_learning_deep_neural_networks.htm). Neural networks were a major area of research in both neuroscience and computer science until 1969, when, according to computer science lore, they were killed off by the MIT mathematicians Marvin Minsky and Seymour Papert, who a year later would become co-directors of the new MIT Artificial Intelligence Laboratory.

The technique then enjoyed a resurgence in the 1980s, fell into eclipse again in the first decade of the new century, and has returned like gangbusters in the second, fueled largely by the increased processing power of graphics chips.

## Math Foundation

### Forward Propagation
Each neuron in a layer receives input from the previous layer, which it sums into a linear combination followed by a non-linear activation:
   - **Linear Combination:** $z = w^T x + b$
   - **Activation:** $a = f(z)$

Where $w$, $x$, and $b$ represent weights, input vectors, and bias, respectively, and $f$ denotes a non-linear activation function like sigmoid or ReLU.

### Backpropagation

To learn model parameters, MLPs use backpropagation, which involves:
   1. **Loss Calculation:** Using a function like Mean Squared Error:
      $L = \frac{1}{2} \sum (y_{pred} - y_{true})^2$
   2. **Gradient Calculation:** The gradient of the loss function with respect to the weights is computed using the chain rule:
      $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$
   3. **Weight Update:** Weights are updated in the direction that minimizes the loss:
      $w = w - \eta \cdot \frac{\partial L}{\partial w}$
   Where $\eta$ is the learning rate.


## Tasks
1. **Implementing by Basic Class**: Discover how to construct a multilayer feedforward neural network from scratch using fundamental libraries like NumPy. Explore the essential mathematical concepts and delve deep into the detailed mechanics of how neural networks function.
   
2. **Implementing by TensorFlow**: "Employ advanced frameworks such as TensorFlow to streamline the implementation and training of a multilayer perceptron. This approach emphasizes using cutting-edge deep learning libraries to simplify the development of neural networks.

## Applications
- Image Recognition and Processing
- [Natural Language Processing](https://www.ibm.com/topics/natural-language-processing)
- Speech Recognition

## Dataset
We use the [Fashion MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist) dataset. Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
