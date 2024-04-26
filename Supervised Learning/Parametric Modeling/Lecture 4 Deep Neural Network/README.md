# Deep Neural Network

# Fashion MNIST Classification with Multilayer Perceptron (MLP)

## Overview
This repository contains two main tasks showcasing the implementation of a Multilayer Perceptron (MLP) for classifying images from the Fashion MNIST dataset. Fashion MNIST is a dataset of 70,000 grayscale images, each 28x28 pixels, categorized into 10 different fashion items such as T-shirts/tops, trousers, and sneakers.

## Tasks
1. **Implementing MLP from Scratch**: Learn how to build a multilayer feedforward neural network from the ground up using only basic libraries like NumPy. Dive into the mathematical fundamentals and understand the workings of neural networks in detail. [View Notebook](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/blob/main/Supervised_Learning/5_Deep_Neural_Network/Task1_Deep_NN_scratch.ipynb)
   
2. **MLP with TensorFlow and Keras**: Utilize high-level frameworks like TensorFlow and Keras to efficiently implement a multilayer perceptron. This task focuses on leveraging modern deep learning libraries to simplify network implementation and training. [View Notebook](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/blob/main/Supervised_Learning/5_Deep_Neural_Network/Task2_Deep_NN_Tensorflow.ipynb)

## Dataset
Fashion MNIST is designed as a more challenging replacement for the traditional MNIST digit recognition dataset. More information and visualizations of the dataset can be found in [Task1_Deep_NN_scratch Notebook](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/blob/main/Supervised_Learning/5_Deep_Neural_Network/Task1_Deep_NN_scratch.ipynb).

## MLP Architecture
The repository includes examples of MLPs with different configurations:
- A single hidden layer MLP
- A deeper MLP with two hidden layers

Both architectures are explored in terms of design, implementation, and their effectiveness in classifying complex image data.

## Mathematical Foundation
The MLP models in this project use layers of nodes connected through weights and biases, with non-linear activation functions. The network's learning is guided by backpropagation and optimized using stochastic gradient descent (SGD).

## Dataset
We use the [Fashion MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist) dataset. Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
