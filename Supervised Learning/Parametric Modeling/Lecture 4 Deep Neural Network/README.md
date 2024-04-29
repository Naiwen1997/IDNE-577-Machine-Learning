# Deep Neural Network

![image](https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/MLP.png)

## Overview
This repository contains two main tasks showcasing the implementation of a Deep Neural Network. Neural networks were a major area of research in both neuroscience and computer science until 1969, when, according to computer science lore, they were killed off by the MIT mathematicians Marvin Minsky and Seymour Papert, who a year later would become co-directors of the new MIT Artificial Intelligence Laboratory.

The technique then enjoyed a resurgence in the 1980s, fell into eclipse again in the first decade of the new century, and has returned like gangbusters in the second, fueled largely by the increased processing power of graphics chips.

## Math Foundation
The mathematical formulation of an MLP can be described as follows. Let $X$ be the input to the network, and let $Y$ be the desired output. The network consists of $L$ layers, where the first layer is the input layer, the last layer is the output layer, and the remaining layers are hidden layers. The output of the $jth$ node in the $ith$ layer is denoted by $h_i^j$. The weights connecting the $jth$ node in layer $i$ to the $kth$ node in layer $i+1$ are denoted by $w_i^j,k$, and the biases of the $kth$ node in layer $i+1$ are denoted by $b_i^k$.

To compute the output of the network, we first apply the following transformation to the input:

$h_1^j = x_j$

For each subsequent layer $i$, we compute the weighted sum of the outputs of the previous layer, and apply a non-linear activation function $g_i$ to obtain the output of each node:

$h_i^j = g_i(\sum_{k=1}^{n_{i-1}}w_{i-1}^k,jh_{i-1}^k + b_{i-1}^j)$

where $n_{i-1}$ is the number of nodes in layer $i-1$, and $j$ and $k$ are indices over the nodes in layer $i$ and layer $i-1$, respectively.

The output of the network is given by the output of the final layer:

$y_k = h_L^k$

The parameters of the network, namely the weights and biases, are learned by minimizing a loss function. A common choice of loss function is the mean squared error (MSE), which is defined as:

$MSE(Y, \hat{Y}) = \frac{1}{N}\sum_{i=1}^N\sum_{k=1}^K (y_i^k - \hat{y_i^k})^2$

where $N$ is the number of samples in the dataset, $K$ is the number of output classes, $y_i^k$ is the true label of the $ith$ sample for the $kth$ class, and $\hat{y_i^k}$ is the predicted label of the $ith$ sample for the $kth$ class.

The parameters of the network are updated using stochastic gradient descent (SGD), which involves computing the gradient of the loss function with respect to the parameters and updating the parameters in the opposite direction of the gradient. The update equation for the weights and biases are given by:

$w_{i}^j,k \leftarrow w_{i}^j,k - \alpha\frac{\partial L}{\partial w_{i}^j,k}$

$b_{i}^k \leftarrow b_{i}^k - \alpha\frac{\partial L}{\partial b_{i}^k}$

where $\alpha$ is the learning rate, and $\frac{\partial L}{\partial w_{i}^j,k}$ and $\frac{\partial L}{\partial b_{i}^k}$ are the gradients of the loss function with respect to the weights and biases, respectively, which can be computed using backpropagation, a recursive algorithm for computing the gradients of the loss function with respect to the parameters.

The backpropagation algorithm starts by computing the gradient of the loss function with respect to the output layer:

$\frac{\partial L}{\partial h_L^k} = \frac{1}{N}(y_i^k - \hat{y_i^k})$

For each subsequent layer $i$ in reverse order, we compute the gradients of the outputs with respect to the weighted sum and the gradients of the weighted sum with respect to the weights and biases:

$\frac{\partial h_i^j}{\partial z_i^j} = g_i'(z_i^j)$

$\frac{\partial z_i^j}{\partial w_{i}^j,k} = h_{i-1}^k$

$\frac{\partial z_i^j}{\partial b_{i}^k} = 1$

where $g_i'$ is the derivative of the activation function $g_i$.

Finally, we compute the gradient of the loss function with respect to the weights and biases using the chain rule:

$\frac{\partial L}{\partial w_{i}^j,k} = \frac{\partial L}{\partial h_i^j}\frac{\partial h_i^j}{\partial z_i^j}\frac{\partial z_i^j}{\partial w_{i}^j,k}$

$\frac{\partial L}{\partial b_{i}^k} = \frac{\partial L}{\partial h_i^j}\frac{\partial h_i^j}{\partial z_i^j}\frac{\partial z_i^j}{\partial b_{i}^k}$

Using these gradients, we can update the weights and biases using the SGD update equation mentioned earlier.

By adjusting the hyperparamters such as number of layers, the number of nodes in each layer, and the choice of activation function, we can design networks that can capture complex patterns in the data and achieve high levels of accuracy on various tasks.

## Tasks
1. **Implementing MLP from Scratch**: Learn how to build a multilayer feedforward neural network from the ground up using only basic libraries like NumPy. Dive into the mathematical fundamentals and understand the workings of neural networks in detail.
   
2. **MLP with TensorFlow and Keras**: Utilize high-level frameworks like TensorFlow and Keras to efficiently implement a multilayer perceptron. This task focuses on leveraging modern deep learning libraries to simplify network implementation and training.

## Applications
- 
- 
- 

## Dataset
We use the [Fashion MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist) dataset. Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
