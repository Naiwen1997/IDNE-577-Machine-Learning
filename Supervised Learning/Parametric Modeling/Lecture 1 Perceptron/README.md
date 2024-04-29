# Perceptron - A Binary Linear Classifier

![Perceptron](https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/perceptron.jpg)

Welcome to the Perceptron repository! This project is dedicated to the implementation and exploration of the Perceptron, one of the simplest types of artificial neural networks and a foundational element in the field of machine learning.

## Overview
The [Perceptron](https://www.w3schools.com/ai/ai_perceptrons.asp) is a type of supervised learning algorithm developed by Frank Rosenblatt in 1957. It is primarily used as a binary classifier, which means it can categorize new inputs into one of two classes. The Perceptron makes its classifications based on a linear predictor function combining a set of weights with the feature vector.

## How It Works

### Data Processing
To prepare data, we must first determine the two output groups and classify data in numerical terms, at which point the data will be in a format that will allow the perceptron to be trained. For the input features, we also need to normalize their values to make sure they are in the same scale 0 to 1.

### Prediction of Results
![image](https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/step_function.png)

The Perceptron model functions on the basic principle of a linear equation:
$$Z=φ(w_1x_1+w_2x_2+\ldots + w_nx_n+b)$$
where:

- $x_1$, $x_2$, $\ldots$, $x_n$ are the input features
- $w_1$, $w_2$, $\ldots$, $w_n$ are the corresponding weights for each input feature
- $b$ is the bias weight
- $φ$ is the [step function](https://en.wikipedia.org/wiki/Step_function): The step function is the activation function which returns 1 if the weighted sum of the inputs and bias is greater than or equal to 0, and 0 otherwise. In other words, the perceptron outputs a 1 if the input falls on one side of the decision boundary and a 0 if it falls on the other side. We can aslo use other functions such as [sign function](https://en.wikipedia.org/wiki/Sign_function) or [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) as the acitvation function.

### Weight Update
The perceptron learns by adjusting the weights based on the error between the predicted output and the true output. The weights are updated using the following formula:

$$w_i = w_i + \mathrm{learning_-rate} \times (target - output) \times x_i$$

where:

- $w_i$ is the weight for the current input feature $x_i$
- $learning_-rate$ is the hyperparameter that determines the step size for adjusting the weights
- $target$ is the desired output for the given input
- $output$ is the actual output produced by the perceptron for the given input
- $x_i$ is the value of the current input feature being considered for weight update

## Applications
Perceptrons are useful in fields such as:
- [Optical character recognition](https://en.wikipedia.org/wiki/Optical_character_recognition)
- Speech recognition
- Decision-making processes

## Dataset
We use the [Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) dataset, which includes a total of 3810 rice grain's images were taken for the two species, processed and feature inferences were made. 7 morphological features were obtained for each grain of rice.
