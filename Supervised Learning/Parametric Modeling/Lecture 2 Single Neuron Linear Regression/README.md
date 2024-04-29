# Linear Regression: Single Neuron Model

![Linear Regression Model](https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/linear_regression.jpg)

## Overview
This project demonstrates the implementation of a single neuron linear regression model to predict the compressive strength of concrete, measured in MPa (Mega Pascals), based on various concrete composition features like cement, ash, water, and others. The implementation is crafted from scratch, showcasing the fundamental concepts of linear regression in a neural network-like architecture using Python.

## Gradient Descent
Gradient Descent is an optimization algorithm for finding a local minimum of a differentiable function. Gradient descent in machine learning is simply used to find the values of a function's parameters (coefficients) that minimize a cost function as far as possible.

### Batch Gradient Descent Algorithm
1. For each epoch **do**
2. Calculate the full gradient by finding $\frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial w_1}$ and $\frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial b}$.
3. $w \leftarrow w - \alpha \frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial w_1}$
4. $b \leftarrow b - \alpha \frac{\partial C(w_1, b; \mathbf{X}, y)}{\partial b}$

### Stochastic Gradient Descent Algorithm
1. For each epoch **do**
2. For $i = 1, \dots, N$ **do**
3. Calculate $\frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial w_1}$ and $\frac{\partial C(\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)}))}{\partial b}$.
2. $w \leftarrow w - \alpha \frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial w_1}$
3. $b \leftarrow b - \alpha \frac{\partial C(w_1, b; \mathbf{x}^{(i)}, y^{(i)})}{\partial b}$

Stochastic gradient descent is the most common way work around for this problem, which is to update $w_1$ and $b$ by calculating the gradient with respect to one entry of data at a time. 

<p align="center">
    <img src="https://github.com/Naiwen1997/IDNE-577-Machine-Learning/blob/master/Images/Gradient_Descent.png" width="500">
</p>

## Linear Regression Model
[Linear regression](https://en.wikipedia.org/wiki/Linear_regression) is employed to predict a continuous variable using a linear function defined by weight parameters. In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. Such models are called linear models. Most commonly, the conditional mean of the response given the values of the explanatory variables (or predictors) is assumed to be an [affine function](https://en.wikipedia.org/wiki/Affine_transformation) of those values; less commonly, the conditional median or some other quantile is used. Like all forms of regression analysis, linear regression focuses on the conditional probability distribution of the response given the values of the predictors, rather than on the joint probability distribution of all of these variables, which is the domain of multivariate analysis.

### Mathematical Formulation
The linear model predicts the target variable y  as a linear combination of the input features $\( x_1, x_2, ..., x_n \)$:

$y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + ... + \beta_n x_{ni} + \epsilon_i = X_i^T \beta + \epsilon_i$

where $w_0$ is the bias and $=w_1=$ to $=w_n=$ are the weight coefficients.

### Optimization via Gradient Descent
The model parameters are optimized by minimizing the residual sum of squares (RSS) cost function:

$$J(w) = \frac{1}{2n} \sum_{i=1}^{n} (y_{pred}[i] - y[i])^{2}$$

The gradients of the cost function with respect to the weights are calculated as:

$$\frac{\partial J}{\partial w} = \frac{1}{n} X^{T} (y_{pred} - y)$$

Updates to the model parameters are made using the gradient descent rule:

$$w = w - \alpha \frac{\partial J}{\partial w}$$

where $\alpha$ is the learning rate.

## Application
- Real Estate for property price prediction
- Healthcare for Medical Diagnosis
- Agriculture for Crop Yield Prediction

## Dataset
We use the [Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) dataset, which includes a total of 3810 rice grain's images were taken for the two species, processed and feature inferences were made. 7 morphological features were obtained for each grain of rice.

