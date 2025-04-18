
# Loss Functions 

## negative log entropy

This loss measures the uncertainty within a single probability distribution by penalizing uniform predictions.

## cross entropy

Cross entropy quantifies the difference between two probability distributions by measuring how well 𝑞  approximates the true distribution 𝑝


## KL divergence

KL divergence measures how much information is lost when using distribution 𝑞
 to approximate the true distribution 𝑝.



## MSE

MSE calculates the average of the squared differences between the predicted values and the actual values

$$\sum_{i=1}^{D}(x_i-y_i)^2$$



# Non Linear Activations Functions


## arc tangent

The arctangent activation squashes inputs to a range between -pi/2 and pi/2 with smooth gradients

## sigmoid
The sigmoid function maps inputs to the range (0,1), often used for binary classification.

$$\sigma(z) = \frac{1} {1 + e^{-z}}$$

## ReLU

ReLU is a piecewise linear function that outputs the input if it's positive and zero otherwise — fast and widely used

$$Relu(z) = max(0, z)$$

## eLU

ELU improves ReLU by allowing negative outputs, helping the network learn with small negative values.

## SeLU

SELU is a self-normalizing activation function that scales ELU to maintain mean and variance across layers.

## Sofmax

Softmax turns a vector of values into a probability distribution — great for multi-class classification

