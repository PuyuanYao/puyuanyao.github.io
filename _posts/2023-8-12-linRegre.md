---
title: Linear Regression
date: 2023-8-12
categories: [regression]
tags: [linear regression]
---
# Linear Regression
Linear regression can be used when we want to approximate a set of data to a line or a hyperplane. In general there are three approches to this method. 
We define dataset and parameters
$$D = \{(x_1, y_1), (x_2, y_2),...,(x_n, y_n)\},$$
$$W = \{w_1, w_2, ..., w_n\}.$$
Therefore the function will be
$$h(x) = \sum_{i=1}^{n}\theta_i x_i.$$
To measure how well a certain line or hyperplane represents the data, we define the loss function
$$L(\theta) = \frac{1}{2}\sum_{i=0}^{n}(h(x_i)-y_i)^2.$$
The goal would be finding a function that minimizes the loss function. 

## Gradient Decent
Since the derivative represent the slop of the function, we can follow this slop all the way to a "top" or a "bottom" of a function where the derivative is zero. Of course we can just solve for the point where gradient equals zero, which we will talk about in next part. In this case, the function we care about is the loss function.   
For every derivative we take with respect to each parameter, we update the corresponding parameter $\theta_i$ towards the direction of that slop a step size of $\alpha$, the learning rate. 
$$\theta_i := \theta_i - \alpha\frac{\partial}{\partial\theta_i}L(\theta).$$
It is possible that gradient decent will lead you to a local maxima/minima, but it won't in this case. $L(\theta)$ is a convex quadratic function which has a only a global maximum. this means that you will always get to the optimum point eventually. 

## The normal equations