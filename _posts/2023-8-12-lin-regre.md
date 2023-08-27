---
title: Linear Regression
date: 2023-8-12
categories: [regression]
tags: [linear regression]
math: true
---

# Linear Regression
Linear regression can be used when we want to approximate a set of data to a line or a hyperplane. In general there are three approches to this method. 
We define dataset and parameters
$$\{(x_1, y_1), (x_2, y_2),...,(x_n, y_n)\}\text{, where }x\in R^p$$
$$\theta\in R^p.$$
Therefore the function will be
$$h(x_i) = \theta^T x_i.$$
To measure how well a certain line or hyperplane represents the data, we define the loss function
$$L(\theta) = \sum_{i=0}^{n}(h(x_i)-y_i)^2.$$
The goal would be finding a function that minimizes the loss function. 

## Gradient Decent
Since the derivative represent the slop of the function, we can follow this slop all the way to a "top" or a "bottom" of a function where the derivative is zero. Of course we can just solve for the point where gradient equals zero, which we will talk about in next part. In this case, the function we care about is the loss function.   
For every derivative we take with respect to each parameter, we update the corresponding parameter $\theta_i$ towards the direction of that slop a step size of $\alpha$, the learning rate. 
$$\theta_i := \theta_i - \alpha\frac{\partial}{\partial\theta_i}L(\theta).$$
It is possible that gradient decent will lead you to a local maxima/minima, but it won't in this case. $L(\theta)$ is a convex quadratic function which has a only a global maximum. this means that you will always get to the optimum point eventually. 

## The normal equations
As mentioned in last section, we could just directly solve for derivative of $L(\theta)$ equals zero. To do this, we have to represent $L(\theta)$ in matrix form.   
First, we're going to rewrite $X$, $Y$, and $\theta$ in form of vector and matrix:
$$X=
\begin{pmatrix} 
    \text{ }—\text{ }(x_1)^T—\text{ } \\
    \text{ }—\text{ }(x_2)^T—\text{ } \\
    ... \\
    \text{ }—\text{ }(x_n)^T—\text{ } \\
\end{pmatrix}_{n\times p},\quad
Y=
\begin{pmatrix}
    y_1\\
    y_2\\
    ...\\
    y_n\\
\end{pmatrix}_{n\times 1},\quad
\theta=
\begin{pmatrix} 
    \theta_1\\
    \theta_2\\
    ...\\
    \theta_n\\
\end{pmatrix}_{p\times 1},$$
where each data $x_n$ has $p$ dimensions corresponded with $p$ parameters.  
Now, expand the original loss function
$$\begin{aligned} 
L(\theta)&=\sum_{i=0}^{n}(h(x_i)-y_i)^2\\
&=\sum_{i=0}^{n}(\theta^Tx_i-y_i)^2\\
&=
\begin{pmatrix}
\theta^Tx_1-y_1&\theta^Tx_2-y_2&...&\theta^Tx_n-y_n&
\end{pmatrix}
\begin{pmatrix}
\theta^Tx_1-y_1\\
\theta^Tx_2-y_2\\
...\\
\theta^Tx_n-y_n\\
\end{pmatrix}\\
&=(\theta^TX^T-Y^T)(X\theta-Y)\\
&=\theta^TX^TX\theta-\theta^TX^TY-Y^TX\theta+Y^TY\\
&=\theta^TX^TX\theta-2\theta^TX^TY+Y^TY\\
\end{aligned}$$
therefore the derivative will be
$$\frac{\partial}{\partial\theta_i}L(\theta)=2X^TX\theta-2X^TY,$$
solve for the derivative equals to zero gets
$$X^TX\theta=X^TY$$
$$\theta=(X^TX)^{-1}X^TY.$$