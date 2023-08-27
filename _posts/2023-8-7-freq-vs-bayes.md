---
title: Frequentist vs Bayesian
date: 2023-8-7
categories: [math foundation]
tags: [MLE, MAP]
math: true
---

# Frequentist versus Bayesian

When treating data as a result of sampling from a probability distribution, it is natural to introduce statistical methods to machine learning algorithms. Frequentist and Bayesian have different views on the parameters that determine the distribution. Frequentist believes parameters are unknown value while Bayesian believes parameters are random variables. Therefore, even though they both care about what parameters maximises the probability of given data, their approaches are different. 

Given dataset $X$ and parameters $\theta$

## Frequentist approach: 
Maximum Likelihood Estimation (MLE).
$$\theta_{MLE} = \arg\max_\theta P(X|\theta)$$
## Bayesian approach: 
Maximum A Posterior (MAP). 
$$\theta_{MAP}=\arg\max_\theta P(\theta|X)=\arg\max_\theta \frac{P(X|\theta)P(\theta)}{P(X)}$$
since $P(X)=\int_\theta P(X|\theta)P(\theta)\text{ }d\theta\text{, }$
$$\theta_{MAP}=P(X|\theta)P(\theta)$$
It also make sense by using joint probability since $\theta$ and $X$ are both random variables. 
$$\theta_{MAP}=\arg\max_\theta P(X,\theta)=\arg\max_\theta P(X|\theta)P(\theta)$$
When calculating for $\theta$, since
$$P(X|\theta)=\prod_{i=1}^{n}P(x^{(i)}|\theta)\text{, }$$
taking the log likelihood instead of the likelihood makes calculation easier by converting product to sum. 

Notice that MAP is not Bayesian estimation since it does not calculate the distribution of $\theta$. In real world, 
$$P(X)=\int_{\theta}P(X|\theta)P(\theta)\text{ }d\theta$$
is usually very hard to calculate.