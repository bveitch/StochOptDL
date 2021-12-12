# StochOptDL
Stochastic Optimization in Deep Learning

A paper looking at the use of second-order solvers (NLCG and BFGS within scipy) for training Neural Networks. Code is comprised of four sources,

## src-rosen
Comparisons of gradient descent, line-search strategies and Newton's method using the classical Rosenbrock function

## src-logistic
Comparisons of gradient descent, line-search strategies, Newton's method with NLCG and BFGS using logistic regression. 
Code contains tests with and without mini-batch sampling

## src-mnist
Comparisons of gradient descent with NLCG and BFGS using cross-entropy loss with shallow neural nets. 
Code contains objective function wrappers, gradient test, convergence analysis and train-dev accuracy comparisons.

## tex-project
TeX Source code for final project
