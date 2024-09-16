import numpy as np
import computeCost as cp


def computeParitalDerivative(x, y, theta, x_index, alpha, theta_j):
    # establish J and H varible types
    H = np.zeros(shape=x.shape[0], dtype=np.float64)
    dJdTheta = 0
    J = 0
    # calculate H for each x_i
    for i in range(0, x.shape[0]):
        H_temp = np.matmul(x[i, :], theta)
        H[i] = H_temp
    # calculate summation cost for every h_i
    for j in range(y.shape[0]):
        J += float(H[j] - y[j]) * x[j][x_index]
    dJdTheta = theta_j - float(alpha / y.shape[0]) * J
    return dJdTheta


def gradientDescent(X_train, y_train, alpha, iters):
    cost_history = np.zeros(iters)
    theta = np.random.random(size=X_train.shape[1])
    for iters_completed in range(0, iters):
        theta_new = np.zeros(X_train.shape[1])
        for i in range(0,theta.shape[0]):
            theta_new[i] = computeParitalDerivative(X_train,y_train,theta,i,alpha,theta[i])
        theta = theta_new
        cost_history[iters_completed] = cp.computeCost(X_train, y_train, theta)
    return theta, cost_history




