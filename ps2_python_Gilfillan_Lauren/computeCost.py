import numpy as np

def computeCost(x, y, theta):
    #establish J and H varible types
    H = np.zeros(shape = x.shape[0], dtype=np.float64)
    J = 0
    #calculate H for each x_i
    for i in range(0,x.shape[0]):
        H_temp = np.matmul(x[i,:],theta)
        H[i] = H_temp
    #calculate summation cost for every h_i
    for j in range(y.shape[0]):
        J += np.pow((float(H[j] - y[j])),2)
    J = float(J/(2*y.shape[0]))
    return J
