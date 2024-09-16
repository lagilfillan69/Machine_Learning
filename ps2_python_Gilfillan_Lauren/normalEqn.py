
import numpy as np
def normalEqn(x_train,y_train):
    x_transpose = np.transpose(x_train)
    inverse = np.linalg.pinv(np.matmul(x_transpose,x_train))
    xy = np.matmul(x_transpose,y_train)
    theta = np.matmul(inverse,xy)
    return theta