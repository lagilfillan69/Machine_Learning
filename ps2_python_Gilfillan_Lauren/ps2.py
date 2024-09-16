import gradientDescent
import computeCost
import normalEqn
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas

######################################################################
# ############## INSTRUCTIONS ##########
# Uncomment each problem as you want to run it,
# they each have similar varibles, so I would reccomend
# only doing one problem at a time
########################################

def main():
    toy_set = np.array([[1,1,1,1.5],
         [1,2,2,3],
         [1,3,3,4.5],
         [1,4,4,6]])


    theta_1 = np.array([0,1,0.5])
    theta_2 = np.array([10, -1, -1])
    theta_3 = np.array([3.5, 0, 0])

#   PROBLEM 1   #
##################################################################
    # print("\nQUESTION 1 : \n")
    # cost_theta_1 = computeCost.computeCost(x = toy_set[:,0:3],y = toy_set[:,3:4],theta = theta_1)
    #
    # print("Results for J(Theta_1) : ", cost_theta_1)
    #
    # cost_theta_2 = computeCost.computeCost(x=toy_set[:, 0:3], y=toy_set[:, 3:4], theta=theta_2)
    #
    # print("Results for J(Theta_2) : ", cost_theta_2)
    #
    # cost_theta_3 = computeCost.computeCost(x=toy_set[:, 0:3], y=toy_set[:, 3:4], theta=theta_3)
    #
    # print("Results for J(Theta_3) : ", cost_theta_3)

# #   PROBLEM 2   #
# ##################################################################
#     print("\nQUESTION 2 : \n")
#     #run gradient descent
#     theta, cost = gradientDescent.gradientDescent(toy_set[:,0:3],toy_set[:,3:4], 0.001, 15)
#
#     print("theta = ",theta)
#     print("cost after 15 iters :", cost[cost.size-1])
#
#
# #   PROBLEM 3   #
# ##################################################################
#     print("\nQUESTION 3 : \n")
#     #normal eq
#
#     theta = normalEqn.normalEqn(toy_set[:,0:3],toy_set[:,3:4])
#     print("theta = ", theta)
#
#

    #   PROBLEM 4   #
    #####################################################################
   #  #4.a import csv file
   #  data = pandas.read_csv("./input/hw2_data1.csv",header=None)
   #  xy = data.to_numpy()
   #
   #  #4.b plot the figure
   #  x_fig, x_ax = plt.subplots()
   #  x_ax.set_title("4.b Horsepower vs Car Price")
   #  x_ax.grid()
   #  x_ax.minorticks_off()
   #  x_ax.plot(xy[:,0], xy[:,1],'x', markeredgewidth=1)
   #  x_fig.show()
   #  x_fig.savefig("ps2-4-b.png", format="png")
   #
   #  #4.c #make temp arrays
   #  X = np.array(xy[:, 0])
   #  X = np.transpose(X)
   #  Y = np.array(xy[:, 1])
   #
   #  print("\nQUESTION 4.c : \n")
   #  print("X shape = ", X.shape)
   #  print("Y shape = ", Y.shape)
   #
   # # 4.d divide data randomly
   #  index = np.arange(0,X.shape[0])
   #  np.random.shuffle(index)
   #  spilt_index = int(np.floor(X.shape[0] * .9))
   #
   #  x_train = np.ones((spilt_index,2))
   #  y_train = np.ones(spilt_index)
   #  x_test = np.ones((X.shape[0] - spilt_index ,2))
   #  y_test = np.ones((X.shape[0] - spilt_index))
   #
   #  for i in range(0,X.shape[0]):
   #      if(i < spilt_index):
   #          x_train[i,1] = X[index[i]]
   #          y_train[i] = Y[index[i]]
   #      else :
   #          x_test[i - spilt_index,1] = X[index[i]]
   #          y_test[i - spilt_index] = Y[index[i]]
   #
   #  # 4.e
   #  theta, cost = gradientDescent.gradientDescent(x_train, y_train,0.3, iters = 500)
   #
   #  cost_fig, cost_ax = plt.subplots()
   #  cost_ax.set_title("4.e Cost v Iterations")
   #  cost_ax.grid()
   #  cost_ax.minorticks_off()
   #  cost_ax.plot(np.arange(1,501),cost, 'x', markeredgewidth=1)
   #  cost_fig.show()
   #  cost_fig.savefig("ps2-4-e.png", format="png")
   #
   #
   # # 4.f print estimated equation over data
   #
   #  #generate points for graph
   #  x_theta = [X.min(),X.max()]
   #  y_theta = np.empty(2)
   #  y_theta[0] = theta[0] + theta[1] * x_theta[0]
   #  y_theta[1] = theta[0] + theta[1] * x_theta[1]
   #  x_ax.plot(x_theta,y_theta)
   #  x_ax.set_title("4.f Horsepower vs Car Price")
   #  x_fig.show()
   #  x_fig.savefig("ps2-4-f.png", format="png")
   #
   #  # 4.g
   #  print("\nQUESTION 4.g : \n")
   #  y_pred = np.empty(y_test.shape[0])
   #  square_error = 0
   #
   #  for q in range(0,x_test.shape[0]):
   #      y_pred[q] = theta[0] + x_test[q][1] * theta[1]
   #      square_error += pow((y_pred[q] - y_test[q]), 2)
   #  square_error = square_error / float(y_pred.shape[0])
   #  print("Square error Linear Regiression= ", square_error)
   #
   #  #4.h
   #  print("\nQUESTION 4.h : \n")
   #  #normal eq run
   #  theta_norm = normalEqn.normalEqn(x_train,y_train)
   #  y_pred_norm = np.empty(y_test.shape[0])
   #  square_error = 0
   #  for q in range(0, x_test.shape[0]):
   #      y_pred_norm[q] = theta_norm[0] + x_test[q][1] * theta_norm[1]
   #      square_error += pow((y_pred_norm[q] - y_test[q]), 2)
   #  square_error = square_error / float(y_pred_norm.shape[0])
   #  print("Square error Normal Equation = ", square_error)

    # ## PORBLEM 5 #
    # ####################################################################
    # # 5.a #import data
    # data = pandas.read_csv("./input/hw2_data3.csv", header=None)
    # xy = data.to_numpy()
    #
    # Y = np.array(xy[:, xy.shape[1] - 1])
    # X = np.array(xy[:, :xy.shape[1] - 1])
    #
    # x_mean = np.empty(X.shape[1])
    # x_std = np.empty(X.shape[1])
    # for i in range(0, X.shape[1]):
    #     x_mean[i] = X[:, i].mean()
    #     x_std[i] = np.std(X[:, i])
    #
    # for j in range(0, len(x_mean)):
    #     for k in range(0, X.shape[0]):
    #         X[k][j] = float(float(X[k][j] - x_mean[j]) / float(x_std[j]))
    #
    # print("\nQUESTION 5.a : \n")
    # print("X1 mean = ", x_mean[0])
    # print("X2 mean = ", x_mean[1])
    # print("X1 std = ", x_std[0])
    # print("X2 std = ", x_std[1])
    #
    # x_bias = np.ones((X.shape[0], 1))
    # X = np.hstack((x_bias, X))
    # print(X.shape)
    # print(Y.shape)
    #
    # # 5.b
    # print("\nQUESTION 5.b : \n")
    # theta, cost = gradientDescent.gradientDescent(X, Y, 0.01, iters=750)
    #
    # cost_fig, cost_ax = plt.subplots()
    # cost_ax.set_title("Cost v iters")
    # cost_ax.grid()
    # cost_ax.minorticks_off()
    # cost_ax.plot(np.arange(1, 751), cost, 'x', markeredgewidth=1)
    # cost_fig.show()
    # cost_fig.savefig("ps2-5-b.png", format="png")
    #
    # print("theta = ", theta)
    #
    # #5.c
    # print("\nQUESTION 5.c : \n")
    # prediction = theta[0] + theta[1]*(float(2300-x_mean[0])/x_std[0]) + theta[2]*(float(1300-x_mean[1])/x_std[1])
    # print("Prediciton = ",prediction)
    # return 0
###########################################################################

if __name__ == "__main__":
    main()
    exit(0)


