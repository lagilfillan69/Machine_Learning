import numpy as np
import matplotlib.pyplot as plt
import time as time

def main():
    ###########################################################
    # 3.a
    # create a guassian curve vector with a million points
    print("Running 3.a")
    x = np.random.normal(1.5,6,1000000000)
    ###########################################################
    # 3.b
    #create a uniform vector with a million points
    print("Running 3.b")
    z = np.random.uniform(low=-1.0,high=3.0,size=1000000000)
    ###########################################################
    # 3.c
    # create a histogram of the 2 vectors and save the photo
    print("Running 3.c")
    # x_histogram, x_bin_edges = np.histogram(x,)
    # z_histogram, z_bin_edges = np.histogram(z,)

    x_fig, x_ax = plt.subplots()
    x_ax.set_title("X Plot")

    z_fig, z_ax = plt.subplots()
    z_ax.set_title("Z Plot")

    x_counts, x_bins, patches = x_ax.hist(x, density=True)
    z_counts, z_bins, patches = z_ax.hist(z, density=True)

    plt.show()

    x_fig.savefig("ps1-3-c-1.png",format="png",)
    z_fig.savefig("ps1-3-c-2.png",format="png")
    ###########################################################
    #3.d
    #add 1 value to x determine the length by shape with a for loop
    print("Running 3.d")

    before = time.time()
    for i in range(0,x.shape[0]):
        x[i] += 1
    after = time.time()

    for_timing = after - before
    print("3.d For Loop Time Difference =", for_timing)
    # ###########################################################
    #3.e
    #add 1 value to x determine the length by shape with vector addition
    print("Running 3.e")
    before = time.time()
    x_new = np.add(x,np.ones(shape=1000000000))
    after = time.time()

    add_timing = after - before
    print("3.e Addition Time Difference =", add_timing)
    ###########################################################
    # 3.f
    print("Running 3.f")
    y = np.empty(1000000000)
    y.fill(1.5)
    output = np.less(z,y)
    output = np.delete(output, np.where(output == False))
    print("Length of values less than 1.5 = ", output.shape[0])
    ###########################################################
    #4.a
    #find the following without using loops
    print("Running 4.a")

    A = np.array([[2,1,3],[5,4,8],[6,3,10]])
    #minimum in each collum
    print(A)

    min_collum_locations = np.argmin(A, axis=0)
    print("Minimum Collum Values : \n" ,A[min_collum_locations[0]][0],"\n",A[min_collum_locations[1]][1], "\n", A[min_collum_locations[2]][2], "\n")

    #maximum in each row
    max_row_locations = np.argmax(A,axis=1)
    print("Maximum Row Value : \n", A[0][max_row_locations[0]], "\n", A[1][max_row_locations[1]], "\n", A[2][max_row_locations[2]], "\n")

    #highest value in A
    max_A = np.max(A)
    print("Maximum A : ", max_A)

    #sum of each collum
    sum_A_collums = np.sum(A,axis=0)
    print("Sum of Collum Values :", sum_A_collums)

    #sum of all elements
    sum_A = np.sum(A)
    print("Sum of A : ", sum_A)

    #find b, who is the square of all elements in A
    B = np.power(A,2)
    print("B : ", B)
    ############################################################
    # #4.b
    eq_var = np.array([[2,5,2], [2,6,4], [6,8,18]])
    eq_res = np.array([12,6,15])
    eq_solv = np.linalg.solve(eq_var,eq_res)

    print("x = ", eq_solv[0], ", y = ", eq_solv[1], ", z = ", eq_solv[2])
    ############################################################
    #4.c
    print("Running 4.c")
    x1 = [-1.5,0,0.5]
    x2 = [-1,-1,0]

    L1_x1_norm = np.linalg.norm(x1,1)
    L1_x2_norm = np.linalg.norm(x2,1)
    L2_x1_norm = np.linalg.norm(x1)
    L2_x2_norm = np.linalg.norm(x2)

    print("L1 x1_norm =", L1_x1_norm, "\n L1 x2_norm = ", L1_x2_norm)
    print("L2 x1_norm =", L2_x1_norm, "\n L2 x2_norm = ", L2_x2_norm)
    ###########################################################
    #5.a
    print("Running 5.a")
    y = [[0],
         [1],
         [2],
         [3],
         [4],
         [5],
         [6],
         [7],
         [8],
         [9]]


    X_base = [[1],
         [2],
         [3],
         [4],
         [5],
         [6],
         [7],
         [8],
         [9],
         [10]]

    X_prime = np.hstack((y,y))
    X = np.hstack((y,X_prime))

    # print(X)

    np.random.shuffle(y)

    xtrain = np.empty((8,3))
    xtest = np.empty((2,3))
    ############################################################
    # 5.b
    print("Running 5.b")
    np.random.shuffle(y)
    for i in range(0,10):
        if i < 8 :
            xtrain[i][:] = X[y[i]][:]
        else :
            xtest[i-8][:] = X[y[i]][:]
    print("Xtrain = \n", xtrain, "\n")
    print("Xtest = \n", xtest, "\n")
    ############################################################
    #5.c
    print("Running 5.c")
    #since y is already shuffled we can just
    ytrain = y[:8]
    ytest = y[8:]

    print("Ytrain = \n",ytrain,"\n")
    print("Ytest = \n", ytest,"\n")
    ###########################################################
    return 0


if __name__ == "__main__":
    i = main()
    exit(1)

