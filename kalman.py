# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
def kalman(A, y):
    """
    Formulars from 《Optimal State Estimation》,Page88 & P128
    :param A: array in the sklearn form. each "row" represents
     an input (the measurement coeffcient).
    :param y:vector of the response variable (measuremen)
    :return: x:the estimates. p: covariance of x
    """
    dim = A.shape[1]
    n = A.shape[0]
    # initialize the estimate of the state(s)
    # and its(their) covirance
    x0 = np.zeros((dim, 1))
    P0 = 1e5 * np.identity(dim)
    R = 1  # the measurement variance
    # convert numpy arrays to matrix(suitable for matrix manupulation)
    x0 = np.mat(x0)
    P0 = np.mat(P0)
    # start iteration
    x, p = [], []
    for i in range(n):
        Hk = np.mat(A[i, :][:, np.newaxis]).T
        Kk = P0 * Hk.T * (Hk * P0 * Hk.T + R).I
        x_new = x0 + Kk * (y[i] - Hk * x0)
        Pk = (np.mat(np.identity(dim)) - Kk * Hk) * P0 * \
             (np.mat(np.identity(dim)) - Kk * Hk).T + Kk * R * Kk.T
        #Note! this is not universal! We assume that the "time update"
        #step is trival, i.e., the states are constants
        #Modify the following line as you need.
        P0 = Pk
        x0 = x_new
        # for return
        x.append(x0)
        p.append(P0)
    #convert x back to array
    x=[np.array(xi).squeeze() for xi in x]
    return x,p
def testKalman():
    #prepare training data (3-dimension)
    dataLen = 100
    A = np.random.random_sample((dataLen, 1)) * 5
    A = np.c_[A ** 2, A, np.ones((dataLen, 1))]
    x = np.array([1, 2, 3])
    y = np.dot(A, x) + np.random.randn(dataLen) * 0.1
    x_,p= kalman(A, y)
    print "final estimate:",x_[-1]
    #extract the confidence interval of the 0th state
    y_err=[np.sqrt(pi[0,0]) for pi in p]
    #now plot!
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.plot(range(dataLen),np.ones(dataLen)*x[0], 'r-.')
    # ax.plot(range(dataLen),[xi[0] for xi in x_], 'b-x', alpha=0.5)
    ax.errorbar(range(dataLen),[xi[0] for xi in x_], yerr=y_err, fmt='o')
    ax.set_ylim([-0.5,2])
    ax.set_title('Original & Prediction')
    ax.set_xlabel(u'Iteration Times')
    ax.set_ylabel(u'True and Estimated states')
    plt.show()
    pass
if __name__ == '__main__':
    testKalman()
