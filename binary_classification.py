import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

x = []
y = []
lam = 0.001 #hyperparameter

result_steepest_j = []
result_newton_j = []

def generate_dataset():
    """generate random dataset by the following rules"""

    global x,y
    n = 40;
    omega = randn(1, 1);
    noise = 0.8 * randn(n, 1);
    x = randn(n, 2);
    y = 2 * ((omega * x[:,0] + x[:,1]).T + noise > 0) - 1;

def cost_function(w):
    """calculate J(w)"""

    global x,y,lam
    cost=0

    for i in range(len(x)):
        cost += np.log(1+np.exp((-y[i] * np.dot(w, x[i])))) + lam * np.dot(w.T, w)

    return cost

def gradient(w):
    """get the gradient of J(w)"""

    global x,y,lam
    d = 0
    for i in range(len(x)):
        d += -y[i]*x[i]*(np.exp(-y[i] * np.dot(w, x[i])) / (1 + np.exp(-y[i] * np.dot(w, x[i]))))
    return d + 2 * lam * w

def hessian(w):

    global x, y, lam
    hessian = 0
    for i in range(len(x)):
        val = np.exp(-y[i] * np.dot(w, x[i])) / (1 + np.exp(-y[i] * np.dot(w, x[i])))**2
        hessian += val * np.outer(x[i], x[i])

    hessian += 2 * lam * np.eye(2)
    return hessian


def lipschitz_constant():
    """return the lipschitz_constant"""

    global x,lam
    t = np.dot(x, x.T)
    val = np.dot(x, x.T) + 2* lam * np.eye(len(x))
    eigen,_ = np.linalg.eig(val)
    return max(eigen)/4

def optimization_steepest():
    """optimize parameter w """

    # initialization
    w = np.array([0.1, 0.1])

    # iteration(training)
    for i in range(10):
        f = cost_function(w)
        learning_rage = 1/lipschitz_constant()
        w = w - learning_rage * gradient(w)
        result_steepest_j.append(f)

    return w

def optimization_newton():

    # initialization
    w = np.array([0.1, 0.1])

    # iteration(training)
    for i in range(10):
        f = cost_function(w)
        w = w - np.dot(np.linalg.inv(hessian(w)),gradient(w))
        result_newton_j.append(f)
    return w

# generate dataset for optimizing
generate_dataset()

# optimize by batch steepest gradiant method
w_st = optimization_steepest()
# optimize by newton based method
w_nt = optimization_newton()

# show the cost J(w) w.r.t. t

x = np.arange(0, 10, 1)
plt.plot(x, result_steepest_j, label='steepest')
plt.plot(x, result_newton_j, label='newton')
plt.xlabel("t")
plt.ylabel("J(w)")
plt.legend()
plt.show()


# # evaluate
# colors = ['r' if y > 0 else 'b' for y in y[:,0]]
# plt.scatter(x[:,0], x[:,1], c=colors)
# x = np.arange(-2.0, 2.0, 0.1)
# y1 = -w_st[0]/w_st[1]*x
# y2 = -w_nt[0]/w_nt[1]*x
# plt.plot(x, y1, label='steepest')
# plt.plot(x, y2, label='newton')
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.legend()
# plt.show()
