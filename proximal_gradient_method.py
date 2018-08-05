import time
from math import sqrt
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# definition according to the given condition
A = np.array([[3, 0.5],
     [0.5, 1]])
mu = np.array([1, 2])

ista_diff_w = []
fista_diff_w = []

def calculate_J(w, ramd):
    global A,mu
    J = np.dot(np.dot((w-mu), A), w-mu) + ramd * np.linalg.norm(w,1)
    return J

def soft_thresholding(u, q):
    """soft-thresholding operation"""
    return np.sign(u) * np.maximum(np.abs(u) - q, 0)


def proximal_gradient(ramd, iteration):
    """optimization"""

    global A,mu

    # initialize w(0) with given condition
    w = np.array([3, -1])

    # Lipschitz constant
    eigen, _ = np.linalg.eig(2*A)
    L = max(eigen)

    # update w
    for _ in range(iteration):
        w = soft_thresholding(w - np.dot(A+A.T, w-mu) / L, ramd / L)
        if ramd == 2:
            ista_diff_w.append(calculate_J(w, ramd))

    return w

# evaluation
iteration = 100
print("iteration:",iteration)
lamd = 2
print("lambda:2 - w:",proximal_gradient(lamd, iteration))
lamd = 4
print("lambda:4 - w:",proximal_gradient(lamd, iteration))
lamd = 6
print("lambda:6 - w:",proximal_gradient(lamd, iteration))

def fista(ramd, iteration):
    global A, mu

    # initialize t
    t = 1

    # initialize w(0) with given condition
    w = np.array([3., -1.])
    a = w.copy()

    # Lipschitz constant
    eigen, _ = np.linalg.eig(2 * A)
    L = max(eigen)

    for _ in range(iteration):

        # copy
        w_old = w.copy()
        t_old = t

        # update w
        a -= np.dot(A + A.T, w - mu) / L
        w = soft_thresholding(a, ramd / L)
        fista_diff_w.append(calculate_J(w, ramd))

        # accelerator
        t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
        a = w + ((t_old - 1.) / t) * (w - w_old)

    return w

fista(2, 100)

x = np.arange(0, 100, 1)
t_ista = np.array(ista_diff_w)
t_fista = np.array(fista_diff_w)
plt.xlabel("iteration")
plt.ylabel("J(w)-J(w_best)")
arr_w0 = [ista_diff_w[len(ista_diff_w)-1]]*100
arr_w1 = [fista_diff_w[len(fista_diff_w)-1]]*100
plt.plot(x, t_ista-arr_w0, label='ista')
plt.plot(x, t_fista-arr_w1, label='fista')
plt.legend()
plt.show()
