import time
from math import sqrt
import numpy as np
from scipy import linalg

# random design
#A = rng.randn(m, n)  # random design
A = np.array([[3, 0.5],
     [0.5, 1]])
mu = np.array([1, 2])

def soft_thresholding(u, q):
    """soft-thresholding operation"""
    return np.sign(u) * np.maximum(np.abs(u) - q, 0)


def proximal_gradient(ramd, iteration):

    global A,mu

    w = np.array([3, -1])
    pobj = []
    eigen, _ = np.linalg.eig(2*A)
    L = max(eigen)  # Lipschitz constant
    for _ in range(iteration):
        a = np.dot(A+A.T, w-mu)
        w = soft_thresholding(w - np.dot(A+A.T, w-mu) / L, ramd / L)
    return w

iteration = 100
print("iteration:",iteration)
lamd = 2
print("lambda:2 - w:",proximal_gradient(lamd, iteration))
lamd = 4
print("lambda:4 - w:",proximal_gradient(lamd, iteration))
lamd = 6
print("lambda:6 - w:",proximal_gradient(lamd, iteration))
