# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 15:53:45 2025

@author: hibado
"""

from scipy.stats import multivariate_normal as norm
import numpy as np
import matplotlib.pyplot as plt 

target = norm([25,25], np.array([[13, 10], [7, 23]]))
x_range = np.linspace(0, 50)
y_range = np.linspace(0, 50)
X,Y = np.meshgrid(x_range,y_range)

pdf = target.pdf(np.vstack([X.ravel(),Y.ravel()]).T)

pdf = pdf.reshape(X.shape)
plt.imshow(pdf, origin ="lower")


U = -np.log(pdf)
grad_U = np.array(np.gradient(U))
M = np.eye(2)
M_inv = np.linalg.inv(M)
dt = 0.1
L = 100

x = [np.random.uniform([x_range[0], y_range[0]], [x_range[-1], y_range[-1]])]
p = [norm.rvs([0,0], M)]
traj = []
for i in range(2000):
    p_prime = norm.rvs([0,0], M)
    x_prime = [x[-1].copy()]
    
    for _ in range(L):
        p_prime = p_prime - dt/2*grad_U[:,int(np.fix(x_prime[-1][0])), int(np.fix(x_prime[-1][1]))]
        x_prime.append(x_prime[-1] +dt*M_inv@p_prime)
        p_prime = p_prime - dt/2*grad_U[:,int(np.fix(x_prime[-1][0])), int(np.fix(x_prime[-1][1]))]
    H_0 = U[int(np.fix(x[-1][0])), int(np.fix(x[-1][1]))] + 1/2*p[-1].T@M_inv@p[-1]
    H_1 = U[int(np.fix(x_prime[-1][0])), int(np.fix(x_prime[-1][1]))] + 1/2*p_prime.T@M_inv@p_prime
    A = min(1, np.exp(-H_1)/np.exp(-H_0))
    if np.random.rand()<A:
        x.append(x_prime[-1])
        p.append(p_prime)
        traj += x_prime
x = np.array(x)    
#%%
plt.figure()
plt.imshow(pdf.T, origin ="lower", cmap = "Blues")
plt.plot(x[:,0], x[:,1],".", color = "red", alpha=0.01)

#%%
plt.figure()
for loc in traj:
    # plt.clear()
    plt.imshow(pdf.T, origin ="lower", cmap = "Blues")
    plt.plot(loc[0],loc[1],".", color = "red")
    plt.show()
    plt.pause(0.01)
    