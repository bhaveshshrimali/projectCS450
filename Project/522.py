# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 14:50:36 2016

@author: bhavesh
"""

import numpy as np
from scipy.optimize import fsolve
import math as ma
import matplotlib.pyplot as plt

#ans = np.array([[11],[1],[1],[1]])
def g(x):
    return np.array([-1/81*ma.cos(x[0]) + 1/9 *(x[1])**2 ...
    + 1/3*(ma.sin(x[2])),(1/3*(ma.sin(x[0])+ma.cos(x[2]))),-1/9*ma.cos(x[0])...
    +1/3*(x[1])+1/6*(ma.sin(x[2]))])

# True solution as given by fsolve: 
true_sol = fsolve(g,np.array([0,0.33,0]))

# Initial Starting guess for the problem: 0
err = 1
epsilon = np.finfo(float).eps
citer = 1
xin = np.array([1,1,1])
error_f = ([])
total_it = ([])
while err > epsilon:
    if citer ==1:
        xn = g(xin)
    else:
        xn = g(xn)
    err = np.linalg.norm(xn-g(xn))
    error_f.append(err)    
    total_it.append(citer)    
    citer = citer+1
    
error_f = np.array(error_f)
total_it = np.array(total_it)
logerr = np.log(error_f)
logit = np.log(total_it)

# Calculating the best fit polynomial for calculating the relation...
# between the log error and number of iterations: 
fit = np.polyfit(logit,logerr,1)
print("best fit line slope = {} \n\n".format(fit[0]))
xsol = xn
# Computing the Spectral radius of the corresponding Jacobian Matrix:
def G(x):
    return np.array([[1/81*ma.sin(x[0]),2/9*x[1],1/3*ma.cos(x[2])],...
    [1/3*ma.cos(x[0]),0,-1/3*ma.sin(x[2])],[1/9*ma.sin(x[0]),...
    1/3,1/6*ma.cos(x[2])]])

#print("\n\nxsol: {}\n\n".format(G(xsol)))
eigvl,eigvec = (np.linalg.eig(G(xsol)))
rho = np.max(np.abs(eigvl))

print("\nC = {}\n\n".format(rho))

# Plotting the error for observation: 
plt.figure(1)
plt.semilogy(total_it,error_f);
plt.xlabel('Number of Iterations')
plt.ylabel('Absolute Error in Fixed Point Iteration (log scale)')
plt.title('Fixed Point Iteration (Absolute Error vs Number of Iterations)')
plt.legend(['Absolute Error'],loc = 0,prop={'size':14})
plt.grid(True)


# Solving using the Newton Raphson method: 
niter = 1
err_newton = 1
total_itn = ([])
error_n=([])
# Definition of the function to solve for:
def f(x):
    return np.array([-1/81*ma.cos(x[0]) + 1/9 *(x[1])**2 + 1/3*(ma.sin(x[2]))..
    -x[0],(1/3*(ma.sin(x[0])+ma.cos(x[2])))-x[1],-1/9*ma.cos(x[0])+1/3*(x[1])..
    +1/6*(ma.sin(x[2]))-x[2]])


# Definition of the Jacobian: 
def J(x):
    return np.array([[1/81*ma.sin(x[0])-1.0,2/9*x[1],1/3*ma.cos(x[2])],...
    [1/3*ma.cos(x[0]),-1.0,-1/3*ma.sin(x[2])],[1/9*ma.sin(x[0]),1/3,...
    1/6*ma.cos(x[2])-1.0]])

# Newton Loop for the iteration:
while err_newton > epsilon :
    if niter ==1:
        xk = np.array([0,0,0])
    else:
        sk = np.linalg.solve(-1.0*J(xk),f(xk))
        xk = xk+sk
    err_newton = np.linalg.norm(f(xk))
    error_n.append(err_newton)
    total_itn.append(niter)
    niter= niter+1

error_n=np.array(error_n)
total_it=np.array(total_it)

# Plotting the error for observation: 
plt.figure(2)
plt.semilogy(total_itn,error_n);
plt.xlabel('Number of Iterations')
plt.ylabel('Absolute Error in Newton\'s method (log scale)')
plt.title('Newton\'s Method (Absolute Error vs Number of Iterations)')
plt.legend(['Absolute Error'],loc = 0,prop={'size':14})
plt.grid(True)
