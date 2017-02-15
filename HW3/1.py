# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 12:53:06 2016

@author: bhavesh
"""

import numpy as np
import scipy.linalg as sp

# Initialization of the Matrix A 
A = np.array([[6,2,1],[2,3,1],[1,1,1]])
x_0 = np.array([0,0,1])
shift = 2
max_iter = 15 
B = A-shift*np.identity(3)
print("\n\n A = \n {}\n\n B = A -\u03C3 I = \n {}\n\n".format(A,B))

#Computing the PLU Factorization of A 
LU,piv = sp.lu_factor(B)
print ("The PLU factorization of B is\n\n LU = \n{}\n\n piv = \n{}\n".format(LU,piv))

#Inverse Iteration with the given shift
for iter in range(max_iter):
    if iter == 0:
        y = sp.lu_solve((LU,piv),x_0)
        x = y/(np.linalg.norm(y,np.inf))
        temp = np.linalg.norm(y,np.inf)
    else:
        y = sp.lu_solve((LU,piv),x)
        x = y/(np.linalg.norm(y,np.inf))
        temp = np.linalg.norm(y,np.inf)
        

#Eigen-values using the built-in subroutine 
true_eig,true_vec = np.linalg.eig(A)
true_val = np.amin(abs((true_eig) - 2)) + 2
eigval = 1/temp + shift
eigvec = x
diffval = (eigval-true_val)/(true_val)




