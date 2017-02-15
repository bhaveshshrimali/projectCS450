# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:19:09 2016

@author: bhavesh
"""

#Submitted by Bhavesh Shrimali
#Importing the libraries

import math as mp
import numpy as np
import scipy as sp

#Initializing the Matrix A and vector b
A = np.array([[2,4,-2],[4,9,-3],[-2,-1,7]]);
print("The matrix A is \n ", A);
b = np.array([2,8,10]);
print("\n b =\n",b);
print("The matrix b is \n",b);

#Calculating P, L and U respectively
P,L,U = sp.linalg.lu(A);

# Calculating the inverse of P to be premultiplied on the right side
Pinv = np.matrix.transpose(P);
b=Pinv.dot(b);
y = sp.linalg.solve_triangular(L,b,lower=True);
x = sp.linalg.solve_triangular(U,y);
print("\n x=\n",x);print("\n y=\n",y);print("\n b=\n",b);

#Assigning corresponding value to sol1
sol1 = x;

#Calculating the solution for Ax = c
c=np.array([4,8,-6]);
c=Pinv.dot(c);
y = sp.linalg.solve_triangular(L,c,lower=True);
x = sp.linalg.solve_triangular(U,y);

#Assigning corresponding value to sol2
sol2 = x;

#Initializing the correpsonding vectors u and v
u = np.array([1,0,0]);
v = np.array([0,2,0]);

#Applying the Shermon-Morrison update
u = Pinv.dot(u);
x1 = sp.linalg.solve_triangular(L,u,lower=True);
z = sp.linalg.solve_triangular(U,x1);
y = sol1;

sol3 = y + (v.dot(y)/(1-v.dot(z)))*z
 
