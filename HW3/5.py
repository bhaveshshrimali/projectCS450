# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 16:33:33 2016

@author: bhavesh
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as mp

def f1(x):
    z =  (x**2+2)/3
    return z
    
def f2(x):
    z =  (3*x-2)**0.5
    return z

def f3(x):
    z =  3-2/x
    return z

def f4(x):
    z =  (x**2-2)/(2*x-3)
    return z

def function(n,start):
    if n==0:
        out = f1(start)
    if n==1:
        out = f2(start)
    if n==2:
        out = f3(start)
    if n==3:
        out = f4(start)
    return out
        
x = np.zeros((10,4));
niter = 10
start = 3

x[0,:] = np.array([3,3,3,3])
# Fixed point iteration 
for method in range(4):
    g = function(method,start)
    for iteration in range(niter-1):
        x[iteration+1,method] = g
        temp = x[iteration+1,method]
        g=function(method,temp)
      
print("\n\n",x)
error = x - 2*np.ones((10,4))
print("\n\n Modified x \n\n",error)

error=abs(error)
error = np.divide(error,(2*np.ones((10,4))))
print("\n\n The relative error is \n\n",error)
xval = np.array([1,2,3,4,5,6,7,8,9,10])

errorg1 = error[:,0]
errorg2 = error[:,1]
errorg3 = error[:,2]
errorg4 = error[:,3]


mp.figure(1)
mp.semilogy(xval,errorg1,xval,errorg2,xval,errorg3,xval,errorg4)
mp.xlabel('Iteration(n)')
mp.ylabel('Relative Error')
mp.title('Relative Error vs Iterations')
mp.legend(['g1(x)','g2(x)','g3(x)','g4(x)'],loc = 0)
mp.grid(True)

mp.figure(2)
mp.semilogy(xval,errorg2,xval,errorg3,xval,errorg4)
mp.xlabel('Iteration(n)')
mp.ylabel('Relative Error')
mp.title('Relative Error vs Iterations (Only Convergent Fixed-point iteration)')
mp.legend(['g2(x)','g3(x)','g4(x)'],loc = 0)
mp.grid(True)