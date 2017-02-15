# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 00:18:37 2016

@author: bhavesh
"""

import numpy as np
import matplotlib.pyplot as mp
import math as ma


def f1(input):
    z = (input[0]+3)*(input[1]**3-7) + 18
    return z
    
def f2(input):
    z = ma.sin(input[1]*ma.exp(input[0]-1))
    return z
    
def Jk(input):
    z1 = input[1]**3-7
    z2 = 3*input[1]**2*(input[0]+3)    
    z3 = input[1]*ma.exp(input[0])*ma.cos(input[1]*ma.exp(input[0])-1)    
    z4 = ma.exp(input[0])*ma.cos(input[1]*ma.exp(input[0])-1)    
    return np.array([[z1,z2],[z3,z4]])
    
def f(input):
    z1 = (input[0]+3)*(input[1]**3-7) + 18
    z2 = ma.sin(input[1]*ma.exp(input[0])-1)
    z = np.array([z1,z2])
    return z
    
truesol = np.array([0.0,1.0])
trialsol = np.array([-0.5,1.4])
x = trialsol
err_zer = np.linalg.norm(trialsol-truesol)
err = 1
i = 0
mach_epsilon = np.finfo(float).eps

#Implementing the Newton's Method: 
error_N=([])
iter1=([])
while (err>=mach_epsilon):
    s = np.linalg.solve(Jk(x),-1*f(x))
    x = x+s
    temp = x-truesol
    err = np.linalg.norm(temp)/np.linalg.norm(truesol)
    error_N.append(err)
    i=i+1
    iter1.append(i)
    
niterN = i
sol = x
error_N=np.array(error_N)
iter1=np.array(iter1)

#Implementation of Broyden's Method 
Bk=Jk(trialsol)
err1 = 1
j=0
xb = trialsol
error_B=([])
iter2=([])
while(err1>=mach_epsilon):
    sk=np.linalg.solve(Bk,-1.0*f(xb))
    xb = xb+sk
    yk = f(xb)-f(xb-sk)
    zk = np.outer((yk-Bk.dot(sk)),np.transpose(sk))/(sk.dot(sk))
    Bk = Bk + zk
    err1 = np.linalg.norm(xb-truesol)/np.linalg.norm(truesol)
    error_B.append(err1)
    j=j+1
    iter2.append(j)
    
sol_b=xb
niterB=j
error_B=np.array(error_B)
iter2=np.array(iter2)

mp.semilogy(iter1,error_N,iter2,error_B)
mp.grid(True)
mp.xlabel('Number of Iterations')
mp.ylabel('Error in $L^2$ Norm')
mp.legend(['Newton Method','Broyden Method'],loc=0,prop={'size':10})
mp.title('Comparison of Newton and Broyden Method')
