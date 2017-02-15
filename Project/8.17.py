# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:01:01 2016

@author: bhavesh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N = np.linspace(3,15,7)
for n in N:
    h = 1/(n-1)
    t = np.linspace(0,1,n)
    s = t 
    S,T = np.meshgrid(s,t)
    
    w1 = np.ones(int(n))
    w1 [range(1,int(n)-1,2)]= 4
    w1 [range(2,int(n)-2,2)]= 2
    w1 = w1*h/3
    A = (S**2 + T**2)**0.5
    for j in range(len(A[1,:])):
        A [:,j] = A [:,j] * w1[j]
    
    plt.figure(1)
    plt.semilogy(n,np.linalg.cond(A),'o')
    plt.xlabel('Size of the Matrix (n)')
    plt.ylabel('Condition Number of the Matrix (c(n))')
    plt.title('Condition Number vs Matrix Size : n = 3 - 15')
    
#     Gaussian-Elimination with partial pivoting :
    
    f = ((s**(2)+1)**(3/2) - s**3)/3
    xgauss = (np.linalg.solve(A,f))
    plt.figure(2)
    plt.plot(t,xgauss)
    plt.title('Gaussian-Elimination : Partial Pivoting')
    plt.xlabel('Size of the Matrix (n)')
    plt.ylabel('Solution by Gaussian Elimination : Partial Pivoting')    
    
    plt.figure(3)
    U,S,V = np.linalg.svd(A)
    l = 0
    x2 = np.zeros(int(n))
    Areq = 1000
    
    for l in range(len(S)):
        if S[1]/S[l] <= Areq:
            x2 = x2 + (np.transpose(U[:,l]).dot(f)/S[l])*V[l,:]
    
    plt.figure(4)
    plt.plot(t,x2) 
    plt.title('SVD')
    plt.xlabel('Size of the Matrix (n)')
    plt.ylabel('Solution by SVD')        
    
    ns = np.zeros(20)
    nr = np.zeros(20)
    j = 0
    
    for mu in np.linspace(0.1,1,20):
        A2 = np.concatenate((A,(mu**0.5)*np.identity(int(n))),axis = 0)
        y2 = np.concatenate((f,np.zeros(int(n))),axis = 0)
        x3 = np.linalg.lstsq(A2, y2)[0]
        ns[j] = np.linalg.norm(x3)
        nr[j] = np.linalg.norm(y2-A2.dot(x3))
        j +=1
        plt.figure(5)
        plt.plot(t,x3)
        plt.title('Regularization')
        plt.xlabel('Size of the Matrix (n)')
        plt.ylabel('Solution by Regularization') 
    plt.figure(6)
    plt.plot(ns,nr)
    plt.title('Regularization')
    plt.xlabel('Norm of the Solution')
    plt.ylabel('Norm of the Residual') 

    xo = np.ones(int(n))
    
    def Res(q):
        return(np.linalg.norm(f-A.dot(q)))
    bnds = tuple((0,None) for x in xo)
    x4 = minimize(Res,xo, method='SLSQP', bounds=bnds).x
   
    plt.figure(7)
    plt.plot(t,x4)
    plt.title('Optimization')
    plt.xlabel('Size of the Matrix (n)')
    plt.ylabel('Solution : u(x,t)')
    
    xo = np.ones(int(n))   
    cons= ({'type': 'ineq', 'fun': lambda x: x[1] - x[0]},
           {'type': 'ineq', 'fun': lambda x: x[2] - x[1]},
           {'type': 'ineq', 'fun': lambda x: x[3] - x[2]},
           {'type': 'ineq', 'fun': lambda x: x[4] - x[3]},
           {'type': 'ineq', 'fun': lambda x: x[5] - x[4]},
           {'type': 'ineq', 'fun': lambda x: x[6] - x[5]},
           {'type': 'ineq', 'fun': lambda x: x[7] - x[6]},
           {'type': 'ineq', 'fun': lambda x: x[8] - x[7]},
           {'type': 'ineq', 'fun': lambda x: x[9] - x[8]},
           {'type': 'ineq', 'fun': lambda x: x[10] - x[9]},
           {'type': 'ineq', 'fun': lambda x: x[11] - x[10]},
           {'type': 'ineq', 'fun': lambda x: x[12] - x[11]},
           {'type': 'ineq', 'fun': lambda x: x[13] - x[12]},
           {'type': 'ineq', 'fun': lambda x: x[14] - x[13]})
    conss = cons[0:int(n-1)]
    x5 = minimize(Res,xo, method='SLSQP', bounds=bnds,constraints= conss).x
    
    plt.figure(8)
    plt.plot(t,x5)
    plt.title('Constrained - Optimization')
    plt.xlabel('Size of the Matrix (n)')
    plt.ylabel('Solution : u(x,t)')    
