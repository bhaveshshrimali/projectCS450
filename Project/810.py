# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 00:29:48 2016

@author: bhavesh
"""

import numpy as np
import scipy as sp
import math as ma
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import simps
  
xs = 1.0
deg = 101
xf = 10
dx = 0.1
nx = int(xf/dx) + 1    
xnum = np.linspace(xs,xf,nx)
Gam = ([])
lim = 1e5
dt = 0.1
nt = int(lim/dt) + 1
t_tr = np.linspace(0,lim,nt)
arr = ([])
arr_quad = ([])
arr_quad_tr = ([])
arr_gauss = ([])

# Computing the weights and sample points of the Gauss-Laguerre Quadrature: 
sam,wgt = np.polynomial.laguerre.laggauss(deg)


def f(t,x):
    return t**(x-1.0)*np.e**(-1.0*t)
    
    
for x in xnum: 
#     Composite Quadrature: Simpson's Rule
    arr.append(f(t_tr,x))
    def g(t):
        return f(t,x)
    arr_quad.append(quad(g,0.0,np.inf)[0])
    arr_quad_tr.append(quad(g,0.0,1e2)[0])
    arr_gauss.append(f(sam,x).dot(wgt))

#     Gauss-Laguerre Polynomial: 
    
arr_quad=np.array(arr_quad)
arr_quad_tr=np.array(arr_quad_tr)

# Number of rows and columns: 
r = len(arr)
c = len(arr[0])
r_gauss = len(arr_gauss)
#c_gauss = len(arr_gauss[0])

b = np.zeros([r,c])
a = np.zeros(r)
#b_gauss = np.zeros([r_gauss,c_gauss])
err_comp = np.zeros(r)
err_quad = np.zeros(r)
err_quadtr = np.zeros(r)
err_gauss = np.zeros(r_gauss)

for i in range(len(arr)):
    b[i,:] = arr[i]
    a[i] = simps(b[i,:])
    val = ma.gamma(xnum[i])
    
    err_comp[i] = np.abs(a[i]-val)
    err_quad[i] = np.abs(arr_quad[i]-val)
    err_quadtr[i] = np.abs(arr_quad_tr[i]-val)    
    err_gauss[i] = np.abs(arr_gauss[i]-val)
    
plt.figure(0)
plt.plot(xnum,err_comp,xnum,err_quad,xnum,err_quadtr,xnum,err_gauss)
plt.xlabel('x',fontsize=18)
plt.ylabel('Error in $\Gamma (x)$',fontsize=18)
plt.legend(['Simpson\'s Rule','Adaptive Quadrature: Truncated','Adaptive Quadrature: Infinite','Gauss-Laguerre Quadrature ( Degree = 101)'])
plt.title('Error in $\Gamma (x)$ vs x',fontsize = 22)
plt.grid(True)
