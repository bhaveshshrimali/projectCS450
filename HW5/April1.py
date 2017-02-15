# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:18:13 2016

@author: bhavesh
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate as integ
import random 

#Initialization#
ns = np.arange(1000, 1, -5)
hs = 1./ns
mid = np.zeros(len(hs))
trap = np.zeros(len(hs))
simp = np.zeros(len(hs))
true_pi = np.ones(len(hs))*np.pi
count = 0
ax=0
bx=1
monte=([])


#Function to Integrate#
def f(x):
    return 4./(1+x**2)

#Mid-point, Trapezoid and Simpson Integration#
for n in ns:
    xs = np.linspace(0., 1., n)
    xmid = 0.
    xtrap = 0.
    xsimp = 0.
    xromberg=0.
    for i in np.arange(1, len(xs)):
        b = xs[i]
        a = xs[i-1]
        xmid += (b-a)*f((a+b)/2.)
        xtrap += (b-a)/2.*(f(a) + f(b))
        xsimp += (b-a)/6.*(f(a) + 4.*f((a+b)/2.) + f(b))
    mid[count] = xmid
    trap[count] = xtrap
    simp[count] = xsimp
    count += 1

#Monte Carlo integration#
for n in ns:
    total = 0.0
    for i in range(n):
        x = random.uniform(ax, bx)
        total += f(x)
    monte.append((1.0/n * total)*((bx-ax)))

#Romberg Integration#
def trapezcomp(f, a, b, n):
    h = (b - a) / n
    x = a
    In = f(a)
    for k in range(1, n):
        x  = x + h
        In += 2*f(x)
    return (In + f(b))*h*0.5
def romberg(f, a, b, p):
    I = np.zeros((p, p))
    for k in range(0, p):
        I[k, 0] = trapezcomp(f, a, b, 2**k)
        for j in range(0, k):
            I[k, j+1] = (4**(j+1) * I[k, j] - I[k-1, j]) / (4**(j+1) - 1)
    return I
p_rows = np.arange(20, 1, -1)
true_pir = np.ones(len(p_rows))*np.pi
rom=([])
hr=([])
for i in np.arange(0, len(p_rows)):
    p=p_rows[i]
    hr.append(1.0/(2.0*p))
    I=romberg(f,0,1,p)
    r=I[p-1,p-1]
    rom.append(r)

#Relative Errors#
emid = np.abs((mid - true_pi)/ true_pi)
etrap = np.abs((trap - true_pi)/ true_pi)
esimp = np.abs((simp - true_pi)/ true_pi)
emonte = np.abs((monte- true_pi)/ true_pi)
erom=np.abs((rom-true_pir)/ true_pir)
#Plots#
plt.figure(1)
plt.loglog(hs, emid, label="Midpoint")
plt.loglog(hs, etrap, label="Trapezoid")
plt.loglog(hs, esimp, label="Simpson")
plt.loglog(hr, erom, label=" Romberg ")
plt.xlabel('Step size')
plt.ylabel('Relative Error')
plt.title('Numerical Integration Method Accuracy Comparision')
plt.legend(loc =4)
plt.show()
plt.figure(2)
plt.loglog(hs, emonte, label="Monte Carlo")
plt.xlabel('Step size')
plt.ylabel('Relative Error')
plt.title('Accuracy of Monte Carlo Method ')
plt.legend(loc =4)

#comment#
print('Simpson Method and Romberg Method stop improving because their error reaches machine precision due to the rounding error')