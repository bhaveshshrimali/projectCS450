# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 16:02:40 2016

@author: Ahmed
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

t = np.array([0.0,1.0,2.0,3.0,4.0,5.0])
y = np.array([1.0,2.7,5.8,6.6,7.5,9.9])
#Print Polynomial fitting
print('Polynomil')
for n in range(6):
    print('n =', n)
    x = np.polyfit(t,y,n)
    x = np.flipud(x)
    print ('x =', x)
    dx = np.zeros(len(x)-1)
    for i in range (len(x)-1):
        dx[i]=x[i+1]*(i+1)
    print('dx =', dx)
    dx_val = np.zeros(len(t))
    for z in range(len(t)):
        for l in range(n):
            dx_val[z] += dx[l]*t[z]**l
    print('dx_val =', dx_val)
#Cubic spline
print('Cubic Spline')
SP_coeffs = interpolate.UnivariateSpline(t,y)
plt.figure()
plt.plot(t,y,'-o')
T = np.linspace(0,5,100)
plt.plot(T,SP_coeffs(T),'-x')
for z in range(len(t)):
    dx_val[z] = SP_coeffs.derivatives(t[z])[1]
print('dx_val =', dx_val)

#smoothing spline
print('Smoothing Spline')
K = np.array([1,2,4,5])
for k1 in K:
    print('k =', k1)
    SP_coeffs = interpolate.UnivariateSpline(t,y,k=k1)
    plt.figure()
    plt.plot(t,y,'-o')
    plt.plot(T,SP_coeffs(T),'-x')
    for z in range(len(t)):
        dx_val[z] = SP_coeffs.derivatives(t[z])[1]
    print('dx_val =', dx_val)

#Hermit spline
print('Hermit Spline')
Her_coeffs = interpolate.PchipInterpolator(t,y)
plt.figure()
plt.plot(t,y,'-o')
plt.plot(T,Her_coeffs(T),'-8')
Her_der = Her_coeffs.derivative(nu=1)
print('dx_val =', Her_der(t))