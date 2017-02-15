# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:01:01 2016

@author: bhavesh
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
#input data
t = np.array([0.0,1.0,2.0,3.0,4.0,5.0])
y = np.array([1.0,2.7,5.8,6.6,7.5,9.9])
#Perturbing y
y = (np.random.randint(low=-1, high=2, size=len(y))*5/100 +1)*y
#Print Polynomial fitting
print('Polynomil')
for n in range(6):
    print('n =', n)
    x = np.polyfit(t,y,n)
    x = np.flipud(x)
    print ('x =', x)
    T = np.linspace(0,5.0,100)
    x_val = np.zeros(len(T))
    for z in range(len(T)):
        for l in range(n+1):
            x_val[z] += x[l]*T[z]**l
    dx = np.zeros(len(x)-1)
    for i in range (len(x)-1):
        dx[i]=x[i+1]*(i+1)
    print('dx =', dx)
    dx_val = np.zeros(len(t))
    for z in range(len(t)):
        for l in range(n):
            dx_val[z] += dx[l]*t[z]**l
    print('dx_val =', dx_val)
    plt.figure(n)
    plt.plot(t,y,'ok', label = 'Data Points')
    plt.plot(T,x_val, label = 'Polynomial of degree %s'%n)
    plt.title('Least Square Polynomial Fit, Degree%s'%n)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
#Cubic spline
print('Cubic Spline')
SP_coeffs = UnivariateSpline(t, y,s=0.0001)
plt.figure()
plt.plot(t,y,'ok', label = 'Data Points')
plt.plot(T,SP_coeffs(T),label = 'Cubic Spline')
plt.title('Cubic Spline')
plt.xlabel('t')
plt.ylabel('y')
plt.grid()
plt.legend()
for z in range(len(t)):
    dx_val[z] = SP_coeffs.derivatives(t[z])[1]
print('dx_val =', dx_val)
#smoothing spline
print('Smoothing Spline')
K = np.array([1,2,4,5])
for k1 in K:
    print('k =', k1)
    SP_coeffs = interpolate.UnivariateSpline(t,y,k=k1,s=0.0001)
    for z in range(len(t)):
        dx_val[z] = SP_coeffs.derivatives(t[z])[1]
    print('dx_val =', dx_val)
    plt.figure()
    plt.plot(t,y,'ok', label = 'Data Points')
    plt.plot(T,SP_coeffs(T),label = 'Smoothing Spline, k = %s'%k1)
    plt.title('Smoothing Spline, k = %s'%k1)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
#Hermit spline
print('Hermit Spline')
Her_coeffs = interpolate.PchipInterpolator(t,y)
Her_der = Her_coeffs.derivative(nu=1)
print('dx_val =', Her_der(t))
plt.figure()
plt.plot(t,y,'ok', label = 'Data Points')
plt.plot(T,Her_coeffs(T),label = 'Hermit Spline')
plt.title('Hermit Spline')
plt.xlabel('t')
plt.ylabel('y')
plt.grid()
plt.legend()
