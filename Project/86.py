# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:49:53 2016

@author: bhavesh
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def f(x,k):
    return np.e**(-1)*x**k*np.e**x

# Comparison of Forward Backward and Adaptive Quadrature: 
num = np.array([20,25,30,35,40])
I_sum = ([])
Iquad = ([])
I_ch = ([])

for n in num:
    I = ([])
    for k in range(n+1):
        I.append(quad(f,0,1,(k)))
    I = np.array(I)
    I = I[:,0]
    Iquad.append(I)

# Forward Recurrence Relation: 
    If = np.zeros(n+1)
    If[0] = 1-np.e**(-1)
    for k in range(1,n+1):
        If[k] = 1-k*If[k-1]
    I_ch.append(If)
    
# Backward Recurrence Relation: 
    Ik = np.zeros(n+1)
    for m in range(n+1): 
        Ik[n-1-m] = (1-Ik[n-m])*1.0/(n-m)
    I_sum.append(Ik)

# Dividing the values into specific arrays: 

# For Adaptive Quadrature: 
I1 = Iquad[0]
n1 = np.linspace(0,len(I1)-1,len(I1))
I2 = Iquad[1]
n2 = np.linspace(0,len(I2)-1,len(I2))
I3 = Iquad[2]
n3 = np.linspace(0,len(I3)-1,len(I3))
I4 = Iquad[3]
n4 = np.linspace(0,len(I4)-1,len(I4))
I5 = Iquad[4]
n5 = np.linspace(0,len(I5)-1,len(I5))

#For Forward Recurrence Relation: 
I1r = I_ch[0]
n1r = np.linspace(0,len(I1r)-1,len(I1r))
I2r = I_ch[1]
n2r = np.linspace(0,len(I2r)-1,len(I2r))
I3r = I_ch[2]
n3r = np.linspace(0,len(I3r)-1,len(I3r))
I4r = I_ch[3]
n4r = np.linspace(0,len(I4r)-1,len(I4r))
I5r = I_ch[4]
n5r = np.linspace(0,len(I5r)-1,len(I5r ))

# Backward Recurrence Relation: 
I1b = I_sum[0]
n1b = np.linspace(0,len(I1b)-1,len(I1b))
I2b = I_sum[1]
n2b = np.linspace(0,len(I2b)-1,len(I2b))
I3b = I_sum[2]
n3b = np.linspace(0,len(I3b)-1,len(I3b))
I4b = I_sum[3]
n4b = np.linspace(0,len(I4b)-1,len(I4b))
I5b = I_sum[4]
n5b = np.linspace(0,len(I5b)-1,len(I5b))


wtest = [2 for i in range(5)]
# Adaptive Quadrature: 
plt.figure(0)
plt.plot(n1,I1,n2,I2,n3,I3,n4,I4,n5,I5);
plt.xlabel('k')
plt.ylabel('Integral: I(k)')
plt.title('Adaptive Quadrature')
plt.legend(['k=20','k=25','n=30','n=35','n=40'],loc = 0,prop={'size':14})
plt.grid(True)

# Forward Recurrence Relation: 
plt.figure(1)
plt.plot(n1r,I1r,n2r,I2r,n3r,I3r,n4r,I4r,n5r,I5r);
plt.xlabel('k')
plt.ylabel('Integral: I(k)')
plt.title('Forward Recurrence Relation')
plt.legend(['k=20','k=25','n=30','n=35','n=40'],loc = 0,prop={'size':14})
plt.grid(True)

# Backward Recurrence Relation: 
plt.figure(2)
plt.plot(n1b,I1b,n2b,I2b,n3b,I3b,n4b,I4b,n5b,I5b);
plt.xlabel('k')
plt.ylabel('Integral: I(k)')
plt.title('Backward Recurrence Relation')
plt.legend(['k=20','k=25','n=30','n=35','n=40'],loc = 0,prop={'size':14})
plt.grid(True)

# Comparison between the bcakward recurrence and adaptive quadrature: 

plt.figure(3)
plt.plot(n5b,I5b,n5,I5);
plt.xlabel('k')
plt.ylabel('Integral: I(k)')
plt.title('Backward Recurrence Relation vs Adaptive Quadrature (n=40)')
plt.legend(['Backward-Recurrence','Adaptive Quadrature'],loc = 0,prop={'size':14})
plt.grid(True)

