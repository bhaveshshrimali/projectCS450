# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:33:59 2016

@author: bhavesh
"""

from scipy.optimize import curve_fit
import numpy as np

#Getting the Data Points :

t_sample = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
y_sample = np.array([6.8,3.0,1.5,0.75,0.48,0.25,0.2,0.15])

#Defining the Function : 
def f1(t,x1,x2):
    return x1*np.exp(x2*t)
    
x1,x2 = curve_fit(f1,t_sample,y_sample)

nl_x = x1

#Using Linear Least Squares:
y_ln = np.log(y_sample)

#Linear Least Squares on Log-Log data:
A = np.vstack([t_sample,np.ones(len(t_sample))]).T
print("\nA:\n{}\n".format(A))
x2,lnx1 = np.linalg.lstsq(A,y_ln)[0]
x1 = np.exp(lnx1)

#x1,x2 obtained from linear least squares: 
l_x = np.array([x1,x2])


#Printing the final values obtained from curve fitting: 
print("\nLinear Least Squares Solution: {}\nNonlinear Least Squares Solution: {}".format(l_x,nl_x))
