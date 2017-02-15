# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 13:02:45 2016

@author: bhavesh
"""

from scipy import optimize
import numpy as np
import math as ma
import scipy
import cmath as cm

s=3.5

def f(x):
    return np.array([1/(x[0]*ma.tan(x[0]))-1/x[0]**2-1/x[1]**2-1/x[1],x[0]**2+x[1]**2-3.5**2])

def jac(x):
    return np.array([[(2*ma.cos(2*x[0]) + x*ma.sin(2*x[0]) + 2*x[0]**2 - 2)/(x[0]**3*(ma.cos(2*x[0])) - 1),(x[1] + 2)/x[1]**3],[2*x[0],2*x[1]]])
    
sol=optimize.root(f,[1.0,1.0])
answer=np.array(sol.x)
x2,y2=answer

def f1(x):
    return x**2+y2**2-3.5**2
    
x2=scipy.optimize.root(f1,x2)
x2=x2.x

x2 = float(cm.sqrt(s**2-y2**2))