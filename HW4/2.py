# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 21:59:28 2016

@author: bhavesh
"""

from scipy.optimize import fsolve
import math as ma
import numpy as np
import scipy 
mach_epsilon = np.finfo(float).eps

def system(p):
    x,y= p
    return (x/ma.tan(x)+y,x**2+y**2-3.5**2)
    
x1,y1= fsolve(system,(1.0,1.0))

def system2(p):
    x,y = p
    return (1/(x*ma.tan(x))-1/(x**2)-1/y-1/(y**2),x**2+y**2-3.5**2)

x2,y2= scipy.optimize.root(system2,[x1,y1])
print("\nx2: {}\ny2: {}\n\n".format(x2,y2))
temp = 1
iter = 1
while 1:    
    temp,y2=fsolve(system2,[x2,y2])
    if (abs(temp-x2)<=mach_epsilon):
        break
    x2=temp
    iter=iter+1

        
print("iterations: {}\nx2: {}".format(iter,x2))  

    
    
    