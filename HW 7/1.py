# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:29:12 2016

@author: bhavesh
"""

import numpy as np 
from scipy.optimize import fsolve 
import matplotlib.pyplot as plt

# Definition of the number of mesh points  
num = np.array([1,3,7,15])
solution=([])

# Initilization of the outer loop on the number of mesh points:
for n in num:
    temp = (np.linspace(0,1,n[j]))    
    h = 1.0/(n[j]+1)    
    def func(u):
        out=([])        
        a1=u[0]-0        
        out.append(a1.tolist()) 
        for i in range(1,n[j]):
            if i==1:
                a=u[1] - ( (h**2)*(10*u[0]**3+3*u[0]+temp[0]**2) )
                out.append(a.tolist())
            else:
                b=u[i]-(h**2)*(10*u[i-1]**3+3*u[i-1]+temp[i-1]**2)-2*u[i-1]+u[i-2]                
                out.append(b.tolist())
            c=u[n[j]-1]-1.0
            out.append(c.tolist())
        return out  
    trial=temp.tolist()
    sol=fsolve(func,temp)
    solution.append(sol)
#        
#            
#        
#temp = np.linspace(0,1,10)
#def func2(u):
#    out = u[0]-0.0
#    out.append(u[1] - ( (h**2)*(10*u[0]**3+3*u[0]+temp[0]**2) ) )
#    return out
