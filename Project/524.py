# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:03:41 2016

@author: bhavesh
"""

import numpy as np
from scipy.optimize import fsolve,root

def f(x):
    w1,w2,x1,x2 = x
    out = [w1+w2-2.0,w1*x1+w2*x2-0.0,w1*x1**2+w2*x2**2-(2.0/3.0),w1*x1**3+w2*x2**3-0.0]
    return out
    
# Choosing a guess:
guess1 = [1,1,1.0/np.sqrt(3),-1.0/np.sqrt(3)]
guess2 = [1,1,0.0,0.0]
guess3 = [1,1,-1,1]
#guess4 = [0,0,0,0]


# Solving the nonlinear-system: 
sol_nl=root(f,guess4)
sol_nl=sol_nl.x
true_sol = np.array([1.0,1.0,-1.0/np.sqrt(3),1.0/np.sqrt(3)])
# Output the solution: 
print("Solution:\n\nw1 = {}\nw2 = {}\nx1 = {}\nx2 = {}".format(sol_nl[0],sol_nl[1],sol_nl[2],sol_nl[3]))
print('\n\nw1+w2 = {};\nw1*x1+w2*x2 = {};\nw1*x1^2+w2*x2^2 = {}\nw1*x1^3+w2*x2^3 = {}'.format(sol_nl[0]+sol_nl[1],sol_nl[0]*sol_nl[2]+sol_nl[1]*sol_nl[3],sol_nl[0]*sol_nl[2]**2+sol_nl[1]*sol_nl[3]**2,sol_nl[0]*sol_nl[2]**3+sol_nl[1]*sol_nl[3]**3))
print("\n\nTrue Solution: {}".format(true_sol))