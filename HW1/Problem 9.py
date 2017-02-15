# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 21:14:23 2016

@author: bhavesh
"""

import numpy as np
import math as mp

D = np.matrix('2,0,0;0,2,0;0,0,2');
A = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]]);
b = np.array([[0.1],[0.3],[0.5]]);
B = 1e-3*np.linalg.inv(A);
solution = np.linalg.solve(A*D,b);
solution = D*solution;
condition = np.linalg.cond(A);
correct = np.log10(condition);
correct = 16-mp.floor(correct) +1
print('Matrix A is\n',A)
print('Vector b is\n',b)
print('The conditioning number is\n',condition)
print('Inverse of A is \n',B)
print('The solution is\n',solution)
print('Number of accuracte digits are \n',correct)