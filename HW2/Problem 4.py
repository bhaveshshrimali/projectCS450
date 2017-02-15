# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 23:52:22 2016

@author: bhavesh
"""
import numpy as np

A = np.array([[1.0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
              [1,-1,0,0],[1,0,-1,0],[1,0,0,-1],[0,1,-1,0],[0,1,0,-1],
              [0,0,1,-1]]);
b = np.array([2.95,1.74,-1.45,1.32,1.23,4.45,1.61,3.21,0.45,-2.75]);
x = (np.linalg.lstsq(A,b,))[0]
print("\n x=\n",x)
rel_errors = (np.array([2.95,1.74,-1.45,1.32])-x);
rel_errors = np.divide(rel_errors,x);
print("\n",rel_errors)
              