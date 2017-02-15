# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:12:25 2016

@author: bhavesh
"""

R = np.zeros([len(H),len(H)])
R[:,0] = IT
for m in range(len(H)-1):
    for n in range(m+1):
        R[m+1,n+1] = ((4**(n+1))*R[m+1,n] - R[m,n])/(4**(n+1) - 1)