# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:21:16 2016

@author: bhavesh
"""

import numpy as np
import matplotlib.pyplot as mp


maxsize = 15
ratio = ([])


for index in range(maxsize):
    A = -1*np.ones([index+1, index+1])
    A = np.triu(A)
    print("\n\n",np.transpose(A).dot(A))
    for i in range(index+1):
        A[i,i] = 1
    U=np.array([])
    s =np.array([])
    V=np.array([])
    U,s,V = np.linalg.svd(A)
    ratio.append((np.amax(s))/(np.amin(s)))
ratio = np.array(ratio)

msize = ([])
for i in range(maxsize):
    msize.append(i+1)
    
msize = np.array(msize)
mp.plot(msize,ratio)
mp.xlabel('Matrix Size(n')
mp.ylabel('$\u03C3_{max}$/$\u03C3_{min}$')
mp.title('Plot of Singular values vs Size of Matrix')
mp.grid(True)
print("\n\n",A)
U,s,V = np.linalg.svd(A)
print("\n\n",s)