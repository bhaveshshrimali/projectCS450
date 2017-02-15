# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:51:41 2016

@author: bhavesh
"""

import numpy as np
from scipy.integrate import dblquad

xcap = np.linspace(2,10,10)

def f(xt,yt,x,y):
    return 1.0/np.sqrt((xt-x)**2+(yt-y)**2)

z=([])
for x1 in xcap:
    for y1 in xcap:
        def g(x,y):
            return f(x1,y1,x,y)
        z.append(dblquad(g,-1.0,1.0,lambda x:-1.0,lambda x:1.0))

z=(np.array(z))[:,0]    
pot_10_10 = z[99]
print("\n\nThe potential at the point (10,10) is: {}\n\n".format(pot_10_10))

X,Y = np.meshgrid(xcap,xcap)
#print("\n\nx: \n{}\n\nX: \n{}".format(xcap,X))
l = len(xcap)
Z = np.reshape(z,(-1,10))
print("\n\nZ:{}".format(Z))

# Plotting the function
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


fig = plt.figure()
pot = fig.gca(projection='3d')
surf = pot.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0.5, antialiased=False)
plt.title('Electrostatic Potential')
pot.set_xlabel('xhat')
pot.set_ylabel('yhat')
pot.set_zlabel('PHI')
pot.view_init(30, -15)

