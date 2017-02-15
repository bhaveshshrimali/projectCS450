# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:58:38 2016

@author: bhavesh
"""

import numpy as np
from scipy.integrate import dblquad
@np.vectorize
def Integral(m,n):
    In = dblquad(lambda x,y:1/((m-x)**2+(n-y)**2)**0.5,-1,1,lambda x:-1,lambda x: 1)
    return(In)
x_bar = np.linspace(2,10,10)
y_bar = np.linspace(2,10,10)
X, Y = np.meshgrid(x_bar, y_bar)
I = Integral(X,Y)[0]
pot_10_10 = I[len(x_bar)-1,len(y_bar)-1]
print('Potential at location (10,10) = ',pot_10_10)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
fig = plt.figure()
pot = fig.gca(projection='3d')
surf = pot.plot_surface(X, Y, I, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('Interpolated Population curves for the United States')
pot.set_xlabel(r'$\hat{x}$')
pot.set_ylabel(r'$\hat{y}$')
pot.set_zlabel(r'$\Phi$')

fig.colorbar(surf, shrink=0.5, aspect=14)
plt.show()
print("\n\n",(I))