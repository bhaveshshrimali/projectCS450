# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 00:09:29 2016

@author: bhavesh
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

xf = 1.0
tf = 2
dt = 0.05
nt=int(tf/dt)
tw=np.linspace(0,tf,nt+1)
delx = np.array([10,20,30,40,50,60,70,80,90,100])
delx = 1.0/delx
err=([])
for dx in delx:
    nx=int(xf*1.0/dx + 1.0)
    xnum = np.linspace(0,xf,nx)
    y0 = (np.concatenate((np.sin(np.pi*xnum),np.zeros(len(xnum))))).tolist()
    hl=int(0.5*len(y0))
    def sys1(y,t):
        y1=(y[0:hl])
        y1p = y[hl:len(y)]
        dydt=y1p.tolist()

# Imposing the Initial Conditions: 
        dydt.append(0)
        for j in range(1,len(y1)-1):
            dydt.append((y1[j+1]-2*y1[j]+y1[j-1])/(dx**2))
        dydt.append(0)
#        dydt.append(-2*y1[len(y1)-1]+y1[len(y1)-2])
        return dydt
    
#    ans=sys1(y0,tw)
#    break    
    sol=odeint(sys1,y0,tw)
    u=sol[:,0:len(xnum)]
    u_tr = np.outer(np.cos(np.pi*tw),((np.sin(np.pi*xnum))))
    err.append((np.amax((u_tr-u))))

err=np.array(err)
T_ml,Xml=np.meshgrid(tw,xnum)
u_tr = np.outer(np.cos(np.pi*tw),((np.sin(np.pi*xnum))))

# Plotting the function:

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
 
fig = plt.figure(0)
pot = fig.gca(projection='3d')
surf = pot.plot_surface(Xml, T_ml, np.transpose(u), rstride=1, cstride=1,cmap=cm.coolwarm,
                       linewidth=0.5, antialiased=False)
plt.title('$\mathbf{Method\ of\ Lines}$')
pot.set_xlabel('$\mathbf{x\ (Space)}$')
pot.set_ylabel('$\mathbf{t\ (Time)}$')
pot.set_zlabel('$\mathbf{Solution:\ u(x,t)}$')
plt.grid(True)
plt.tight_layout()

# Plotting the Error: 

plt.figure(1)
plt.loglog(delx,err,'-bo')
plt.title('$\mathbf{Absolute\ Maximum\ Error\ vs\ Mesh\ Size}$')
plt.xlabel('$\mathbf{Delta\ x\ (Mesh\ Size)}$')
plt.ylabel('$\mathbf{Absolute\ Maximum\ Error}$')
plt.tight_layout()
plt.grid(True)



fig1 = plt.figure(2)
pot1 = fig1.gca(projection='3d')
surf1 = pot1.plot_surface(Xml, T_ml, np.transpose(u_tr), rstride=1, cstride=1,cmap=cm.coolwarm,
                       linewidth=0.5, antialiased=False)
plt.title('$\mathbf{Method\ of\ Lines\ :\ True\ Solution}$')
pot1.set_xlabel('$\mathbf{x\ (Space)}$')
pot1.set_ylabel('$\mathbf{t\ (Time)}$')
pot1.set_zlabel('$\mathbf{True Solution:\ u(x,t)}$')
plt.tight_layout()
#    

#for i1 in range(6):
#        plt.figure(i1)
#        plt.plot(xnum,u_tr[10*i1,:],xnum,u[10*i1,:])
#        plt.legend(['True Solution','Approximate Solution'])

