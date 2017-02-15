# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:28:36 2016

@author: bhavesh
"""

import numpy as np
from scipy.optimize import fsolve,root
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Initialize the paramteters: 
dx = 0.05
dt = 0.0012
tf = 0.06
xf = 1

# Mesh_grid (x-t) for the problem (FDM)
nx = int(xf/dx)
nt = int(tf/dt) 
u_f=np.zeros([nx+1,nt+1])  # Boundary Conditions are automatically imposed here

# Initial shape of the solution: 
xnum=np.array([i*dx for i in range(nx+1)])
tnum=np.array([i*dt for i in range(nt+1)])

# Initial Condition: 
i = 0
for x in xnum:
    if x<=0.5*(xnum[len(xnum)-1]+xnum[0]):
        u_f[i,0] = 2*x
        i=i+1
    else:
        u_f[i,0] = 2-2*x
        i=i+1


for k in range(1,len(u_f[0,:])):
    for i in range(1,len(u_f[:,0])-1):
        u_f[i,k] = u_f[i,k-1]+(dt)/(dx)**2 * (u_f[i+1,k-1]-2*u_f[i,k-1]+u_f[i-1,k-1])

X,T = np.meshgrid(xnum,tnum)

# Plotting the function
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure(0)
pot = fig.gca(projection='3d')
surf = pot.plot_wireframe(X,T, u_f.T, rstride=1, cstride=1,linewidth=1.2)
plt.title('$\mathbf{Finite-Difference\ Solution\ (\Delta t\ =\  0.0012)}$')
pot.set_xlabel('$\mathbf{x\ (Space)}$')
pot.set_ylabel('$\mathbf{t\ (Time)}$')
pot.set_zlabel('$\mathbf{Solution:\ u(x,t)}$')
plt.tight_layout()




# Second Figure using different time step 
# Initialize the paramteters: 
dt1 = 0.0013
nt1 = (tf/dt1) 
r=int(nt1+1)
tnum1=np.array([i*dt1 for i in range(r)])
X,T1 = np.meshgrid(xnum,tnum1)

# Mesh_grid (x-t) for the problem (FDM)
u_f2=np.zeros([nx+1,r])  # Boundary Conditions are automatically imposed here

# Initial Condition: 
u_f2[:,0] = u_f[:,0]

for k in range(1,len(u_f2[0,:])):
    for i in range(1,len(u_f2[:,0])-1):
        u_f2[i,k] = u_f2[i,k-1]+(dt1)/(dx)**2 * (u_f2[i+1,k-1]-2*u_f2[i,k-1]+u_f2[i-1,k-1])

# Plotting the function
fig1 = plt.figure(1)
pot1 = fig1.gca(projection='3d')
surf2 = pot1.plot_wireframe(X,T1, u_f2.T, rstride=1, cstride=1,linewidth=1.2)
plt.title('$\mathbf{Finite-Difference\ Solution\ \Delta t\ =\  0.0013}$')
pot1.set_xlabel('$\mathbf{x\ (Space)}$')
pot1.set_ylabel('$\mathbf{t\ (Time)}$')
pot1.set_zlabel('$\mathbf{Solution:\ u(x,t)}$')
plt.tight_layout()


# Implicit Method solution: 
dt2 = 0.0005
nt2 = int(tf/dt2) 
u_i=np.zeros([nx+1,nt2+1])

# Initial  Condition: 
u_i[:,0] = u_f[:,0]

#print("\n\n{}".format(u_i[0,:]))

for k in range(1,len(u_i[0,:])):    
    def f3(u):
        out=np.zeros(len(u_i[:,0]))
        out[0] = u[0]
        out[len(u)-1] = u[len(u)-1]
        for i in range(1,len(u_i[:,0])-1):
            out[i] = u[i]-(u_i[i,k-1]+(dt2/(dx)**2)*(u[i+1]-2*u[i]+u[i-1]))
        return out
    sol3=fsolve(f3,u_i[:,k-1])
    u_i[:,k] = sol3
#print("\n\n{}".format(sol3))


tnum_i=np.array([i*dt2 for i in range(nt2+1)])

Xi,Ti = np.meshgrid(xnum,tnum_i)

# Plotting the function
fig2 = plt.figure(2)
pot2 = fig2.gca(projection='3d')
surf2 = pot2.plot_wireframe(Xi,Ti, u_i.T, rstride=1, cstride=1,linewidth=1.2)
plt.title('$\mathbf{Finite-Difference\ Solution\ (Implicit\ Discretization)\ \Delta t\ =\  0.005}$')
pot2.set_xlabel('$\mathbf{x\ (Space)}$')
pot2.set_ylabel('$\mathbf{t\ (Time)}$')
pot2.set_zlabel('$\mathbf{Solution:\ u(x,t)}$')
pot2.set_zlim(0,1)
plt.tight_layout()
 
 
#  Crank-Nicholson Discretization Scheme: 
# Implicit Method solution: 
u_cn=np.zeros([nx+1,nt2+1])

# Initial  Condition: 
u_cn[:,0] = u_f[:,0]

#print("\n\n{}".format(u_i[0,:]))

for k in range(1,len(u_cn[0,:])):    
    def f4(u):
        out=np.zeros(len(u_cn[:,0]))
        out[0] = u[0]
        out[len(u)-1] = u[len(u)-1]
        for i in range(1,len(u_cn[:,0])-1):
            out[i] = u[i]-(u_cn[i,k-1]+(dt2/(2*(dx)**2))*(u[i+1]-2*u[i]+u[i-1])+u_cn[i+1,k-1]+u_cn[i-1,k-1]-2*u_cn[i,k-1])
        return out
    sol4=fsolve(f4,u_cn[:,k-1])
    u_cn[:,k] = sol4
#print("\n\n{}".format(sol3))

# Plotting the function
fig3 = plt.figure(3)
pot3 = fig3.gca(projection='3d')
surf2 = pot3.plot_wireframe(Xi,Ti, u_cn.T, rstride=1, cstride=1,linewidth=1.2)
plt.title('$\mathbf{Finite-Difference\ Solution\ (Crank\ Nicholson)\ \Delta t\ =\  0.005}$')
pot3.set_xlabel('$\mathbf{x\ (Space)}$')
pot3.set_ylabel('$\mathbf{t\ (Time)}$')
pot3.set_zlabel('$\mathbf{Solution:\ u(x,t)}$')
pot3.set_zlim(0,1.0)
plt.tight_layout()
        
        
# Semi-Discrete Method for solving the PDE: 
y0 =  (u_f[:,0]).tolist()
D = 1.0/((dx)**2)   
J = ([])
time_sd=np.linspace(0,tf,nt2+1)
def sys(y,t):
    y[0] = 0
    y[len(y0)-1] = 0
    dydt=[0]   
    for j in range(1,len(y0)-1):
        dydt.append(D*(y[j+1]-2*y[j]+y[j-1]))  
        J.append(j)
    dydt.append(0)
    return dydt

sol_sd = np.transpose(odeint(sys,y0,time_sd))
sol_sd[0,:] = np.zeros(len(sol_sd[0,:]))
sol_sd[20,:] = np.zeros(len(sol_sd[20,:]))
Xi,T_sd = np.meshgrid(xnum,time_sd)

# Plotting the function
fig4 = plt.figure(4)
pot4 = fig4.gca(projection='3d')
surf2 = pot4.plot_wireframe(Xi, T_sd, sol_sd.T, rstride=1, cstride=1,linewidth=1.2)
plt.title('$\mathbf{Finite-Difference\ Solution\ (Semi-Discrete)\ \Delta t\ =\  0.005}$')
pot4.set_xlabel('$\mathbf{x\ (Space)}$')
pot4.set_ylabel('$\mathbf{t\ (Time)}$')
pot4.set_zlabel('$\mathbf{Solution:\ u(x,t)}$')
pot3.set_zlim(0,1.0)
plt.tight_layout()
