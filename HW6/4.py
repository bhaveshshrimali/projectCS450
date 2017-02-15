# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:30:01 2016

@author: bhavesh
"""

import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt

GM = 1.0
e = np.array([0,0.5,0.9])
l=int(len(e))

def sys(vec,t,GM):
    x,y,xp,yp = vec
    r=(x**2+y**2)**1.5
    dydt = [xp,yp,-GM*x/r,-GM*y/r]
    return dydt

time = np.linspace(0,1000,1000)
solution=([])

for ei in e:
    vec0 = ([1-ei,0.0,0.0,((1+ei)/(1-ei))**(0.5)])
    sol = odeint(sys,vec0,time,(GM,))
    solution.append(sol)
solution = np.array(solution)
solt1 = solution[0,:,:]
solt2 = solution[1,:,:]
solt3 = solution[2,:,:]

# Plotting the curves

# Figure 1
plt.figure(1)
plt.subplot(311)
plt.plot(time,solt1[:,0]);
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.title('x(t) for e = 0')
plt.legend(['x(t)'],loc = 0,prop={'size':14})
plt.grid(True)

plt.subplot(312)
plt.plot(time,solt1[:,1]);
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.legend(['y(t) for e = 0'],loc = 0,prop={'size':14})
plt.grid(True)

plt.subplot(313)
plt.plot(solt1[:,0],solt1[:,1]);
plt.xlabel('x (t)')
plt.ylabel('y(t)')
plt.legend(['y (x) for e = 0'],loc = 0,prop={'size':14})
plt.grid(True)

# Figure 2
plt.figure(2)
plt.subplot(311)
plt.plot(time,solt2[:,0]);
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.title('x(t)for e = 0.5')
plt.legend(['x(t)'],loc = 0,prop={'size':14})
plt.grid(True)

plt.subplot(312)
plt.plot(time,solt2[:,1]);
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.legend(['y(t) for e = 0.5'],loc = 0,prop={'size':14})
plt.grid(True)

plt.subplot(313)
plt.plot(solt2[:,0],solt2[:,1]);
plt.xlabel('x (t)')
plt.ylabel('y(t)')
plt.legend(['y (x) for e = 0.5'],loc = 0,prop={'size':14})
plt.grid(True)

# Figure 3
plt.figure(3)
plt.subplot(311)
plt.plot(time,solt3[:,0]);
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.title('x(t) for e = 0.9')
plt.legend(['x(t)'],loc = 0,prop={'size':14})
plt.grid(True)

plt.subplot(312)
plt.plot(time,solt3[:,1]);
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.legend(['y(t) for e = 0.9'],loc = 0,prop={'size':14})
plt.grid(True)

plt.subplot(313)
plt.plot(solt3[:,0],solt3[:,1]);
plt.xlabel('x (t)')
plt.ylabel('y(t)')
plt.legend(['y (x) for e = 0.9'],loc = 0,prop={'size':14})
#plt.title('y(t) vs x(t) for e = 0.9')
plt.grid(True)

# Calculating the Energy and Angular Momentum for e = 0.9:
E = 0.5*(solt3[:,2]**2+solt3[:,3]**2) - 1/(solt3[:,0]**2+solt3[:,1]**2)**0.5
A = solt3[:,0]*solt3[:,3] - solt3[:,1]*solt3[:,2]

plt.figure(4)
plt.plot(time,E,time,A);
plt.xlabel('Time (t)')
plt.ylabel('E(t) and A(t)')
plt.title('Energy: E(t) and Angular Momentum:  A(t) for e = 0.9')
plt.legend(['E(t)','A(t)'],loc = 0,prop={'size':14})
plt.grid(True)


plt.figure(4)
Ener=plt.subplot(211)
Ener.set_ylim([-1.0,1.0])
plt.plot(time,E);
plt.xlabel('Time (t)')
plt.ylabel('E(t)')
plt.title('Energy: E(t) and Angular Momentum:  A(t) for e = 0.9')
plt.legend(['Energy'],loc = 0,prop={'size':14})
plt.grid(True)

Ang = plt.subplot(212)
Ang.set_ylim([-1.0,1.0])
plt.plot(time,A);
plt.xlabel('Time (t)')
plt.ylabel('A(t)')
plt.legend(['Angular Momentum'],loc = 0,prop={'size':14})
plt.grid(True)


#print("\n\nEnergy:\n{}\n\nAngular Momentum:\n{}".format(E,A))