# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:46:55 2016

@author: bhavesh
"""

import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt

c = 1.0
d = 5.0

# Defining the First Order ODE System to be sovled for:
def sys(y,t,c,d):
    y_1,y_2,y_3 = y
    dydt = [-c*y_1*y_2,c*y_1*y_2-d*y_2,d*y_2]
    return dydt
    

y0 = [95.0,5.0,0.0]
time = np.linspace(0,1,100)

sol = odeint(sys,y0,time,args=(c,d))
y1 = np.array(sol[len(time)-1,:])
# Plotting the reuslts 
plt.figure(1)
plt.plot(time,sol[:,0],time,sol[:,1],time,sol[:,2]);
plt.xlabel('Time (t)')
plt.ylabel('y1,y2,y3')
plt.title('Question 2')
plt.legend(['y1','y2','y3'],loc = 0,prop={'size':14})
plt.grid(True)


