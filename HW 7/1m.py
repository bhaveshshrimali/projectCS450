# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:51:50 2016

@author: bhavesh
"""
# Using the Finite-Difference-Method 
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import sympy as sp

num = np.array([1,3,7,15])
solution =([])
time=([])

for n in num:
    t=(np.linspace(0,1,n+2))
    h=1./(n+1)
    def func(u):
        out = np.zeros(len(t))
        out[0] = u[0]
        for i in range(1,len(t)-1):
                out[i]=(u[i+1]-(h**2*(10*u[i]**3+3*u[i]+t[i]**2)+2*u[i]-u[i-1]))
        out[n+1]=(u[n+1]-1)
        return out
    sol=fsolve(func,t)
    solution.append(sol)
    time.append(t)

# Plotting the figure:
plt.figure(1)
plt.plot(time[0],solution[0],'-bo',time[1],solution[1],'-go',time[2],solution[2],'-ro',time[3],solution[3],'-mo');
plt.xlabel('Time (t)')
plt.ylabel('Solution: u(t)')
plt.title('u(t): Finite Difference Method')
plt.legend(['n=1','n=3','n=7','n=15'],loc = 0,prop={'size':14})
plt.grid(True)


# Using the Collocation-Method
num_col = np.array([3,4,5,6])
solution_col=([])
time_col=([])

# Time-steps for plotting the final solution: 
timecol = np.linspace(0,1,50)

# Function for the polynomial: 

def func_col(a,t,m):
    A = np.vander(t,m)
    return A.dot(np.fliplr([a])[0])
    

for n1 in num_col:
    t1=(np.linspace(0,1,n1))
    h=1./(n-1)
    def f(uk):
        utemp = np.zeros(len(t1))
        upptemp = np.zeros(len(t1))
        out=np.zeros(len(t1))
        for index1 in range(n1):
            for index2 in range(len(t1)-2):
                upptemp[index1] = upptemp[index1] + (index2+2)*(index2+1)*uk[index2+2]*(t1[index1])**(index2)
            for i3 in range(len(t1)):
                utemp[index1] = utemp[index1] + uk[i3]*(t1[index1])**(i3)
            if index1==0:
                out[0] = utemp[0] - 0
            else:
                if index1==n1-1:
                    out[index1] = utemp[index1]-1.0
                else:
                    out[index1] = upptemp[index1]-10*(utemp[index1])**3-3*utemp[index1]-(t1[index1])**2
        return out
        
    a_col=fsolve(f,np.ones(len(t1)))  
    solution_col.append(func_col(a_col,timecol,n1))

# Plotting the figure: 
plt.figure(2)
plt.plot(timecol,solution_col[0],timecol,solution_col[1],timecol,solution_col[2],timecol,solution_col[3]);
plt.xlabel('Time (t)')
plt.ylabel('Solution: u(t)')
plt.title('u(t): Collocation Method')
plt.legend(['n=3','n=4','n=5','n=6'],loc = 0,prop={'size':14})
plt.ylim([0,1])
plt.grid(True)
                
            
    