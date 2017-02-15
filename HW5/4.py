# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 23:17:42 2016

@author: bhavesh
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def f(x):
    return 4.0/(1+x**2)

# Choosing Sample h sizes for computing using the Newton-Coates: 
#step = np.linspace(1e-6,1,num=10)
step=([])
h = 1.0
while h >= 1e-4:
    step.append(h)
    h=h/2.0

step = np.array(step)      
rom_step = step[0:len(step)]

# True value of pi to be used in computing the 
true_val = np.pi
min_h = np.min(step)

# Using Midpoint Rule, Trapezoidal Rule and Simpson's: 
int_mid = np.zeros(len(step))
int_trap = np.zeros(len(step))
int_simp = np.zeros(len(step))
int_rom = np.zeros([len(step),len(step)])

# Calculating Romberg integral for the first step: 


for i in range(len(step)):    
    a = 0.0    
    b = a + step[i]    
    while b <= 1.0:
        int_mid[i] = int_mid[i] + (b-a)*f(0.5*(a+b))
        int_trap[i] = int_trap[i]+0.5*(b-a)*(f(a)+f(b))
        int_simp[i] = int_simp[i]+(b-a)/6*(f(a)+f(b)+4*f(0.5*(a+b)))
        b = b+step[i]
        a = a+step[i]

# Calculation of the first iteration Romberg Integral based on Trapezoidal 
# (h) and Trapezoidal (h/2)
int_rom[:,0] = int_trap
# Extrapolating the obtained Romberg Integrals in a triangular fashion: 
for j in range(len(step)-1):
    for k in range(j+1):
        int_rom[j+1,k+1] = ((4**(k+1))*int_rom[j+1,k] - int_rom[j,k])/(4**(k+1) - 1)

# Extracting the diagonal of the Romberg Matrix: 
integ_rom = np.diag(int_rom)

rel_mid = np.abs((true_val-int_mid)/true_val)
rel_trap = np.abs((true_val-int_trap)/true_val)
rel_simp = np.abs((true_val-int_simp)/true_val)

# Relative Error for Romberg Integration:
rel_rom = np.abs((true_val-integ_rom)/true_val)
# Plottin the relative error: 

print("\n\nRomberg Integral: \n{}\n\nRelative Error:\n{}".format(integ_rom,rel_rom))


plt.figure(1)
plt.loglog(step,rel_mid,step,rel_trap,step,rel_simp,rom_step,rel_rom);
plt.xlabel('Step Size')
plt.ylabel('Relative Error')
plt.title('Error in Newton-Coates Integration and Romberg Integration')
plt.legend(['Midpoint Integral','Trapezoidal Integral','Simpson\'s Integral','Romberg Integral'],loc = 0,prop={'size':14})
plt.grid(True)

# Introducing the Monte-Carlo Integration: 
int_mcarl = np.zeros(10)
step_mcarl = (np.zeros(10))
for index in range(len(int_mcarl)):
    step_mcarl[index] = int((1+index)**7)
    val1 = np.random.random_sample(int(step_mcarl[index],))
    int_mcarl[index] = np.mean(f(val1))

rel_mcarl = np.abs((true_val-int_mcarl)/true_val)

plt.figure(2)
plt.loglog(step_mcarl,rel_mcarl);
plt.xlabel('Step Size')
plt.ylabel('Relative Error')
plt.title('Monte-Carlo Integration')
plt.legend(['Monte-Carlo'],loc = 0,prop={'size':14})
plt.grid(True)

