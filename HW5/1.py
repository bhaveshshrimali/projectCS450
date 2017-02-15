# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:49:11 2016

@author: bhavesh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

nyear = np.array([1,2,3,4,5,6,7,8,9])
year1 = np.array([1900.0,1910,1920,1930,1940,1950,1960,1970,1980])
year2 = year1-1900.0
year3 = year1-1940.0
year4 = (year1-1940.0)/40
modyear4 = np.linspace(1900,1980,num=81)
explt_year = np.linspace(1900,1990,num=91) 
population = np.array([76212168.0,92228496,106021537,123202624,132164569,151325798,179323175,203302031,226542199])
round_population = np.array([76000000.0,92000000,106000000,123000000,132000000,151000000,179000000,203000000,227000000])
#Funtion to implement the horner's nested scheme: 

def horner(arr,cons):
    z = arr[0]
    for i in range(len(arr)-1):
        z = cons*z+arr[i+1]
    return z
        

mat1 = np.vander(year1,len(nyear),increasing=True)
mat2 = np.vander(year2,len(nyear),increasing=True)
mat3 = np.vander(year3,len(nyear),increasing=True)
mat4 = np.vander(year4,len(nyear),increasing=True)


cond1 = np.linalg.cond(mat1)
cond2 = np.linalg.cond(mat2)
cond3 = np.linalg.cond(mat3)
cond4 = np.linalg.cond(mat4)

# Since the condition number corresponding to mat4 is the least, we use it to
# determine the cooefficients of the basis functions to be used for 
# interpolation

coeffs = np.linalg.solve(mat4,population)

# Rounding to nearest million and solving for the modified coefficients: 
round_coeffs = np.linalg.solve(mat4,round_population)

# Error in the coefficients due to the change in the polynomial: 
err_coeffs = np.abs((coeffs-round_coeffs)/coeffs)

modyear4 = np.linspace(1900,1980,num=81)
alpha = (modyear4-1940)/40
beta = (explt_year-1940.0)/40
# Reversing the array to forward implement the horner's nested scheme
revcoeff = np.fliplr([coeffs])[0]

ans_hor = alpha*(alpha*revcoeff[0]+revcoeff[1])+revcoeff[1]
print("\n\n{}\n\n{}\n\n{}".format(revcoeff[0],alpha,ans_hor))
answer_horner = horner(revcoeff,alpha)

# Implementing the Hermite Spline: 
HSp = interpolate.PchipInterpolator(year1,population)
ans_her = HSp(modyear4)

# Implementing the Cubic Spline: 
CSp = interpolate.UnivariateSpline(year1,population)
ans_cub = CSp(modyear4)

# Extrapolating the Population to 1990: 
ans_extr_cub = CSp(explt_year)
ans_extr_her = HSp(explt_year)
ans_extr_hor = horner(revcoeff,beta)

val_extr_cub = ans_extr_cub[90]
val_extr_her = ans_extr_her[90]
val_extr_hor = ans_extr_hor[90]

true_val = 248709873.0

err_polynomial = np.abs(true_val-val_extr_hor)/true_val
err_hermite = np.abs(true_val-val_extr_her)/true_val
err_spline = np.abs(true_val-val_extr_cub)/true_val

# Plotting the results : 

plt.figure(1)
plt.plot(modyear4,answer_horner,modyear4,ans_her,modyear4,ans_cub,year1,population,'ro');
plt.xlabel('Year (1900-1980)')
plt.ylabel('Population (n)')
plt.title('Population in United States (1900-1980)')
plt.legend(['Polynomial','Hermite','Cubic-Spline','Individual Data'],loc = 0,prop={'size':10})
plt.grid(True)
