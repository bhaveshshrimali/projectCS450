#Submitted by Bhavesh Shrimali
#Importing the libraries
import math as mp
import numpy as np

#Defines a function func, with inputs x and n defined as follows: 
#x: the value at which the function is evaluated
#n: the number of terms in the taylor expansion

def func(x,n):
    result = 0;
    for index1 in range(1,n+1):
        result = (result+(x)**(index1-1)/mp.factorial(index1-1));
    result = 1/result;
    return result

#Initializing the arrays
p=np.array([1,5,10,15,20]);
neg=(-1)*p;
#print (func(20,3));

#Calculating the maximum number of terms, required for positive (for the highest value in the p/n)
err = 1;
i =1;
while err > 0:
    input = np.amax(p);  
    input = np.float(input);
    err = abs(func(input,i+1)-func(input,i))
    if err == 0:
        n1 = i;        
        break
    i=i+1;
#Calculating the maximum number of terms, required for negative (for lowest value in the p/n)
err=1;
i =1;
while err > 0:
    input = np.amax(p);  
    input = np.float(input);
    err = abs(func(input,i+1)-func(input,i))
    if err == 0:
        n2 = i;        
        break
    i=i+1;

#Initializing negative
negative = np.ones(5);   

s = len(p);
for index in range(len(p)):
    ninp=float(p[index]);
    negative[index] = abs((func(ninp,n2)-np.exp(-1*ninp))/(np.exp(-1*ninp)));
#    
#rel_p=np.ones(5); 
#rel_n=rel_p;
#print(func(p))
#
#for index in range(len(p)):
#    rel_p[index] = (func(p[index])-positive[index]);
#print (rel_p)