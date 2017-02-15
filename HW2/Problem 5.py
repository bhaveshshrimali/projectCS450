# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 22:34:38 2016

@author: bhavesh
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

A = ([]);
#Generation of the Hilbert Matrix
for n in range(2,13):
    matrix1 = [[0 for x in range(n)] for x in range(n)];
    print("\n\n",type(matrix1))
    for i in range(n):
        for j in range(n):
            matrix1[i][j] = 1/(i+j+1)
    matrix1 = np.array(matrix1);
    A.append(matrix1)

b = np.array(A[0])
hilbert = A;

ortho = ([]);
orthom = ([]);
orthor=([]);

storeQ = ([]);
# Carrying out Gram-Schmidt
for n in range(2,13):
    orth = 0;
    Q = np.array([[0.0 for x in range(n)] for x in range(n)]);
    h = np.copy(hilbert[n-2]);
    for k in range(0,n):
        r=0.0;
        Q[0:n,k] = (h[0:n,k]);
        for j in range(0,k):  
            r= Q[0:n,j].dot(h[0:n,k]);
            Q[0:n,k] = Q[0:n,k] - r*Q[0:n,j] ;
        Q[0:n,k] = np.array(Q[0:n,k])/(np.linalg.norm(Q[0:n,k]));

    storeQ.append(Q);
    orth= -np.log10(np.linalg.norm((np.identity(n)-(np.transpose(Q).dot(Q)))));
    ortho.append(orth);

# Carrying out Modified Gram-Schmidt
for n in range(2,13):
    orth2 = 0;
    Q = np.array([[0.0 for x in range(n)] for x in range(n)]);
    h = np.copy(hilbert[n-2]);
    for k in range(0,n):
        Q[0:n,k] = (h[0:n,k])/(np.linalg.norm(h[0:n,k]));
        for j in range(k+1,n):
            r = Q[0:n,k].dot(h[0:n,j])
            h[0:n,j] = h[0:n,j]-r*Q[0:n,k]
    orth2= -np.log10(np.linalg.norm((np.identity(n)-(np.transpose(Q).dot(Q)))));
    orthom.append(orth2);     

print("\n",hilbert[0]);

# Applying Gram-Schmidt on the obtained Q itself
for n in range(2,13):
    orth = 0;
    Q1 = np.array([[0.0 for x in range(n)] for x in range(n)]);
    h = np.array(storeQ[n-2]);
    for k in range(0,n):
        r=0.0;
        Q1[0:n,k] = (h[0:n,k]);
        for j in range(0,k):  
            r= Q1[0:n,j].dot(h[0:n,k]);
            Q1[0:n,k] = Q1[0:n,k] - r*Q1[0:n,j] ;
        Q1[0:n,k] = np.array(Q1[0:n,k])/(np.linalg.norm(Q1[0:n,k]));

    storeQ.append(Q);
    orth3= -np.log10(np.linalg.norm((np.identity(n)-(np.transpose(Q1).dot(Q1)))));
    orthor.append(orth3);



# Householder Transformation on Hilbert Matrices
orthoh=([])
for n in range(2,13):
    orth = 0;
    Qh = np.array([[0.0 for x in range(n)] for x in range(n)]);
    h=np.copy(hilbert[n-2]);
    H = np.identity(n);   
    for i in range(0,n):
        if h[i,i] > 0:
            alpha = np.zeros(n)
            alpha[i] = 1
            v = np.concatenate((np.zeros(i),h[i:n,i])) + np.linalg.norm(np.concatenate((np.zeros(i),h[i:n,i])))*(alpha)
        else:
            alpha = np.zeros(n)
            alpha[i] = 1            
            v = np.concatenate((np.zeros(i),h[i:n,i])) - np.linalg.norm(np.concatenate((np.zeros(i),h[i:n,i])))*(alpha)
        print("\n",v)
        H=(np.identity(n)-2.0*np.outer(v,np.transpose(v))/(v.dot(v))).dot(H)
        h=H.dot(h)
    orth4 = -np.log10(np.linalg.norm((np.identity(n)-(np.transpose(H).dot(H)))));
    orthoh.append(orth4)
    
print("\n\nNumber of Accurate Digits in Gram-Schmidt:\n\n",ortho);
print("\n\nNumber of Accurate Digits in Modified Gram-Schmidt:\n\n",orthom);
print("\n\nNumber of Accurate Digits in Repeated Gram-Schmidt:\n\n",orthor)
print("\n\nNumber of Accurate Digits in Householder:\n\n",orthoh);

ortho = np.array(ortho)
orthom = np.array(orthom)
orthor = np.array(orthor)
orthoh = np.array(orthoh)

f = np.array([2,3,4,5,6,7,8,9,10,11,12])

plt.figure(1)
plt.plot(f,ortho,f,orthom,f,orthor,f,orthoh);
plt.xlabel('Dimension of the Q Matrix (n)')
plt.ylabel('Number of Accurate Digits (d)')
plt.title('Number of Accurate Digits vs Dimension of Q Matrix')
plt.legend(['Gram-Schmidt','Modified Gram-Schmidt','Repeated Gram-Schmidt','Householder'],loc = 0,prop={'size':10})
plt.grid(True)
