# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 21:14:23 2016

@author: bhavesh
"""

import numpy as np

n=12;
x = np.random.rand(n);
sequence = x;

def twopass(arr):
    dim = len(arr);
    if dim < 10:
        print ("The array isn't long enough");
    else:
        index1 = 0;
        xbar = 0.0;

        while index1 < dim:
            xbar = xbar + (1/dim)*(arr[index1]);
            index1 += 1;
    
        index2 = 0; 
        var = 0.0;
        while index2 < dim:
            var = var + (1/(dim-1))*(arr[index2] - xbar)**2;
            index2 += 1;
    
        sd_tp = var**0.5;
    
    return sd_tp
    
        
def onepass(arr):    
    dim = len(arr);
    index1 = 0;
    xbar = 0.0;
    var =0.0;
    
    while index1 < dim:       
        xbar += arr[index1];
        var += (1/(dim-1))*(arr[index1])**2
        index1 +=1;
    xbar = xbar/dim;
    var = var-dim/(dim-1)*(xbar)**2;
    sd_op = var**(0.5);
    
    return sd_op
    
var_seq_tp = twopass(sequence);
var_seq_op = onepass(sequence);
bad_sequence = np.array([1.000000000000001,1.000000000000002,1.000000000000003,1.000000000000003,1.000000000000004,1.000000000000005,1.000000000000006,1.000000000000007,1.000000000000008,1.000000000000009,1.00000000000001,1.000000000000011,1.000000000000012,1.000000000000013]);
a1 = twopass(bad_sequence) - onepass(bad_sequence)