# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:28:13 2016

@author: bhavesh
"""

import numpy as np
from scipy.optimize import fsolve

def function(a):
    a1,a2 = a
    out=[a1+a2-1.0,2*a2-10*(a1*0.5+a2*0.25)**3-3*(a1*0.5+a2*0.25)-0.25]
    return out
    
socol=fsolve(function,[1,1])