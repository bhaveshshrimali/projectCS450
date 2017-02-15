clear all; close all; clc;

syms x1 x2 real; 
f = 0.5*(x1^2 - x2)^2 + 0.5*(1-x1)^2;

grad = [diff(f,x1) diff(f,x2)]'
hess = [diff(grad(1),x1) diff(grad(1),x2); 
        diff(grad(2),x1) diff(grad(2),x2)]
    