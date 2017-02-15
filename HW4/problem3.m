clear all; close all; clc;

syms x y real 

F(1) = x^2 - 4*x*y + y^2;
F(2) = x^4 - 4*x*y + y^4;
F(3) = 2*x^3 - 3*x^2 - 6*x*y*(x-y-1);
F(4) = (x-y)^4 + x^2 - y^2 - 2*x + 2*y + 1;


for i = 1:4
    grad = [diff(F(i),x) diff(F(i),y)]'
    hess = [diff(grad(1),x) diff(grad(1),y); diff(grad(2),x) diff(grad(2),y)]
end