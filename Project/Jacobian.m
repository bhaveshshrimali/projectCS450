clear all; close all; clc; 
syms x1 x2 x3 real; 
F(1) = -1/81*cos(x1) + 1/9 *(x2)^2 + 1/3*(sin(x3));
F(2) = 1/3*(sin(x1)+cos(x3));
F(3) = -1/9*cos(x1)+1/3*(x2)+1/6*(sin(x3));

x = [x1;x2;x3]
for i=1:3
    for j=1:3
        J(i,j) = diff(F(i),x(j));
    end
end

J