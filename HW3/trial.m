clear all; close all; clc;
syms gam real

F = [1 gam 0; 0 1 0; 0 0 1];
B = (inv(F))
B = simplify(B')

P = [0 gam 0; gam 0 0; 0 0 0];
P*F'