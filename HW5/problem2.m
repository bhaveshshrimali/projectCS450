clear all; close all; clc;

A = [1 -1 1; 1 0 0 ; 1 1 1];
b = [1;0;1];

x = linsolve(A,b)

A = [1 0 0; 1 1 0 ; 1 2 2];
b = [1;0;1];

x = linsolve(A,b)