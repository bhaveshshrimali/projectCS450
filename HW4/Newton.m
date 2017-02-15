clear all; close all; clc;

fun1 = @F1;
xo = [1.0,1.0];
x1 = fsolve(fun1,xo);

fun2 = @root2d;
xp = [1.0,1.0];
x2 = fsolve(fun2,xp);

