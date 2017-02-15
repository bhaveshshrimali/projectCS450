clear all ; close all ; clc;

x=linspace(-100,100,100);y=x';
[X,Y]=meshgrid(x,y);
z = (X).^4 + Y.^4 - 4*X.^2*Y.^2 ;
surf(x,y,z);
xlabel('x','FontWeight','bold','FontSize',22);
ylabel('y','FontWeight','bold','FontSize',22);
zlabel('(x)^4 + y^4 - 4x^2y^2 ','FontWeight','bold','FontSize',22);
title('Part(b): Surface plot of the given function','FontWeight','bold','FontSize',26)