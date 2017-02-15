clear all; close all; clc;
disp('Implementing the Householder transformation column by column')

A = [1 1 1; 1 2 4; 1 3 9; 1 4 16]
for i = 1:3
    if A(i,i) > 0
        v = [zeros(i-1,1); A(i:4,i)] + norm([zeros(i-1,1); A(i:4,i)])*[zeros(1,i-1) 1 zeros(1,4-i)]'
    else
        v = [zeros(i-1,1); A(i:4,i)] - norm([zeros(i-1,1); A(i:4,i)])*[zeros(1,i-1) 1 zeros(1,4-i)]'
    end  
    A(:,1) = A(:,1)-2*((v'*A(:,1))/(v'*v))*v;
    A(:,2) = A(:,2)-2*((v'*A(:,2))/(v'*v))*v;
    A(:,3) = A(:,3)-2*((v'*A(:,3))/(v'*v))*v;
    A

end

trial = [1;2;3;4];
v1 = [3;1;1;1];
trial-((v1'*trial)/(v1'*v1))*v1

disp('Implementing the Householder transformation as a Matrix')

A = [1 1 1; 1 2 4; 1 3 9; 1 4 16]
for i = 1:3
    if A(i,i) > 0
        v = [zeros(i-1,1); A(i:4,i)] + norm([zeros(i-1,1); A(i:4,i)])*[zeros(1,i-1) 1 zeros(1,4-i)]'
    else
        v = [zeros(i-1,1); A(i:4,i)] - norm([zeros(i-1,1); A(i:4,i)])*[zeros(1,i-1) 1 zeros(1,4-i)]';
    end  
    H = eye(4) - (2/(v'*v))*(v*v');
    H
    A = H*A;
    A
end

