function z = F4(x)
       z(1) = 2*x(1) + 4*(x(1) - x(2))^3 - 2;
       z(2) =   2 - 4*(x(1) - x(2))^3 - 2*x(2);
end
