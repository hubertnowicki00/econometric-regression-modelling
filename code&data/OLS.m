function [beta, L, se] = OLS(X,Y)
beta = (X'*X)^-1*X'*Y; % estimaton of parameters
e = Y-X*beta; % fit of the model - errors
L = e'*e; % sum(e.^2)
se = sqrt(var(e).*diag(inv(X'*X))); %portion of total variance explained by variables
end
