function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute and display base cost and gradient
[J, grad] = costFunction(theta, X, y);

reg_theta = 0;
for j = 2:n
  reg_theta = reg_theta + (theta(j) ^ 2);
endfor
reg_theta = reg_theta * lambda / (2 * m);
J = J + reg_theta;

for j=2:n
  grad(j) = grad(j) + (theta(j) * lambda / m);
endfor

% =============================================================

end
