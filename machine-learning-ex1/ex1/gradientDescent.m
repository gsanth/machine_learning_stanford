function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%fprintf('Theta values at the beginning of gradient descent is %f, %f\n', theta(1), theta(2));

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    dJ = 0;
    dJx = 0;
    for i = 1:m
        h = theta(1) + theta(2)*X(i,2);
        dJ = dJ + (h - y(i));
        dJx = dJx + (h - y(i)) * X(i,2);
    endfor

    theta(1) = theta(1) - alpha * dJ / m;
    theta(2) = theta(2) - alpha * dJx / m;

    %fprintf('Theta values for iteration %f is %f, %f\n', iter, theta(1), theta(2));

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %fprintf('Cost for iteration is %f\n', computeCost(X, y, theta));

end

end
