function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% h0x denotes Hyopthesis of X
h0x = sigmoid(X*theta);
% theta without a normalizer for Theta0
theta_reg = theta(2:length(theta));

%cost function
J = (1/m) * sum((-y'*log(h0x) - (1-y')*log(1-h0x))) + (lambda/(m*2)) * (theta_reg'*theta_reg);

% grad function
grad = ((1./m) * (X'*(h0x - y)));
grad(2:length(grad)) = grad(2:length(grad)) + (lambda/m)*theta_reg;




% =============================================================

end
