function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------
%  x1 = X(:,1);
 % x2 = X(:,2);
%  mean1 = (sum(x1)/length(x1));
 % range1 = max(x1)-min(x1);
%  x1 = (x1-mean1)/(range1);
 % mean2 = (sum(x2)/length(x2));
%  range2 = max(x2)-min(x2);
 % x2 = (x2-mean2)/(range2);
%  X = [x1 x2];
  theta = pinv(X' *X) * X' * y;
%  theta0 = theta0 - alpha*(1/m)*
% -------------------------------------------------------------


% ============================================================

end
