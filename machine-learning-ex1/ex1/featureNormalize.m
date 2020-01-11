function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
%mu = zeros(1, size(X, 2));
%sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

%mu(:,1) = X(:,1).-mean(X(:,1));
%mu(:,2) = X(:,2).-mean(X(:,2));
%mu(:,3) = X(:,3).-mean(X(:,3));
%sigma(:,1) = mu(:,1)/std(X(:,1));
%sigma(:,2) = mu(:,2)./std(X(:,2));
%sigma(:,3) = mu(:,3)./std(X(:,3));
%sigma = sigma;
%X_norm = sigma;

n = size(X,2);
mu(1)=0;sigma(1)=0;
X_norm(:,1)=0;
for i = 2:n
  avg = mean(X(:, i));
  deviation = std(X(:, i));
  X_norm(:, i) = (X_norm(:, i) - avg)/deviation;
  mu(i) = avg;
  sigma(i) = deviation;
end
X_norme(:,1) = 0;
% ============================================================

end