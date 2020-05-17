function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_comb = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_comb = [0.01 0.03 0.1 0.3 1 3 10 30];

mat_error = ones(64,3);

i = 1;

for C=C_comb
  for sigma=sigma_comb
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model,Xval);
    predic_error = mean(double(predictions ~= yval));
    mat_error(i,:) = [C sigma predic_error];
    i = i+1;
  endfor
endfor

%Normally returns one values ie minimum but when asked for 2 values gives index of that value
[result ind] = min(mat_error(:,3));

C = mat_error(ind,1);

sigma = mat_error(ind,2);

% =========================================================================

end
