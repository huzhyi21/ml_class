function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

c_values = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_values = [0.01 0.03 0.1 0.3 1 3 10 30];
%error_index = ones((size(c_values,2)*size(sigma_values,2)),3)*100;
error_index = ones(1,3)*100;

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


for i = 1:size(c_values,2)
  for j = 1:size(sigma_values,2)
    row_index=(i-1)*8+j;
    C = c_values(i);
    %error_index(row_index,1) = C;
    sigma = sigma_values(j);
    %error_index(row_index,2) = sigma;
    fprintf('Training model number %d\n', row_index);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    fprintf('Predicting model number %d\n', row_index);
    predictions = svmPredict(model, Xval);
    %error_index(row_index,3) = mean(double(predictions ~= yval));
    current_error = mean(double(predictions ~= yval));
    if current_error < error_index(3)
      fprintf('Updating Error Index:');
      error_index = [C sigma current_error]
    end
  end
end

C = error_index(1);
sigma = error_index(2);



% =========================================================================

end
