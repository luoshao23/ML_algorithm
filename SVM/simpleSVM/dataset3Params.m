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
sigma = 0.1;

% choice = [0.01 0.03 0.1 0.3 1 3 10 30];

% best_pair = [-9 -9];
% best_error = 999;

% for C = choice
%     for sigma = choice
%         model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%         predictions = svmPredict(model, Xval);
%         error_ = mean(double(predictions ~= yval));
%         fprintf('(C, sigma)=(%f, %f), error=%f \n', C, sigma, error_);
%         if error_ < best_error
%             best_pair = [C, sigma];
%             best_error = error_;
%         end
%     end
% end

% C = best_pair(1);
% sigma = best_pair(2);

% fprintf('best (C, sigma)=(%f, %f) found with error=%f \n', C, sigma, best_error);


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







% =========================================================================

end
