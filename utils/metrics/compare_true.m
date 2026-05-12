function [HD, F1, acc, error] = compare_true(Est, trueTheta, trueBreaks)
%% Compare the obtained estimators and the true estimator, and compute the 
% metrics, including HD, error, F1.
%
% - Usage:
% [HD, F1, acc, error] = compare_true(Est, trueTheta, trueBreaks)
%
% - Input:
%   @Est:       The GFlsL estimator.
%   @trueTheta: The true Theta.
%   @trueBreaks:The true breaks.
%
% - Output:
%   @HD_DTr:    The HD for GFlsL estimator.
%   @F1_DTr:    The F1 for GFlsL estimator.
%   @acc_DTr:   The accuracy for GFlsL estimator.
%   @error_DTr: The error for GFlsL estimator.

% Obtain the sample size T and the dimension p.
[~, p, T] = size(Est.Theta_k);

% Compute the number of all parameters for each matrix.
param = p^2;

% Estimate the breaks obtained by our model and algorithm.
breaks = Est.EstBreaks();

% Compute HD using the compute_HD utility function.
HD = compute_HD(trueBreaks, breaks);

% Compute the averaged errors with respect to Theta.
error = sqrt(sum(vecnorm(reshape(Est.Theta_k - trueTheta, [], T)).^2) / param / T);

% Compute the F1 score.
F1 = F1score(trueTheta, Est.Theta_k);

% Compute the accuracy.
acc = sum((trueTheta & abs(Est.Theta_k) >= 1e-5) | ...
          (~trueTheta & abs(Est.Theta_k) <= 1e-5), 'all') ...
       / (param * T);
end