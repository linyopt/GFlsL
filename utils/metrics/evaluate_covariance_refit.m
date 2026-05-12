function [HD, F1, acc, error_val, nbreak] = evaluate_covariance_refit(Sigma_t, breaks, trueSigma, trueBreaks)
%% Evaluate a post-refit covariance path against the truth.
%
% Usage:
%   [HD, F1, acc, error_val, nbreak] = evaluate_covariance_refit( ...
%       Sigma_t, breaks, trueSigma, trueBreaks);
%
% Input:
%   @Sigma_t: Estimated covariance path of size p-by-p-by-T.
%   @breaks: Estimated break locations.
%   @trueSigma: True covariance path of size p-by-p-by-T.
%   @trueBreaks: True break locations.
%
% Output:
%   @HD: Hausdorff distance between true and estimated breaks.
%   @F1: Support F1 score for the covariance path.
%   @acc: Support accuracy for the covariance path.
%   @error_val: Frobenius-path error, matching compare_true.
%   @nbreak: Number of estimated breaks.

    [~, p, T] = size(Sigma_t);
    param = p ^ 2;

    % Compute break localization, Frobenius-path error, support F1,
    % support accuracy, and the number of estimated breaks.
    HD = compute_HD(trueBreaks, breaks);
    error_val = sqrt(sum(vecnorm(reshape(Sigma_t - trueSigma, [], T)).^2) / param / T);
    F1 = F1score(trueSigma, Sigma_t);
    acc = sum((trueSigma & abs(Sigma_t) >= 1e-5) | ...
        (~trueSigma & abs(Sigma_t) <= 1e-5), 'all') / (param * T);
    nbreak = numel(breaks);
end
