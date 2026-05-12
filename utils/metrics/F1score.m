function score = F1score(truth, est)
%% Compute the F1 score defined by
%                  2 TP
%   score = ------------------.
%             2 TP + FN + FP
% where positive means the element is nonzero, negative means the element
% is zero.
% 
% - Usage:
%   score = F1score(truth, est);
%
% - Input:
%   @truth:     The true solution.
%   @est:       The estimator.
%
% - Output:
%   @score:     The F1 score.

% For estimator, first let all sufficiently small elements be 0.
est(abs(est) <= 1e-5) = 0;

% Compute the four metrics used to compute F1 score.
TP = sum(truth & est, 'all');
FN = sum(truth & ~est, 'all');
FP = sum(~truth & est, 'all');
% TN = sum(~truth & ~est, 'all');

% Compute the F1 score.
score = (2 * TP) / ( 2 * TP + FN + FP );
end