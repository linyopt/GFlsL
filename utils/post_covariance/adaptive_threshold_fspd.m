function Sigma = adaptive_threshold_fspd(X_seg, opts)
%% Adaptive thresholding + fixed-support PD correction.
%
% - Usage:
%   Sigma = adaptive_threshold_fspd(X_seg, opts);
%
% - Input:
%   @X_seg: Segment data matrix of size n-by-p.
%   @opts: Options struct used by post_covariance_refit().
%
% - Optional input:
%   @center: If true, center X_seg before covariance estimation.
%   @delta: Entrywise threshold multiplier.
%   @min_eig: Eigenvalue floor imposed by the final PD correction.
%
% - Output:
%   @Sigma: Estimated covariance matrix of size p-by-p.

    % Use adaptive_threshold() for the first stage.
    Sigma_raw = adaptive_threshold(X_seg, opts);

    % Repair the thresholded matrix so the final estimator is positive
    % definite while keeping the support interpretation as much as possible.
    Sigma = apply_fspd(Sigma_raw, min_eig=opts.min_eig);
end
