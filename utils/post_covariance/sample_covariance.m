function Sigma = sample_covariance(X_seg, opts)
%% Sample covariance refit (with PD projection).
%
% - Usage:
%   Sigma = sample_covariance(X_seg, opts);
%
% - Input:
%   @X_seg: Segment data matrix of size n-by-p.
%   @opts: Options struct used by post_covariance_refit().
%
% - Optional input:
%   @center: If true, center X_seg before covariance estimation.
%   @min_eig: Eigenvalue floor imposed on the sample covariance.
%
% - Output:
%   @Sigma: Estimated covariance matrix of size p-by-p.

    % Compute the empirical covariance for the current segment and enforce a
    % minimal eigenvalue floor so downstream code can safely invert or
    % score the result.
    if opts.center
        Xc = X_seg - mean(X_seg, 1);
    else
        Xc = X_seg;
    end
    S = (Xc' * Xc) / size(Xc, 1);
    S = (S + S') / 2;
    Sigma = project_min_eig(S, opts.min_eig);

end
