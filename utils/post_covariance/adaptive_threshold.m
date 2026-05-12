function Sigma = adaptive_threshold(X_seg, opts)
%% Adaptive thresholding covariance estimator (Cai-Liu-style).
%
% - Usage:
%   Sigma = adaptive_threshold(X_seg, opts);
%
% - Input:
%   @X_seg: Segment data matrix of size n-by-p.
%   @opts: Options struct used by post_covariance_refit().
%
% - Optional input:
%   @center: If true, center X_seg before covariance estimation.
%   @delta: Entrywise threshold multiplier.
%
% - Output:
%   @Sigma: Estimated covariance matrix of size p-by-p.

    % Start from the centered sample covariance on the current segment.
    % We also keep the centered observations because theta_hat is defined
    % through the variability of x_t x_t' around the sample covariance.
    if opts.center
        Xc = X_seg - mean(X_seg, 1);
    else
        Xc = X_seg;
    end
    [n, p] = size(Xc);
    S = (Xc' * Xc) / max(n, 1);
    S = (S + S') / 2;

    % Estimate the entrywise variance proxy theta_hat used by
    % Cai-Liu-style adaptive thresholding.
    theta_hat = zeros(p, p);
    for tt = 1:n
        xt = Xc(tt, :)';
        sample_outer = xt * xt';
        diff_t = sample_outer - S;
        theta_hat = theta_hat + diff_t .^ 2;
    end
    theta_hat = theta_hat / max(n, 1);

    % Convert theta_hat into an entry-specific threshold matrix, then apply
    % thresholding only on the off-diagonal entries.
    lambda_mat = opts.delta * sqrt(theta_hat * log(max(p, 2)) / max(n, 1));
    Sigma = sign(S) .* max(abs(S) - lambda_mat, 0);
    Sigma(1:(p + 1):end) = diag(S);
    Sigma = (Sigma + Sigma') / 2;

end
