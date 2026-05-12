function Sigma = ec2(X_seg, opts)
%% EC2: sparse correlation estimation with eigenvalue constraints.
%
% - Usage:
%   Sigma = ec2(X_seg, opts);
%
% - Input:
%   @X_seg: Segment data matrix of size n-by-p.
%   @opts: Options struct used by post_covariance_refit().
%
% - Optional input:
%   @center: If true, center X_seg before covariance estimation.
%   @lambda: Correlation-scale threshold. If NaN, use the default rate.
%   @lambda_scale: Multiplier for the default threshold when @lambda is NaN.
%   @rho: ADMM penalty parameter.
%   @maxiter: Maximum number of ADMM iterations.
%   @tol: Convergence tolerance for the ADMM residuals.
%   @min_eig: Eigenvalue floor imposed on correlation and covariance matrices.
%
% - Output:
%   @Sigma: Estimated covariance matrix of size p-by-p.

    % Work on the correlation scale because EC2 regularizes correlation
    % entries rather than raw covariance entries.
    if opts.center
        Xc = X_seg - mean(X_seg, 1);
    else
        Xc = X_seg;
    end
    S = (Xc' * Xc) / size(Xc, 1);
    S = (S + S') / 2;
    [n, p] = size(X_seg);
    [R, scales] = cov_to_corr(S);
    lambda = resolve_correlation_lambda(opts.lambda, opts.lambda_scale, n, p);

    % Initialize the ADMM iterates in the notation of (3.4)-(3.7):
    % - Sigma_k: sparse correlation iterate
    % - Gamma_k: eigenvalue-constrained auxiliary iterate
    % - U_k: dual variable (unscaled form)
    Sigma_k = R;
    Gamma_k = R;
    U_k = zeros(p);
    % Alternate between equations (3.4), (3.6), and (3.7):
    % 1) Sigma update by soft-thresholding with fixed diagonal 1,
    % 2) Gamma update by projection onto {lambda_min >= tau},
    % 3) dual update U <- U + rho (Gamma - Sigma).
    for iter = 1:opts.maxiter
        Sigma_prev = Sigma_k;

        % Step 1: Sigma^{t+1} from equation (3.5).
        Sigma_tilde = Gamma_k + U_k / opts.rho;
        Sigma_k = sign(Sigma_tilde) .* max(abs(Sigma_tilde) - lambda / opts.rho, 0);
        Sigma_k(1:(p + 1):end) = 1;
        Sigma_k = (Sigma_k + Sigma_k') / 2;

        % Step 2: Gamma^{t+1} from equation (3.6).
        Gamma_target = (R + opts.rho * Sigma_k - U_k) / (1 + opts.rho);
        Gamma_k = project_min_eig(Gamma_target, opts.min_eig);

        % Step 3: U^{t+1} from equation (3.7).
        U_k = U_k + opts.rho * (Gamma_k - Sigma_k);
        U_k = (U_k + U_k') / 2;

        primal_resid = norm(Gamma_k - Sigma_k, 'fro') / max(1, norm(Gamma_k, 'fro'));
        dual_resid = norm(Sigma_k - Sigma_prev, 'fro') / max(1, norm(Sigma_prev, 'fro'));
        if max(primal_resid, dual_resid) < opts.tol
            break
        end
    end

    % Rescale back to covariance using the sample standard deviations.
    Sigma = diag(scales) * Sigma_k * diag(scales);
    Sigma = (Sigma + Sigma') / 2;
end

function [R, scales] = cov_to_corr(S_in, min_scale)
    % Convert a covariance matrix to its correlation-scale representation.
    if nargin < 2
        min_scale = 1e-8;
    end

    S = (S_in + S_in') / 2;
    scales = sqrt(max(diag(S), min_scale));
    inv_scales = 1 ./ scales;
    R = diag(inv_scales) * S * diag(inv_scales);
    R = (R + R') / 2;
    R(1:(size(R, 1) + 1):end) = 1;
end

function lambda = resolve_correlation_lambda(lambda_in, lambda_scale, n, p)
    % Resolve the effective correlation-scale threshold.
    if ~all(isnan(lambda_in(:)))
        lambda = lambda_in;
        return
    end

    lambda = lambda_scale * sqrt(log(max(p, 2)) / max(n, 1));
end
