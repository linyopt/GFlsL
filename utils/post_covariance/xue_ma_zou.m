function Sigma = xue_ma_zou(X_seg, opts)
%% Xue-Ma-Zou (2012) positive definite sparse covariance via ADMM.
%
% - Usage:
%   Sigma = xue_ma_zou(X_seg, opts);
%
% - Input:
%   @X_seg: Segment data matrix of size n-by-p.
%   @opts: Options struct used by post_covariance_refit().
%
% - Optional input:
%   @center: If true, center X_seg before covariance estimation.
%   @lambda: Covariance-scale threshold. If NaN, use the default rate.
%   @lambda_scale: Multiplier for the default threshold when @lambda is NaN.
%   @mu: ADMM penalty parameter.
%   @maxiter: Maximum number of ADMM iterations.
%   @tol: Convergence tolerance for the ADMM residuals.
%   @min_eig: Epsilon in the constraint set {Theta >= epsilon I}.
%
% - Output:
%   @Sigma: Estimated covariance matrix of size p-by-p.

    % Compute the segment sample covariance \hat\Sigma_n and resolve the
    % effective covariance-scale regularization parameter lambda.
    if opts.center
        Xc = X_seg - mean(X_seg, 1);
    else
        Xc = X_seg;
    end
    Sigma_n = (Xc' * Xc) / size(Xc, 1);
    Sigma_n = (Sigma_n + Sigma_n') / 2;
    [n, p] = size(X_seg);
    if ~all(isnan(opts.lambda(:)))
        lambda = opts.lambda;
    else
        scale = max(mean(diag(Sigma_n)), 1e-6);
        lambda = opts.lambda_scale * scale * sqrt(log(max(p, 2)) / max(n, 1));
    end
    mu = opts.mu;
    epsilon = opts.min_eig;

    % Build the soft-thresholding estimator used by the paper as the
    % initial value for both Theta^0 and Sigma^0.
    Sigma_soft = sign(Sigma_n) .* max(abs(Sigma_n) - lambda, 0);
    Sigma_soft(1:(p + 1):end) = diag(Sigma_n);
    Sigma_soft = (Sigma_soft + Sigma_soft') / 2;

    % As stated in the paper's implementation note: if the soft-threshold
    % estimator is already positive definite, return it directly.
    if min(eig(Sigma_soft)) >= epsilon
        Sigma = Sigma_soft;
        return
    end

    % Follow Algorithm 1 notation in Xue-Ma-Zou (2012):
    % - Sigma_k is the sparse covariance iterate
    % - Theta_k is the positive-definite auxiliary iterate
    % - Lambda_k is the Lagrange multiplier iterate
    Sigma_k = Sigma_soft;
    Theta_k = Sigma_soft;
    Lambda_k = zeros(p);
    % Alternate between:
    % 1) Theta update: Theta^{k+1} = (Sigma^k + mu * Lambda^k)_+,
    % 2) Sigma update: Sigma^{k+1} = S(mu*(Sigma_n - Lambda^k)+Theta^{k+1}, lambda*mu)/(1+mu),
    % 3) Lambda update: Lambda^{k+1} = Lambda^k - (Theta^{k+1}-Sigma^{k+1})/mu.
    for iter = 1:opts.maxiter
        Sigma_prev = Sigma_k;

        % Project onto {Theta >= epsilon I}, matching (.)_+ defined in the
        % paper for the Theta step.
        Theta_k = project_min_eig(Sigma_k + mu * Lambda_k, epsilon);

        Sigma_k = mu * (Sigma_n - Lambda_k) + Theta_k;
        Sigma_k = sign(Sigma_k) .* max(abs(Sigma_k) - lambda * mu, 0);
        Sigma_k(1:(p + 1):end) = diag(mu * (Sigma_n - Lambda_k) + Theta_k);
        Sigma_k = Sigma_k / (1 + mu);
        Sigma_k = (Sigma_k + Sigma_k') / 2;

        Lambda_k = Lambda_k - (Theta_k - Sigma_k) / mu;
        Lambda_k = (Lambda_k + Lambda_k') / 2;

        primal_resid = norm(Theta_k - Sigma_k, 'fro') / max(1, norm(Sigma_k, 'fro'));
        dual_resid = norm(Sigma_k - Sigma_prev, 'fro') / max(1, norm(Sigma_prev, 'fro'));
        if max(primal_resid, dual_resid) < opts.tol
            break
        end
    end

    % Use the final sparse iterate as the estimator.
    Sigma_hat = (Sigma_k + Sigma_k') / 2;
    Sigma = Sigma_hat;
end
