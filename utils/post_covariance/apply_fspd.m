function A_pd = apply_fspd(A_in, opts)
%% Fixed-support PD correction via shrinkage toward a scaled identity.
%
% This keeps the off-diagonal support of A_in whenever the shrinkage
% coefficient alpha is strictly positive.

    arguments
        A_in double
        opts.min_eig (1, 1) double {mustBePositive} = 1e-4
        opts.mu double = NaN
    end

    % Symmetrize first and inspect the smallest eigenvalue. 
    A = (A_in + A_in') / 2;
    p = size(A, 1);
    eigvals = eig(A);
    lam_min = min(eigvals);

    if lam_min >= opts.min_eig
        A_pd = A;
        return
    end

    % Choose the shrinkage target mu. If the caller does not provide one,
    % use the paper-recommended rule:
    % mu >= mu_SF := max(mu_S, mu_F), where
    % mu_S = max(epsilon, (gamma_max + gamma_min) / 2),
    % mu_F = sum((gamma_i - gamma_min)^2) / sum(gamma_i - gamma_min).
    if isnan(opts.mu)
        gamma = sort(eigvals, 'ascend');
        gamma_min = gamma(1);
        gamma_max = gamma(end);

        mu_S = max(opts.min_eig, (gamma_max + gamma_min) / 2);
        diff_gamma = gamma - gamma_min;
        denom_F = sum(diff_gamma);
        if denom_F <= eps(max(1, norm(diff_gamma, 1)))
            mu_F = mu_S;
        else
            mu_F = sum(diff_gamma .^ 2) / denom_F;
        end
        mu = max(mu_S, mu_F);
    else
        mu = max(opts.mu, opts.min_eig + 1e-6);
    end

    % Compute the smallest alpha in [0,1] such that
    % alpha * A + (1-alpha) * mu I has the requested eigenvalue floor.
    alpha = (mu - opts.min_eig) / max(mu - lam_min, eps);
    alpha = min(max(alpha, 0), 1);

    % Apply the shrinkage and resymmetrize to suppress numerical noise.
    A_pd = alpha * A + (1 - alpha) * mu * eye(p);
    A_pd = (A_pd + A_pd') / 2;

end
