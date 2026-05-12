function Sigma = convex_banding(X_seg, opts)
%% Convex banding via Algorithm 1 in Bien-Bunea-Xiao (2016).
%
% - Usage:
%   Sigma = convex_banding(X_seg, opts);
%
% - Input:
%   @X_seg: Segment data matrix of size n-by-p.
%   @opts: Options struct used by post_covariance_refit().
%
% - Optional input:
%   @center: If true, center X_seg before covariance estimation.
%   @lambda: Fixed convex-banding penalty. If provided, use it directly.
%   @lambda_scale: Constant x in the theory-style default
%       lambda = 2 * x * sqrt(log(p) / n).
%
% - Output:
%   @Sigma: Estimated covariance matrix of size p-by-p.

    % Compute the segment sample covariance on the current segment.
    if opts.center
        Xc = X_seg - mean(X_seg, 1);
    else
        Xc = X_seg;
    end
    S = (Xc' * Xc) / size(Xc, 1);
    S = (S + S') / 2;
    [n, p] = size(X_seg);

    % Build the general hierarchical weights from equation (genw):
    % w_{ell,m} = sqrt(2 ell) / (ell - m + 1), for m <= ell.
    W = zeros(max(p - 1, 0));
    for ell = 1:(p - 1)
        W(ell, 1:ell) = sqrt(2 * ell) ./ (ell - (1:ell) + 1);
    end

    % Resolve lambda using the theory-style rate requested for this repo.
    % If opts.lambda is supplied, it overrides the default formula.
    if ~all(isnan(opts.lambda(:)))
        lambda = max(opts.lambda(1), 0);
    else
        lambda = 2 * opts.lambda_scale * sqrt(log(max(p, 2)) / max(n, 1));
        lambda = max(lambda, 0);
    end

    % Handle the zero-penalty boundary case explicitly.
    % With lambda = 0, the convex banding estimator reduces to S.
    if lambda == 0
        Sigma = S;
        return
    end

    % Initialize the dual blocks A_hat^{(ell)} to zero.
    % We then run the one-pass block coordinate descent from Algorithm 1.
    %
    % Important indexing note:
    % the paper numbers subdiagonals from the outside in:
    %   s_m = {(j, k): |j - k| = p - m}.
    % Therefore, subdiagonal s_m lives at MATLAB offset (p - m), not m.
    Ahat = zeros(p, p, max(p - 1, 0));
    for ell = 1:(p - 1)
        % Compute R_hat^{(ell)} = S - lambda * sum_{ell'=1}^{p-1}
        % W^{(ell')} .* A_hat^{(ell')}.
        Rhat = S;
        for ell_prime = 1:(p - 1)
            if nnz(Ahat(:, :, ell_prime)) > 0
                Rhat = Rhat - lambda * ...
                    convex_banding_weighted_block(Ahat(:, :, ell_prime), W(ell_prime, 1:ell_prime));
            end
        end
        Rhat = (Rhat + Rhat') / 2;

        % Evaluate h_ell(0). If h_ell(0) <= lambda^2, the ellipsoid
        % projection lands inside the feasible set, so the update uses
        % max(nu_hat_ell, 0) = 0 rather than skipping the block.
        h0 = convex_banding_h_of_nu(Rhat, W(ell, 1:ell), p, 0);
        if h0 <= lambda ^ 2
            nu = 0;
        else
            % Otherwise solve lambda^2 = h_ell(nu) for nu >= 0.
            nu_upper = 1;
            while convex_banding_h_of_nu(Rhat, W(ell, 1:ell), p, nu_upper) > lambda ^ 2
                nu_upper = 2 * nu_upper;
            end
            nu = fzero(@(nu_) convex_banding_h_of_nu(Rhat, W(ell, 1:ell), p, nu_) - lambda ^ 2, ...
                [0, nu_upper]);
        end

        % Update the current dual block on subdiagonals s_m, m <= ell.
        Ahat(:, :, ell) = zeros(p);
        for m = 1:ell
            offset = p - m;
            scale = W(ell, m) / (lambda * (W(ell, m) ^ 2 + max(nu, 0)));
            subdiag_vec = scale * diag(Rhat, offset);
            Ahat(:, :, ell) = convex_banding_set_subdiag(Ahat(:, :, ell), offset, subdiag_vec);
        end
    end

    % Recover the primal estimator
    % Sigma_hat = R_hat^{(p)} = S - lambda * sum_ell W^{(ell)} .* A_hat^{(ell)}.
    Sigma = S;
    for ell = 1:(p - 1)
        if nnz(Ahat(:, :, ell)) > 0
            Sigma = Sigma - lambda * convex_banding_weighted_block(Ahat(:, :, ell), W(ell, 1:ell));
        end
    end
    Sigma = (Sigma + Sigma') / 2;

end

function h_val = convex_banding_h_of_nu(Rhat, w_row, p, nu)
%% Evaluate h_l(nu) from the dual BCD update.

    ell = numel(w_row);
    h_val = 0;
    for m = 1:ell
        offset = p - m;
        subdiag_norm = sqrt(2) * norm(diag(Rhat, offset), 2);
        h_val = h_val + (w_row(m) ^ 2) * (subdiag_norm ^ 2) / (w_row(m) ^ 2 + nu) ^ 2;
    end
end

function A = convex_banding_set_subdiag(A_in, offset, values)
%% Write one symmetric subdiagonal and its transpose into a matrix.

    A = A_in;
    p = size(A, 1);
    idx = 1:(p - offset);
    A(sub2ind([p, p], idx, idx + offset)) = values(:);
    A(sub2ind([p, p], idx + offset, idx)) = values(:);
end

function WA = convex_banding_weighted_block(A, w_row)
%% Form W^{(l)} .* A for one dual block.

    p = size(A, 1);
    WA = zeros(p);
    ell = numel(w_row);
    for m = 1:ell
        offset = p - m;
        subdiag_vec = w_row(m) * diag(A, offset);
        WA = convex_banding_set_subdiag(WA, offset, subdiag_vec);
    end
end
