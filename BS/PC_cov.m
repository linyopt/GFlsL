% Compute the projection directions u_m for each random interval.
% This is Algorithm 2 (PC step) in Wang, Yu, Rinaldo (2018).
function u_mat = PC_cov(X, Alpha, Beta)
%% PC_cov
% Principal-component estimation for covariance CUSUM on each random interval.
% Matches Algorithm 2 (PC) in Wang, Yu, Rinaldo (2018).
%
% Inputs (paper indexing).
%   @X      : p x n data matrix with columns X_1,...,X_n
%   @Alpha  : 1 x M vector of starts α_m (integers, 0-based)
%   @Beta   : 1 x M vector of ends   β_m (integers, 0-based)
%
% Output.
%   @u_mat  : p x M matrix; column m is the direction u_m in Algorithm 2.

% Dimensions and the minimum interval-length condition used in Algorithm 2.
[p, n] = size(X);
M = numel(Alpha);
delta = 2 * p * log(n) + 1;

% Preallocate output.
% Intervals failing the length requirement will keep u_m = 0.
u_mat = zeros(p, M);
for m = 1:M
    % Skip intervals that are too short to scan after p log(n) trimming.
    if Beta(m) - Alpha(m) > delta
        % Admissible split points for (Alpha(m), Beta(m)) are trimmed by
        % p log(n) and satisfy Alpha < t < Beta.
        s_star = ceil(Alpha(m) + p * log(n));
        e_star = floor(Beta(m) - p * log(n));
        s_star = max(s_star, Alpha(m) + 1);
        e_star = min(e_star, Beta(m) - 1);
        if s_star <= e_star
            % Find d_m = argmax_t ||S_t^{alpha,beta}||_op.
            best_val = -inf;
            best_t = s_star;
            for t = s_star:e_star
                S = CUSUM_cov(X, Alpha(m), Beta(m), t);
                val = norm(S, 2);
                if val > best_val
                    best_val = val;
                    best_t = t;
                end
            end
            % Set u_m to the maximizer of
            % |v' S_{d_m}^{alpha,beta} v| over ||v||=1.
            % For a symmetric matrix, the maximizer corresponds to the
            % eigenvector whose eigenvalue has the largest magnitude.
            Sbest = CUSUM_cov(X, Alpha(m), Beta(m), best_t);
            Sbest = (Sbest + Sbest.') / 2;
            [V, D] = eig(Sbest);
            [~, idx] = max(abs(diag(D)));
            u_mat(:, m) = V(:, idx);
        end
    end
end
end
