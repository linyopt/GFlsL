% WBSIP: Wild Binary Segmentation through Independent Projection
% (Algorithm 3, Wang, Yu, Rinaldo, 2018).
function result = WBSIP_cov(X, X_prime, s0, e0, Alpha, Beta, tau, delta)
%% WBSIP_cov
% Wild Binary Segmentation for covariance change-points via Independent
% Projection (WBSIP).
% Uses an explicit stack instead of recursion (same splitting behavior).
% Matches Algorithm 3 in Wang, Yu, Rinaldo (2018) and is implemented with an
% explicit stack.
%
% Inputs (paper indexing).
%   @X        : p x n matrix (time on columns, dimensions on rows)
%   @X_prime  : independent copy of X, same size as X (used to estimate PCs)
%   @s0       : left endpoint s (integer, 0-based, inclusive)
%   @e0       : right endpoint e (integer, 0-based, inclusive)
%   @Alpha    : 1 x M vector of random interval starts (integers, 0-based)
%   @Beta     : 1 x M vector of random interval ends   (integers, 0-based)
%   @tau      : threshold parameter τ
%   @delta    : localization-error guard δ (integer)
%
% Output.
%   @result   : struct with fields:
%     S       : estimated change-points (vector)
%     Dval    : corresponding statistics (vector)
%     Level   : detection level for each change (vector)
%     Parent  : 2 x K matrix with [start; end] parent interval per change

    % Precomputation shared across all recursion calls in Algorithm 3.
    [~, n] = size(X);
    Alpha = double(Alpha);
    Beta = double(Beta);
    s0 = double(s0);
    e0 = double(e0);
    delta = double(delta);
    M = numel(Alpha);
    u_mat = PC_cov(X_prime, Alpha, Beta);

    % Project and square: Y_i(u_m) = (u_m' X_i)^2, stored as an M x n matrix.
    y_full = (u_mat.' * X).^2;

    % Prefix sums with a leading 0 column so sums over (s,e] are O(1) under
    % 0-based endpoints.
    pre = [zeros(M, 1), cumsum(y_full, 2)];

    % Accumulators and a stack that mirrors the recursion in Algorithm 3.
    S_cells = {};
    D_cells = {};
    L_cells = {};
    P_cells = {};
    stack = [s0, e0, 0];

    % Process intervals until the stack is empty.
    while ~isempty(stack)
        % Pop one interval (s,e) at a given recursion depth.
        s = stack(end, 1);
        e = stack(end, 2);
        level = stack(end, 3);
        stack(end, :) = [];

        % For the current (s,e), compute a_m and b_m for each random interval m.
        a = -inf(1, M);
        b = nan(1, M);

        % Defensive check: ensure the stack only contains valid endpoints.
        if s < 0 || e > n || ~(s < e)
            error('WBSIP_cov:invalidIndices', 'Require 0 <= s < e <= n.');
        end

        % Intersect with [s,e] and apply delta-trimming.
        % (sm,em) = (ceil(max(alpha,s)+delta), floor(min(beta,e)-delta)).
        Alpha_new = ceil(max(Alpha, s) + delta);
        Beta_new = floor(min(Beta, e) - delta);

        % For each random interval m, compute the maximum absolute univariate
        % CUSUM over admissible split points.
        for m = 1:M
            % Algorithm 3 only searches if (em - sm) is long enough to leave a
            % log(n) margin on both ends.
            if Beta_new(m) - Alpha_new(m) >= 2 * log(n) + 1
                % Candidate t are trimmed by log(n) from both ends and must
                % satisfy Alpha_new < t < Beta_new.
                s_star = ceil(Alpha_new(m) + log(n));
                e_star = floor(Beta_new(m) - log(n));
                s_star = max(s_star, Alpha_new(m) + 1);
                e_star = min(e_star, Beta_new(m) - 1);
                if s_star <= e_star
                    % Vectorize the CUSUM computation across all candidate t
                    % for this m.
                    tspan = s_star:e_star;

                    % Segment lengths and normalization for CUSUM.
                    leftN = tspan - Alpha_new(m);
                    rightN = Beta_new(m) - tspan;
                    denom = (Beta_new(m) - Alpha_new(m));

                    % Means on (Alpha_new, t] and (t, Beta_new] via prefix sums
                    % (0-based endpoints).
                    leftSum = pre(m, tspan + 1) - pre(m, Alpha_new(m) + 1);
                    rightSum = pre(m, Beta_new(m) + 1) - pre(m, tspan + 1);
                    leftMean = leftSum ./ leftN;
                    rightMean = rightSum ./ rightN;

                    % Univariate CUSUM values |Y_t^{sm,em}(u_m)| for all
                    % candidate t.
                    w = sqrt(leftN .* rightN ./ denom);
                    tempVals = w .* abs(leftMean - rightMean);

                    % Keep only the maximum over t for this m.
                    [best_value, k] = max(tempVals);
                    if ~isempty(best_value)
                        a(m) = best_value;
                        b(m) = tspan(k);
                    end
                end
            end
        end

        % If no candidate b_m was computed, stop on this (s,e).
        if all(isnan(b))
            continue;
        end

        % Choose m* that maximizes a_m, then threshold by tau as in
        % Algorithm 3.
        [best_a, m_star] = max(a);
        if best_a <= tau
            continue;
        end

        % Record the detection at b_{m*} and push child intervals (s,b) and
        % (b,e).
        level1 = level + 1;
        parent = [s; e];
        t_best = b(m_star);

        S_cells{end + 1} = t_best;
        D_cells{end + 1} = best_a;
        L_cells{end + 1} = level1;
        P_cells{end + 1} = parent;

        stack(end + 1, :) = [t_best, e, level1];
        stack(end + 1, :) = [s, t_best, level1];
    end

    % Pack outputs.
    result.S = [S_cells{:}];
    result.Dval = [D_cells{:}];
    result.Level = [L_cells{:}];
    result.Parent = [P_cells{:}];
end
