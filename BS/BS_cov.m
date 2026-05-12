% BSOP: Binary Segmentation through Operator Norm (Algorithm 1).
% Reference: Wang, Yu, Rinaldo (2018).
function result = BS_cov(X, s0, e0, tau, level0)
%% BS_cov
% Binary Segmentation through Operator Norm (BSOP).
%
% Indexing (paper convention).
% Endpoints satisfy 0 <= s < e <= n and refer to the half-open sample set
% {X_i}_{i=s+1,...,e}.
% Returned change points b are boundary indices in {1,...,n-1}.
%
% Inputs.
%   @X       : p x n data matrix (columns are time)
%   @s0      : left endpoint (integer, 0-based, inclusive)
%   @e0      : right endpoint (integer, 0-based, inclusive)
%   @tau     : threshold parameter τ
%   @level0  : (optional) starting level (default 0)
%
% Output.
%   @result: struct with fields:
%     S      : estimated change points (vector)
%     Dval   : corresponding max operator-norm values (vector)
%     Level  : recursion depth for each detection (vector)
%     Parent : 2 x K matrix with [s; e] parent interval per detection

    % Default starting level (bookkeeping for the segmentation tree).
    if nargin < 5
        level0 = 0;
    end

    % Dimensions and endpoint validation under the paper indexing convention.
    [p, n] = size(X);
    if s0 < 0 || e0 > n || ~(s0 < e0)
        error('BS_cov:invalidIndices', 'Require 0 <= s0 < e0 <= n.');
    end

    % Stopping rule length from Algorithm 1.
    % Do not process intervals with e-s <= 2 p log(n) + 1.
    minLen = 2 * p * log(n) + 1;

    % Accumulators and an explicit stack.
    % This stack mirrors the recursion in Algorithm 1.
    S_cells = {};
    D_cells = {};
    L_cells = {};
    P_cells = {};
    stack = [s0, e0, level0];

    % Process intervals until the stack is empty.
    while ~isempty(stack)
        % Pop one interval (s,e) at a given recursion depth.
        s = stack(end, 1);
        e = stack(end, 2);
        level = stack(end, 3);
        stack(end, :) = [];

        % Stop if the interval is too short to be scanned.
        % This corresponds to immediately returning from BSOP(s,e,τ).
        if (e - s) <= minLen
            continue;
        end

        % Candidate split points are trimmed by p log(n) from both ends.
        % Candidate split points must also satisfy the strict interior rule
        % s < t < e.
        s_star = ceil(s + p * log(n));
        e_star = floor(e - p * log(n));
        % Even after trimming, enforce s < t < e explicitly.
        % This prevents degenerate cases where t equals an endpoint, which
        % makes the CUSUM scaling ill-defined.
        s_star = max(s_star, s + 1);
        e_star = min(e_star, e - 1);
        % If trimming leaves no admissible split point, stop on this interval
        % and move on to other pending intervals.
        if s_star > e_star
            continue;
        end

        % Compute a = max_t ||S_t^{s,e}||_op over admissible t.
        % Also track b = argmax_t ||S_t^{s,e}||_op.
        best_val = -inf;
        best_t = NaN;
        for t = s_star:e_star
            S = CUSUM_cov(X, s, e, t);
            val = norm(S, 2);
            if val > best_val
                best_val = val;
                best_t = t;
            end
        end

        % Thresholding step (Algorithm 1).
        % Only split if the maximum exceeds tau.
        % This is the "FLAG <- 1" branch in Algorithm 1, implemented here as
        % "do not push children for this interval".
        % We keep processing remaining intervals on the stack rather than
        % breaking out of the whole loop.
        if best_val <= tau
            continue;
        end

        % Record the detection and push the two child intervals (s,b) and
        % (b,e) onto the stack.
        level1 = level + 1;
        parent = [s; e];
        S_cells{end + 1} = best_t;
        D_cells{end + 1} = best_val;
        L_cells{end + 1} = level1;
        P_cells{end + 1} = parent;

        stack(end + 1, :) = [best_t, e, level1];
        stack(end + 1, :) = [s, best_t, level1];
    end

    % Pack outputs.
    result.S = [S_cells{:}];
    result.Dval = [D_cells{:}];
    result.Level = [L_cells{:}];
    result.Parent = [P_cells{:}];
end
