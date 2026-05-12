function [best_lambda1, best_lambda2, phi_hat_best, beta_full_best, cv_all] = ...
    SelectLambdasForBlocks(obj, blocks, cv_index, lambda1_grid, lambda2_grid)
    %% Select (lambda1, lambda2) via LinearDetect-style CV (GGM).
    %
    % This method mirrors the (lambda1, lambda2) selection logic in
    % LinearDetect:::ggm.first.step.blocks().
    %
    % - Enumerate the full grid expand.grid(lambda1, lambda2) with lambda1
    %   varying fastest.
    % - For each (lambda1, lambda2), solve Step I via the MATLAB port of
    %   LinearDetect's C++ solver.
    % - Compute CV error by predicting the last observation of each CV
    %   block for every node j, using the block-wise cumulative
    %   coefficients.
    % - Select the first index that attains the minimum total CV error.
    %
    % - Input:
    %   @blocks:       Block boundaries (1-indexed, includes T+1).
    %   @cv_index:     Block indices used for CV.
    %   @lambda1_grid: Vector of lambda1 values.
    %   @lambda2_grid: Vector of lambda2 values.
    %
    % - Output:
    %   @best_lambda1: Selected lambda1.
    %   @best_lambda2: Selected lambda2.
    %   @phi_hat_best: p-by-((p-1)*n_blocks) increment matrix.
    %   @beta_full_best: Cell array (length n_blocks) of p-by-(p-1) matrices.
    %   @cv_all:       CV scores for all lambda combinations.

arguments
    obj
    blocks (1, :) double {mustBeInteger, mustBePositive}
    cv_index (1, :) double {mustBeInteger, mustBeNonnegative}
    lambda1_grid (1, :) double {mustBeNonnegative}
    lambda2_grid (1, :) double {mustBeNonnegative}
end

T = obj.T;
p = obj.p;
d = p - 1;
n_blocks = numel(blocks) - 1;

lambda1_grid = lambda1_grid(:);
lambda2_grid = lambda2_grid(:);
n_l1 = numel(lambda1_grid);
n_l2 = numel(lambda2_grid);
kk = n_l1 * n_l2;

cv_l = numel(cv_index);
cv_all = zeros(kk, 1);

phi_final = cell(kk, 1);

% LinearDetect warm-starts the Step I solver across the (lambda1, lambda2)
% grid in the expand.grid() order (lambda1 varying fastest).
%
% IMPORTANT (GGM warm-start quirk):
% LinearDetect's C++ implementation for `ggm_break_fit_block()` passes a
% *single* `initial_phi` matrix into the per-node solver
% `lm_break_fit_block_new()`. For GGM, the per-node solver has p_y=1 and
% constructs its initial iterate by reinterpreting the raw memory buffer as
% a 1-by-(d*n_blocks) matrix, i.e. it uses only the first (d*n_blocks)
% entries of `initial_phi` in column-major order. This means the warm start
% is effectively *shared* across nodes and is not simply "row j".
%
% To match LinearDetect, we mirror that behavior by extracting the first
% (d*n_blocks) entries (column-major) from the previous full matrix and
% replicating the resulting row vector across nodes.
phi_prev_full = zeros(p, d * n_blocks);

for i_l2 = 1:n_l2
    for i_l1 = 1:n_l1
        i = (i_l2 - 1) * n_l1 + i_l1;
        lambda1 = lambda1_grid(i_l1);
        lambda2 = lambda2_grid(i_l2);

        % LinearDetect resets the warm start if the previous iterate blew
        % up numerically.
        if max(abs(phi_prev_full), [], "all") > 1e3
            phi_prev_full = zeros(p, d * n_blocks);
        end

        phi_shared = phi_prev_full(:);
        phi_shared = phi_shared(1:(d * n_blocks));
        initial_phi_full = repmat(phi_shared.', p, 1);

        [phi_hat_full, ~, ~] = obj.SolveNeighborhoodSelection( ...
            blocks, lambda1, lambda2, cv_index, initial_phi_full);
        phi_final{i} = phi_hat_full;

        % Warm-start the next grid point from the current solution.
        phi_prev_full = phi_hat_full;

        % LinearDetect evaluates prediction error at the last index of each
        % CV block: blocks[cv.index + 1] - 1 (1-indexed).
        times = blocks(cv_index + 1) - 1;
        if any(times < 1) || any(times > T)
            error("TBFL:SelectLambdasForBlocks:badCvTimes", ...
                "CV times are outside 1..T; check blocks/cv_index.");
        end

        cv_val = 0;
        for j = 1:p
            idx_minus_j = [1:(j - 1), (j + 1):p];
            y = obj.X(:, j);
            X_mj = obj.X(:, idx_minus_j);

            phi_row = phi_hat_full(j, :);
            phi_incr = reshape(phi_row, d, n_blocks);
            beta_blocks = cumsum(phi_incr, 2);

            y_cv = y(times);
            X_cv = X_mj(times, :);
            beta_cv = beta_blocks(:, cv_index).';
            pred = sum(X_cv .* beta_cv, 2);

            cv_val = cv_val + (1 / cv_l) * sum((pred - y_cv).^2);
        end

        cv_all(i) = cv_val;
    end
end

% LinearDetect selects the first index that attains the minimum CV error.
[~, best_idx] = min(cv_all);
best_lambda2 = lambda2_grid(ceil(best_idx / n_l1));
best_lambda1 = lambda1_grid(best_idx - (ceil(best_idx / n_l1) - 1) * n_l1);

phi_hat_best = phi_final{best_idx};

beta_full_best = cell(n_blocks, 1);
beta_full_best{1} = phi_hat_best(:, 1:d);
for b = 2:n_blocks
    beta_full_best{b} = beta_full_best{b - 1} + phi_hat_best(:, (b - 1) * d + 1:b * d);
end
end
