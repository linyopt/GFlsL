function [phi_hat_full, beta_full, flag_full] = SolveNeighborhoodSelection(obj, blocks, lambda1, lambda2, cv_index, initial_phi_full)
    %% Step I: Solve the block fused lasso neighborhood regressions (GGM).
    %
    % This method matches the LinearDetect (R) implementation path:
    % `ggm_break_fit_block()` -> `lm_break_fit_block_new()` (C++).
    %
    % The returned `phi_hat_full` is the increment parameter `theta` in the
    % paper / R code, arranged as:
    %   phi_hat_full(j, ((i-1)*(p-1)+1):(i*(p-1))) = increment for block i.
    %
    % - Input:
    %   @blocks:           Block boundaries (1-indexed, includes T+1).
    %   @lambda1:          Increment sparsity penalty.
    %   @lambda2:          Cumulative sparsity penalty.
    %   @cv_index:         Block indices for CV (drop last obs in each).
    %   @initial_phi_full: Warm start, size p-by-((p-1)*n_blocks).
    %
    % - Output:
    %   @phi_hat_full: p-by-((p-1)*n_blocks) increment matrix.
    %   @beta_full:    Cell array (length n_blocks) of p-by-(p-1) matrices.
    %   @flag_full:    Per-node convergence flags (0 ok, 1 not converged).

    arguments
        obj
        blocks (1, :) double {mustBeInteger, mustBePositive}
        lambda1 (1, 1) double {mustBeNonnegative}
        lambda2 (1, 1) double {mustBeNonnegative}
        cv_index (1, :) double {mustBeInteger, mustBeNonnegative} = []
        initial_phi_full double = []
    end

    p = obj.p;
    d = p - 1;
    n_blocks = numel(blocks) - 1;

    if isempty(initial_phi_full)
        initial_phi_full = zeros(p, d * n_blocks);
    end
    if ~isequal(size(initial_phi_full), [p, d * n_blocks])
        error("TBFL:SolveNeighborhoodSelection:badInitialPhiFull", ...
            "initial_phi_full must be size %d-by-%d.", p, d * n_blocks);
    end

    phi_hat_full = zeros(p, d * n_blocks);
    flag_full = zeros(p, 1);

    % Per-node results (neighborhood selection):
    % Beta{j} is a (p-1)-by-n_blocks coefficient matrix for node j.
    % Delta{j} stores differences across adjacent blocks.
    Beta_cell = cell(p, 1);
    Delta_cell = cell(p, 1);

    for j = 1:p
        % Solve the fused lasso regression for node j. Each node is solved
        % independently, as in the neighborhood selection formulation for
        % GGMs.
        [phi_hat_j, beta_blocks_j, flag_j] = obj.SolveFusedLassoNode( ...
            j, blocks, lambda1, lambda2, cv_index, initial_phi_full(j, :));
        phi_hat_full(j, :) = phi_hat_j;
        Beta_cell{j} = beta_blocks_j;
        flag_full(j) = flag_j;
        if n_blocks > 1
            Delta_cell{j} = diff(beta_blocks_j, 1, 2);
        else
            Delta_cell{j} = zeros(d, 0);
        end
    end

    obj.Beta = Beta_cell;
    obj.Delta = Delta_cell;

    % Convert increments to cumulative coefficients for each block b. This
    % matches the R object `beta.full`, which is used later in Step III.
    beta_full = cell(n_blocks, 1);
    beta_full{1} = phi_hat_full(:, 1:d);
    for b = 2:n_blocks
        beta_full{b} = beta_full{b - 1} + phi_hat_full(:, (b - 1) * d + 1:b * d);
    end
end
