function lambda1_grid = ComputeLambda1Grid(obj, blocks_or_block_size, cv_index)
    %% Compute the LinearDetect lambda1 grid (GGM / lm warm-up).
    %
    % This matches the R code path:
    %   lambda.1.max <- lambda_warm_up_lm(data_y, data_x, blocks,
    %                                    cv.index)$lambda_1_max
    %   epsilon <- 1e-3 if blocks[2] <= p.x + p.y else 1e-4
    %   lambda.1.cv <- log-grid from lambda.1.max to lambda.1.max*epsilon
    %
    % Intuition:
    % lambda_1_max is the smallest lambda1 for which the Step I solution is
    % all-zeros. LinearDetect computes it from blockwise cross-products in
    % a way that is compatible with its C++ solver.
    %
    % For GGM, LinearDetect uses data_x = data_y and does NOT remove the
    % response column when computing lambda.1.max. We replicate that
    % behavior for maximum compatibility.
    %
    % - Input:
    %   @blocks_or_block_size: Either a block boundary vector (1-indexed,
    %                          includes T+1) or a scalar block size.
    %   @cv_index:             Block indices used for CV (drop last obs).
    %
    % - Output:
    %   @lambda1_grid: 1-by-10 vector of lambda1 values (descending).

    arguments
        obj
        blocks_or_block_size
        cv_index (1, :) double {mustBeInteger, mustBeNonnegative} = []
    end

    if isscalar(blocks_or_block_size)
        blocks = obj.BuildBlocks(blocks_or_block_size);
    else
        blocks = blocks_or_block_size;
    end

    T = obj.T;
    p = obj.p;
    n_blocks = numel(blocks) - 1;

    y_b = cell(n_blocks, 1);
    X_b = cell(n_blocks, 1);

    X_all = obj.X.';
    for b = 1:n_blocks
        I_b = blocks(b):(blocks(b + 1) - 1);
        y_b{b} = obj.X(I_b, :);
        X_b{b} = X_all(:, I_b);
    end

    if ~isempty(cv_index)
        for t = 1:numel(cv_index)
            b = cv_index(t);
            ybt = y_b{b};
            Xbt = X_b{b};
            if size(ybt, 1) <= 1
                error("TBFL:ComputeLambda1Grid:cvBlockTooSmall", ...
                    "CV block %d has <=1 observation after blocking.", b);
            end
            y_b{b} = ybt(1:end-1, :);
            X_b{b} = Xbt(:, 1:end-1);
        end
    end

    C_prefix = cell(n_blocks, 1);
    C_prefix{1} = X_b{1} * y_b{1};
    for b = 2:n_blocks
        C_prefix{b} = C_prefix{b - 1} + X_b{b} * y_b{b};
    end
    C_total = C_prefix{n_blocks};

    lambda_1_max = 0;
    for b = 1:n_blocks
        if b == 1
            C_suf = C_total;
        else
            C_suf = C_total - C_prefix{b - 1};
        end
        lambda_1_max = max(lambda_1_max, max(abs(C_suf), [], "all"));
    end

    if blocks(2) <= (2 * p)
        epsilon = 1e-3;
    else
        epsilon = 1e-4;
    end

    nlam = 10;
    if lambda_1_max <= 0
        lambda1_grid = zeros(1, nlam);
        return;
    end

    lambda_1_min = lambda_1_max * epsilon;
    delta_lam = (log(lambda_1_max) - log(lambda_1_min)) / (nlam - 1);
    lambda1_grid = zeros(1, nlam);
    for i = 1:nlam
        lambda1_grid(i) = lambda_1_min * exp(delta_lam * (nlam - i));
    end
end
