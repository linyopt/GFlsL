function SelectBlockSize(obj)
    %% Select block size and lambdas via HBIC.
    %
    % This method evaluates HBIC for each block size in obj.block_size_grid
    % and selects the configuration with the minimum HBIC value.
    %
    % Note:
    % The main LinearDetect-aligned pipeline is implemented in `run(obj)`.
    % This method is retained as an alternative entry point and may not
    % match the exact selection logic used by `run(obj)`.
    %
    % - Output:
    %   obj.selected_block_size: Selected block size.
    %   obj.selected_lambda1: Selected lambda1.
    %   obj.selected_lambda2: Selected lambda2.
    %   obj.hbic_vals: HBIC values for each block size in the grid.

    num_sizes = length(obj.block_size_grid);
    obj.hbic_vals = zeros(num_sizes, 1);

    best_lambda1_per_bs = zeros(num_sizes, 1);
    best_lambda2_per_bs = zeros(num_sizes, 1);
    pts_per_bs = cell(num_sizes, 1);
    beta_per_bs = cell(num_sizes, 1);

    if ~isinf(obj.disp_freq)
        fprintf("TBFL: Selecting block size via HBIC...\n");
        fprintf("TBFL: Block size grid: %s\n", mat2str(obj.block_size_grid));
    end

    for i = 1:num_sizes
        bs = obj.block_size_grid(i);

        % Compute lambda1 grid for this block size.
        lambda1_grid = obj.ComputeLambda1Grid(bs);

        % Select best lambdas via CV.
        [best_l1, best_l2] = SelectLambdasForBlockSizeInternal(obj, bs, lambda1_grid);
        best_lambda1_per_bs(i) = best_l1;
        best_lambda2_per_bs(i) = best_l2;

        % Solve with best lambdas.
        obj.SolveNeighborhoodSelection(bs, best_l1, best_l2);
        beta_per_bs{i} = obj.Beta;

        % Hard threshold to get candidates.
        obj.HardThreshold();
        pts_per_bs{i} = obj.candidates;

        % Compute HBIC.
        obj.hbic_vals(i) = ComputeHBICInternal(obj, bs, obj.Beta, obj.candidates);

        if ~isinf(obj.disp_freq)
            fprintf("TBFL: block_size=%d, lambda1=%.4f, lambda2=%.4f, HBIC=%.4f, ncp=%d\n", ...
                bs, best_l1, best_l2, obj.hbic_vals(i), length(obj.candidates));
        end
    end

    % Select block size with minimum HBIC.
    [~, best_idx] = min(obj.hbic_vals);
    obj.selected_block_size = obj.block_size_grid(best_idx);
    obj.selected_lambda1 = best_lambda1_per_bs(best_idx);
    obj.selected_lambda2 = best_lambda2_per_bs(best_idx);
    obj.Beta = beta_per_bs{best_idx};
    obj.candidates = pts_per_bs{best_idx};
    obj.blocks = obj.BuildBlocks(obj.selected_block_size);

    % Recompute jumps_l2 for the selected block size.
    num_blocks = length(obj.blocks) - 1;
    obj.jumps_l2 = zeros(num_blocks - 1, 1);
    for b = 1:(num_blocks - 1)
        max_jump = 0;
        for j = 1:obj.p
            if ~isempty(obj.Delta{j}) && size(obj.Delta{j}, 2) >= b
                jump_j = norm(obj.Delta{j}(:, b), 2);
                max_jump = max(max_jump, jump_j);
            end
        end
        obj.jumps_l2(b) = max_jump;
    end

    if ~isinf(obj.disp_freq)
        fprintf("TBFL: Selected block_size=%d (HBIC=%.4f)\n", ...
            obj.selected_block_size, obj.hbic_vals(best_idx));
    end
end

function [best_lambda1, best_lambda2] = SelectLambdasForBlockSizeInternal(obj, block_size, lambda1_grid)
    %% Select (lambda1, lambda2) for a fixed block size (legacy helper).
    %
    % This helper implements a simple cross-validation loop over a grid.
    % It is kept for compatibility with older experiments and is not used
    % by the main `run(obj)` pipeline.
    %
    % - Input:
    %   @block_size:   Block size (b_n).
    %   @lambda1_grid: Candidate lambda1 values.
    %
    % - Output:
    %   @best_lambda1: Selected lambda1.
    %   @best_lambda2: Selected lambda2.

    blocks = obj.BuildBlocks(block_size);
    num_blocks = length(blocks) - 1;

    % CV indices.
    bbb = floor(num_blocks / 4);
    if bbb < 1, bbb = 1; end
    cv_index = 4:floor(num_blocks / bbb):num_blocks;
    cv_index = cv_index(cv_index <= num_blocks);
    if isempty(cv_index), cv_index = min(4, num_blocks); end

    best_cv_error = inf;
    best_lambda1 = lambda1_grid(1);
    best_lambda2 = obj.lambda2_cv(1);

    for l1 = lambda1_grid
        for l2 = obj.lambda2_cv
            cv_error = 0;

            for j = 1:obj.p
                idx_minus_j = [1:(j-1), (j+1):obj.p];
                y_full = obj.X(:, j);
                X_full = obj.X(:, idx_minus_j);

                for cv_b = cv_index
                    % Build training data by excluding the CV block.
                    % Preallocate to avoid dynamic growth in a tight loop.
                    block_lens = diff(blocks);
                    n_train = sum(block_lens) - block_lens(cv_b);

                    train_y = zeros(n_train, 1);
                    train_X = zeros(n_train, size(X_full, 2));

                    pos = 1;
                    for b = 1:num_blocks
                        if b == cv_b
                            continue;
                        end
                        I_b = blocks(b):(blocks(b + 1) - 1);
                        m = numel(I_b);
                        train_y(pos:(pos + m - 1)) = y_full(I_b);
                        train_X(pos:(pos + m - 1), :) = X_full(I_b, :);
                        pos = pos + m;
                    end

                    if size(train_X, 1) > size(train_X, 2)
                        XtX = train_X' * train_X;
                        Xty = train_X' * train_y;
                        n_train = size(train_X, 1);

                        beta_train = zeros(obj.p - 1, 1);
                        for iter = 1:20
                            for k = 1:(obj.p - 1)
                                r_k = Xty(k) - XtX(k, :) * beta_train + XtX(k, k) * beta_train(k);
                                beta_train(k) = TBFL.SoftThreshold(r_k / XtX(k, k), ...
                                    l1 * n_train / XtX(k, k));
                            end
                        end

                        I_cv = blocks(cv_b):(blocks(cv_b+1) - 1);
                        y_cv = y_full(I_cv);
                        X_cv = X_full(I_cv, :);
                        pred = X_cv * beta_train;
                        cv_error = cv_error + sum((y_cv - pred).^2);
                    end
                end
            end

            if cv_error < best_cv_error
                best_cv_error = cv_error;
                best_lambda1 = l1;
                best_lambda2 = l2;
            end
        end
    end
end

function hbic = ComputeHBICInternal(obj, block_size, Beta, candidates)
    %% Compute HBIC for a given configuration (legacy helper).
    %
    % This helper computes an HBIC-like score for one (block_size, Beta,
    % candidates) configuration. It is not the primary HBIC implementation
    % used by `run(obj)`.
    %
    % HBIC = sum_j [ T * log(RSS_j / T) ] + 2 * gamma * log(p-1) * count
    %
    % where count is the number of non-zero coefficients.

    blocks = obj.BuildBlocks(block_size);
    num_blocks = length(blocks) - 1;
    m_hat = length(candidates) + 1;

    % Compute change point boundaries.
    if isempty(candidates)
        brk_full = [1, obj.T + 1];
    else
        brk_full = [1, candidates(:)', obj.T + 1];
    end

    hbic_per_node = zeros(obj.p, 1);

    for j = 1:obj.p
        idx_minus_j = [1:(j-1), (j+1):obj.p];
        y_full = obj.X(:, j);
        X_full = obj.X(:, idx_minus_j);

        % Get beta for this node.
        if ~isempty(Beta{j})
            beta_j = Beta{j};
        else
            beta_j = zeros(obj.p - 1, num_blocks);
        end

        % Compute residuals for each segment.
        % Preallocate residuals for this node across all segments.
        residual_full = zeros(obj.T, 1);
        pos = 1;
        for seg = 1:m_hat
            seg_start = brk_full(seg);
            seg_end = brk_full(seg + 1) - 1;
            I_seg = seg_start:seg_end;

            if isempty(I_seg)
                continue;
            end

            y_seg = y_full(I_seg);
            X_seg = X_full(I_seg, :);

            % Find which block this segment corresponds to.
            % Use the middle of the segment.
            mid_seg = round((seg_start + seg_end) / 2);
            block_idx = find(blocks(1:end-1) <= mid_seg & blocks(2:end) > mid_seg, 1);
            if isempty(block_idx)
                block_idx = 1;
            end

            if block_idx <= size(beta_j, 2)
                beta_seg = beta_j(:, block_idx);
            else
                beta_seg = zeros(obj.p - 1, 1);
            end

            pred = X_seg * beta_seg;
            r = y_seg - pred;
            residual_full(pos:(pos + numel(r) - 1)) = r;
            pos = pos + numel(r);
        end

        % Compute RSS.
        rss = sum(residual_full(1:(pos - 1)).^2);
        rss = max(rss, 1e-10);

        % Count non-zero coefficients.
        count = sum(abs(beta_j(:)) > 1e-8);

        % HBIC for this node.
        hbic_per_node(j) = obj.T * log(rss / obj.T) + ...
            2 * obj.gamma_val * log(obj.p - 1) * count;
    end

    hbic = sum(hbic_per_node);
end
