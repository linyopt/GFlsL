function run(obj)
    %% Run TBFL for GGM (aligned to R LinearDetect).
    %
    % This run() method mirrors LinearDetect::tbfl(method="GGM"):
    % - If obj.block_size is empty, search obj.block_size_grid and select
    %   the optimal block size via HBIC (R: optimal.block=TRUE).
    % - For each block size, select (lambda1, lambda2) via CV over a grid
    %   (R: ggm.first.step.blocks()).
    % - Step II: BIC/HBIC-based jump selection + clustering.
    % - Step III: Local exhaustive search refinement.
    %
    % On return, the main outputs are stored in the object:
    %   obj.breaks: Detected change points (time indices in 1..T-1).
    %   obj.beta_hat_list: Segment-wise coefficient matrices (Step III).
    %   obj.selected_*: Selected tuning parameters (block size, lambdas).

    % Use `disp_freq` as a coarse "verbose" switch.
    %
    % Most simulation scripts set `disp_freq=inf` to keep stdout clean.
    % When `disp_freq` is finite, print a small amount of progress info.
    if ~isinf(obj.disp_freq)
        fprintf("TBFL: T=%d, p=%d\n", obj.T, obj.p);
    end

    % Block size selection.
    %
    % LinearDetect's `tbfl()` supports either:
    % 1) `optimal.block=TRUE`: evaluate a grid of block sizes and select
    %    the one with the smallest HBIC (after BIC thresholding), or
    % 2) a user-specified block size: run TBFL once at that block size.
    %
    % In the MATLAB port, `obj.block_size=[]` corresponds to case (1).
    if isempty(obj.block_size)
        % Determine the candidate block size grid.
        %
        % If the user provided `block_size_grid`, use it directly.
        % Otherwise, construct a LinearDetect-style grid based on T and p.
        bs_range = obj.block_size_grid;
        if isempty(bs_range)
            bs_range = obj.ComputeBlockSizeGrid();
        end

        n_method = numel(bs_range);
        hbic_full = zeros(n_method, 1);
        best_pack = cell(n_method, 1);

        % LinearDetect quirk: when optimal.block=TRUE and lambda.1.cv is not
        % user-specified, tbfl() computes lambda.1.cv once (at the first
        % block size in bn.range) and then reuses it for all candidate block
        % sizes. This is because lambda.1.cv is mutated inside the for-loop
        % and the subsequent iterations see is.null(lambda.1.cv)==FALSE.
        %
        % To match the reference behavior, we precompute a shared lambda1
        % grid using the first block size in the range and pass it into each
        % per-block-size run.
        lambda1_grid_shared = [];
        if ~isempty(obj.lambda1_cv)
            lambda1_grid_shared = obj.lambda1_cv;
        else
            bs0 = bs_range(1);
            blocks0 = obj.BuildBlocks(bs0);
            cv_index0 = make_cv_index(numel(blocks0) - 1);
            lambda1_grid_shared = obj.ComputeLambda1Grid(blocks0, cv_index0);
        end
        lambda2_grid_shared = obj.lambda2_cv;

        % Evaluate each candidate block size.
        %
        % Important: `run_for_blocksize()` mutates `obj` because Step II/III
        % store intermediate state on the object. We therefore record a
        % self-contained `pack` for each block size to compare HBIC values
        % without relying on the final `obj` state.
        for k = 1:n_method
            bs = bs_range(k);
            pack = run_for_blocksize(obj, bs, lambda1_grid_shared, lambda2_grid_shared);

            % Apply LinearDetect's final BIC thresholding of coefficients.
            %
            % LinearDetect uses thresholded coefficients (beta.final) when
            % computing HBIC for block size selection. This is distinct from
            % Step II's jump selection, which is performed before threshold.
            pack.beta_final = bic_threshold_ggm(obj, pack.beta_hat_list, pack.cp_final, bs);
            pack.hbic = hbic_optimal_ggm(obj, pack.beta_final, pack.cp_final);
            hbic_full(k) = pack.hbic;
            best_pack{k} = pack;

            if ~isinf(obj.disp_freq)
                fprintf("TBFL: block_size=%d, HBIC=%.6g, ncp=%d\n", ...
                    bs, pack.hbic, numel(pack.cp_final));
            end
        end

        % Select the block size with the smallest HBIC.
        %
        % We store both the selected hyperparameters and the key Step I-III
        % outputs back onto the object for downstream evaluation and plots.
        obj.hbic_vals = hbic_full;
        [~, best_idx] = min(hbic_full);
        pack = best_pack{best_idx};

        obj.selected_block_size = bs_range(best_idx);
        obj.selected_lambda1 = pack.lambda1;
        obj.selected_lambda2 = pack.lambda2;
        obj.blocks = pack.blocks;
        obj.phi_hat_full = pack.phi_hat_full;
        obj.beta_full = pack.beta_full;
        obj.jumps_l2 = pack.jumps_l2;
        obj.candidates = pack.candidates;
        obj.cp_first = pack.cp_first;
        obj.breaks = pack.cp_final;
        obj.beta_hat_list = pack.beta_final;
    else
        % Fixed block size mode (no HBIC-based selection).
        %
        % This path mirrors LinearDetect's behavior when `optimal.block` is
        % disabled and a single block size is specified by the user.
        bs = obj.block_size;
        if numel(bs) ~= 1
            error("TBFL:run:badBlockSize", "block_size must be a scalar.");
        end
        pack = run_for_blocksize(obj, bs);
        obj.selected_block_size = bs;
        obj.selected_lambda1 = pack.lambda1;
        obj.selected_lambda2 = pack.lambda2;
        obj.blocks = pack.blocks;
        obj.phi_hat_full = pack.phi_hat_full;
        obj.beta_full = pack.beta_full;
        obj.jumps_l2 = pack.jumps_l2;
        obj.candidates = pack.candidates;
        obj.cp_first = pack.cp_first;
        obj.breaks = pack.cp_final;

        % Apply the final coefficient thresholding used by LinearDetect.
        %
        % When selecting the block size automatically, this thresholded
        % version is also used for HBIC comparisons. In fixed block size
        % mode, we still return thresholded coefficients to match R outputs.
        obj.beta_hat_list = bic_threshold_ggm(obj, pack.beta_hat_list, pack.cp_final, bs);
    end

    if ~isinf(obj.disp_freq)
        fprintf("TBFL: Selected block_size=%d\n", obj.selected_block_size);
        fprintf("TBFL: Selected lambda1=%.6g, lambda2=%.6g\n", ...
            obj.selected_lambda1, obj.selected_lambda2);
        fprintf("TBFL: Detected %d change points.\n", numel(obj.breaks));
    end

    obj.converged = true;
end

function pack = run_for_blocksize(obj, block_size, lambda1_grid_override, lambda2_grid_override)
    %% Run the full TBFL pipeline for a fixed block size.
    %
    % This helper executes the three TBFL steps for a chosen block size:
    % 1) Step I: Select (lambda1, lambda2) by CV and fit the fused model.
    % 2) Step II: Select candidate change points via hard thresholding.
    % 3) Step III: Locally refine candidates via exhaustive search.
    %
    % - Input:
    %   @block_size: Block size b_n used to build blocks.
    %
    % - Output:
    %   @pack: Struct with intermediate and final results for this b_n.

    % Build equal-length blocks of size `block_size`.
    %
    % The block boundaries define the resolution at which the fused lasso
    % is fit in Step I. Later steps map selected block boundaries back to
    % time indices in 1..T-1.
    blocks = obj.BuildBlocks(block_size);
    n_blocks = numel(blocks) - 1;

    % Construct the cross-validation (CV) index set.
    %
    % LinearDetect uses a deterministic pattern based only on `n_blocks`,
    % not random folds. This is important for matching R outputs exactly.
    % LinearDetect uses a fixed pattern of CV indices based on n_blocks.
    cv_index = make_cv_index(n_blocks);

    % Construct the lambda grids for Step I.
    %
    % In LinearDetect, `lambda2` uses a short fixed grid that scales with
    % sqrt(log(p)/T). `lambda1` uses a data-dependent grid derived from
    % warm-start fits on held-out blocks (see ComputeLambda1Grid()).
    if nargin >= 3 && ~isempty(lambda1_grid_override)
        lambda1_grid = lambda1_grid_override;
    elseif isempty(obj.lambda1_cv)
        lambda1_grid = obj.ComputeLambda1Grid(blocks, cv_index);
    else
        lambda1_grid = obj.lambda1_cv;
    end
    if nargin >= 4 && ~isempty(lambda2_grid_override)
        lambda2_grid = lambda2_grid_override;
    else
        lambda2_grid = obj.lambda2_cv;
    end

    % Step I (Neighborhood selection + block fused lasso).
    %
    % Select (lambda1, lambda2) via CV, then fit the fused model on all
    % blocks. The fitted result is represented both as:
    % - `phi_hat_full`: block-wise increments, and
    % - `beta_full`: cumulative coefficients per block.
    [best_l1, best_l2, phi_hat_best, beta_full_best] = ...
        obj.SelectLambdasForBlocks(blocks, cv_index, lambda1_grid, lambda2_grid);

    % Persist Step I outputs onto the object for Step II/III.
    %
    % Step II/III are implemented as methods that read `obj.beta_full` and
    % write their intermediate results back to the object.
    obj.blocks = blocks;
    obj.phi_hat_full = phi_hat_best;
    obj.beta_full = beta_full_best;

    % Step II: Jump selection + clustering.
    %
    % This computes jump statistics at each block boundary, performs hard
    % thresholding, and clusters nearby candidates into groups that will be
    % refined in Step III.
    obj.HardThreshold();
    if isempty(obj.cp_first)
        % No candidate cluster survived Step II.
        %
        % LinearDetect returns an empty change-point set in this case.
        % For compatibility, we return a single-segment beta list so that
        % downstream HBIC and Omega reconstruction have a well-defined
        % segment representation.
        % LinearDetect returns an empty cp list when no candidates remain.
        cp_final = [];
        beta_hat_list = {beta_full_best{max(1, floor(n_blocks / 2))}};
    else
        % Step III: Local exhaustive search refinement.
        %
        % For each candidate cluster produced by Step II, perform a local
        % search over time indices to refine the change-point location and
        % refit segment-wise coefficients.
        % Step III: Local refinement.
        obj.LocalRefinement();
        cp_final = obj.breaks;
        beta_hat_list = obj.beta_hat_list;
    end

    % Bundle all intermediate state to allow HBIC comparison across b_n.
    %
    % This pack is designed to be self-contained for each block size, so
    % that the outer loop can select by HBIC without depending on the final
    % mutated `obj` state.
    pack = struct();
    pack.block_size = block_size;
    pack.blocks = blocks;
    pack.cv_index = cv_index;
    pack.lambda1 = best_l1;
    pack.lambda2 = best_l2;
    pack.phi_hat_full = phi_hat_best;
    pack.beta_full = beta_full_best;
    pack.jumps_l2 = obj.jumps_l2;
    pack.candidates = obj.candidates;
    pack.cp_first = obj.cp_first;
    pack.cp_final = cp_final;
    pack.beta_hat_list = beta_hat_list;
end

function cv_index = make_cv_index(n_blocks)
    %% Construct the CV block index set (LinearDetect-compatible).
    %
    % LinearDetect chooses CV indices as:
    %   bbb <- floor(n.new/4)
    %   cv.index <- seq(aaa, n.new, floor(n.new/bbb)) with aaa=4
    %
    % - Input:
    %   @n_blocks: Number of blocks (n.new in LinearDetect).
    %
    % - Output:
    %   @cv_index: Vector of block indices used for CV.

    bbb = floor(n_blocks / 4);
    if bbb < 1
        bbb = 1;
    end
    step = floor(n_blocks / bbb);
    if step < 1
        step = 1;
    end
    cv_index = 4:step:n_blocks;
    cv_index = cv_index(cv_index >= 1 & cv_index <= n_blocks);
    if isempty(cv_index)
        cv_index = min(4, n_blocks);
    end
end

function beta_final = bic_threshold_ggm(obj, beta_hat_list, cp_final, b_n)
    %% Threshold segment-wise coefficients using BIC (GGM).
    %
    % This mirrors LinearDetect::BIC.threshold.ggm(), which selects a
    % threshold lambda for each segment by minimizing a BIC score computed
    % on trimmed residuals (drop the first/last b_n samples of the segment).
    %
    % - Input:
    %   @beta_hat_list: Cell array of segment coefficient matrices
    %                  (p-by-(p-1)).
    %   @cp_final:      Detected change points (time indices in 1..T-1).
    %   @b_n:           Block size used for trimming and thresholding.
    %
    % - Output:
    %   @beta_final: Thresholded beta_hat_list (same cell structure).

    TT = obj.T;
    p = obj.p;
    d = p - 1;

    if isempty(beta_hat_list)
        beta_final = {};
        return;
    end

    brk_full = [1; cp_final(:); TT + 1];
    m_hat = numel(brk_full) - 1;
    if numel(beta_hat_list) ~= m_hat
        beta_hat_list = beta_hat_list(1:min(end, m_hat));
        if numel(beta_hat_list) < m_hat
            beta_hat_list{m_hat} = beta_hat_list{end};
        end
    end

    % LinearDetect calls BIC.threshold.ggm(..., nlam=50) in
    % tbfl(method="GGM"). Inside BIC.threshold.ggm, nlam is overwritten
    % in-place when lambda.max / lambda.min >= 1e4, and the new value
    % persists for subsequent segments. We mimic that behavior.
    nlam = 50;
    lambda_best = zeros(m_hat, 1);

    for seg = 1:m_hat
        beta_seg = beta_hat_list{seg};
        temp = beta_seg(:);
        lambda_max = max(abs(temp));
        if lambda_max <= 0
            lambda_best(seg) = 0;
            continue;
        end
        nz = abs(temp(abs(temp) > 0));
        lambda_min = min(nz);

        wide_range = (lambda_max / lambda_min >= 1e4);
        if wide_range
            nlam = 50;
            lambda_min = lambda_max * 1e-4;
        end

        % Log-spaced threshold candidates, descending from lambda_max.
        delta_lam = (log(lambda_max) - log(lambda_min)) / (nlam - 1);
        lambda_vals = zeros(nlam, 1);
        for j = 1:nlam
            lambda_vals(j) = lambda_min * exp(delta_lam * (nlam - j));
        end
        % Guard against tiny floating overshoots outside
        % [lambda_min, lambda_max]. For wide_range segments (where
        % LinearDetect resets lambda_min := lambda_max*1e-4), the R code
        % can intentionally end up with an all-zero segment when the best
        % threshold is at the top of the grid. To match LinearDetect more
        % closely, only clamp in the non-wide_range case.
        if ~wide_range
            lambda_vals = min(lambda_vals, lambda_max);
            lambda_vals = max(lambda_vals, lambda_min);
        end

        lb = brk_full(seg);
        ub = brk_full(seg + 1) - 1;
        data_y_seg = obj.X(lb:ub, :);
        len = size(data_y_seg, 1);

        row_start = b_n;
        row_end = len - b_n;
        if row_end < row_start
            row_start = 1;
            row_end = len;
        end

        BIC_vals = zeros(nlam, 1);
        for j = 1:nlam
            lam = lambda_vals(j);
            beta_thr = beta_seg;
            beta_thr(abs(beta_thr) < lam) = 0;

            % Compute residuals for all nodewise regressions on this segment
            % under the thresholded coefficients.
            residual = zeros(len, p);
            for node = 1:p
                idx_minus = [1:(node - 1), (node + 1):p];
                X_mj = data_y_seg(:, idx_minus);
                y = data_y_seg(:, node);
                beta_node = beta_thr(node, :).';
                residual(:, node) = y - X_mj * beta_node;
            end

            BIS_sum = 0;
            for node = 1:p
                res_trim = residual(row_start:row_end, node);
                phi_node = beta_thr(node, :);
                % Sum BIC across nodes, matching LinearDetect's loop.
                BIS_sum = BIS_sum + bic_scalar(res_trim, phi_node, false, 1);
            end
            BIC_vals(j) = BIS_sum;
        end

        [~, idx_min] = min(BIC_vals);
        lambda_best(seg) = lambda_vals(idx_min);
    end

    beta_final = beta_hat_list;
    for seg = 1:m_hat
        beta_final{seg}(abs(beta_final{seg}) < lambda_best(seg)) = 0;
    end
end

function hbic = hbic_optimal_ggm(obj, beta_final, cp_final)
    %% Compute HBIC for a thresholded segmented GGM fit.
    %
    % LinearDetect selects the block size by minimizing an HBIC score over
    % candidate block sizes. For GGM, HBIC is computed node-by-node using
    % the variance of residuals and a sparsity penalty.
    %
    % This implementation follows LinearDetect::BIC() with method="MLR"
    % applied to univariate residuals (p_y = 1), and then sums over nodes.
    % We use an equivalent scaling by multiplying by T.
    %
    % - Input:
    %   @beta_final: Thresholded segment-wise coefficient matrices.
    %   @cp_final:   Change points defining the segments.
    %
    % - Output:
    %   @hbic: Scalar HBIC score (smaller is better).

    TT = obj.T;
    p = obj.p;
    d = p - 1;
    gamma = obj.gamma_val;

    brk_full = [1; cp_final(:); TT + 1];
    m_hat = numel(brk_full) - 1;

    hbic_res = zeros(p, 1);
    for node = 1:p
        idx_minus = [1:(node - 1), (node + 1):p];
        X_all = obj.X(:, idx_minus);

        % Concatenate residuals and coefficients across segments. This makes
        % counting nonzeros and computing sigma_hat straightforward.
        residual_full = zeros(TT, 1);
        pos = 1;
        beta_concat = zeros(1, m_hat * d);

        for seg = 1:m_hat
            lb = brk_full(seg);
            ub = brk_full(seg + 1) - 1;
            y_seg = obj.X(lb:ub, node);
            X_seg = X_all(lb:ub, :);
            beta_seg = beta_final{seg}(node, :).';
            r = y_seg - X_seg * beta_seg;
            residual_full(pos:(pos + numel(r) - 1)) = r;
            pos = pos + numel(r);
            beta_concat(((seg - 1) * d + 1):(seg * d)) = beta_final{seg}(node, :);
        end

        sigma_hat = sum(residual_full .^ 2) / TT;
        log_det = log(sigma_hat);
        count = sum(beta_concat ~= 0);
        % Multiply the per-sample HBIC by T for a comparable scale.
        hbic_res(node) = log_det * TT + 2 * gamma * log(d) * count;
    end

    hbic = sum(hbic_res);
end

function val = bic_scalar(residual, phi_row, use_hbic, gamma_val)
    %% Compute scalar BIC/HBIC for univariate residuals (LinearDetect::BIC).
    %
    % LinearDetect::BIC() computes log(det(sigma_hat)) plus a complexity
    % term based on the number of nonzero coefficients. For a univariate
    % residual, det(sigma_hat) is simply the residual variance.
    %
    % - Input:
    %   @residual:  Residual vector for one node.
    %   @phi_row:   Coefficient vector used to count model size.
    %   @use_hbic:  If true, compute HBIC; otherwise compute BIC.
    %   @gamma_val: Gamma parameter for HBIC (ignored for BIC).
    %
    % - Output:
    %   @val: Scalar criterion value.

    residual = residual(:);
    T_new = numel(residual);
    count = sum(phi_row(:) ~= 0);
    sigma_hat = sum(residual .^ 2) / T_new;
    if sigma_hat <= 1e-8
        sigma_hat = sigma_hat + 2 * (abs(sigma_hat) + 1e-3);
    end
    log_det = log(sigma_hat);

    if use_hbic
        p_x = numel(phi_row);
        val = log_det + 2 * gamma_val * log(p_x) * count / T_new;
    else
        val = log_det + log(T_new) * count / T_new;
    end
end
