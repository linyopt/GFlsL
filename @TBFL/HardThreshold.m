function HardThreshold(obj)
    %% Step II: Jump selection + clustering (LinearDetect-compatible).
    %
    % This method implements the Step II logic used by
    % LinearDetect:::ggm.first.step.blocks() for method="GGM".
    %
    % High-level idea:
    % 1) Compute jump sizes at each block boundary from Step I increments.
    % 2) Iteratively select "large" jumps using a 2-cluster separation.
    % 3) After each selection, recompute a BIC/HBIC score on the implied
    %    segmented model and stop when the score stops improving.
    % 4) Cluster the selected change points into groups to be refined in
    %    Step III (local exhaustive search).
    %
    % Input prerequisites (must be set before calling):
    % - obj.blocks:       block boundaries
    % - obj.phi_hat_full: Step I increment estimates
    %                    (p-by-((p-1)*n_blocks))
    % - obj.beta_full:    Step I cumulative coefficients per block
    %
    % Output:
    % - obj.jumps_l2: squared L2 jump sizes per block (length n_blocks)
    % - obj.candidates: selected candidate block boundaries (time indices)
    % - obj.cp_first: clustered candidates (cell array, R: pts.list)

    if isempty(obj.blocks) || isempty(obj.phi_hat_full)
        error("TBFL:HardThreshold:missingState", ...
            "blocks/phi_hat_full must be computed before HardThreshold().");
    end

    blocks = obj.blocks;
    TT = obj.T;
    p = obj.p;
    d = p - 1;
    n_blocks = numel(blocks) - 1;

    if n_blocks <= 1
        obj.jumps_l2 = zeros(n_blocks, 1);
        obj.candidates = [];
        obj.cp_first = {};
        return;
    end

    phi_hat_full = obj.phi_hat_full;
    if ~isequal(size(phi_hat_full), [p, d * n_blocks])
        error("TBFL:HardThreshold:badPhiHatFullSize", ...
            "phi_hat_full must be size %d-by-%d.", p, d * n_blocks);
    end

    jumps_l2 = zeros(n_blocks, 1);
    jumps_l1 = zeros(n_blocks, 1);
    for i = 2:n_blocks
        blk = phi_hat_full(:, (i - 1) * d + 1:i * d);
        jumps_l2(i) = sum(blk .^ 2, "all");
        jumps_l1(i) = sum(abs(blk), "all");
    end
    obj.jumps_l2 = jumps_l2;

    % LinearDetect uses the L2 jump sizes for jump selection.
    jumps = jumps_l2;

    blocks_size = diff(blocks);
    % LinearDetect ignores potential jumps too close to the boundaries.
    % For GGM, the rule is:
    %   if (TT < 250) ignore_num <- min(3, round(100 / mean(blocks.size)))
    %   else         ignore_num <- max(6, round(250 / min(blocks.size)))
    if TT < 250
        ignore_num = min(3, round(100 / mean(blocks_size)));
    else
        ignore_num = max(6, round(250 / min(blocks_size)));
    end
    ignore_num = max(0, min(ignore_num, numel(jumps)));

    if ignore_num > 0
        % Avoid selecting jumps too close to the boundaries.
        % This matches the "ignore_num" guard used in LinearDetect.
        jumps(1:ignore_num) = 0;
        jumps((end - (ignore_num - 1)):end) = 0;
    end

    % Greedy BIC/HBIC descent:
    % Each iteration adds the block boundary from the current "large jump"
    % cluster and recomputes the information criterion. Stop when the score
    % no longer decreases.
    BIC_diff = 1;
    BIC_old = 1e8;
    pts_sel = [];
    loc_block_full = [];
    jumps_work = jumps;

    while (BIC_diff > 0) && (numel(unique(jumps_work)) > 1)
        pts_sel_old = pts_sel;

        loc_block = [];
        u = unique(jumps_work);
        if numel(u) > 2
            % LinearDetect uses kmeans(jumps, centers=2) and checks a fit
            % statistic betweenss/totss against 0.2. In 1D, the optimal
            % 2-means partition can be found deterministically by an
            % exhaustive split search, which avoids RNG dependence.
            [clus, centers, fit2] = two_means_1d(jumps_work);
            if fit2 < 0.20
                break;
            end
            if centers(1) > centers(2)
                loc_block = find(clus == 1);
            else
                loc_block = find(clus == 2);
            end
            pts_new = blocks(loc_block(:));
            pts_sel = unique([pts_sel(:); pts_new(:)], "stable");
            loc_block_full = unique([loc_block_full(:); loc_block(:)], "stable");
        elseif numel(u) == 2
            % When there are only two distinct jump sizes remaining, the R
            % code selects the maximum directly.
            [~, loc_block] = max(jumps_work);
            pts_new = blocks(loc_block);
            pts_sel = unique([pts_sel(:); pts_new(:)], "stable");
            loc_block_full = unique([loc_block_full(:); loc_block], "stable");
        end

        % Build a "restricted" increment matrix by keeping only the selected
        % block boundaries. This is the quantity passed into the BIC/HBIC
        % evaluation in LinearDetect.
        phi_hat_full_new = phi_hat_full;
        for i = 2:n_blocks
            if ~ismember(i, loc_block_full)
                phi_hat_full_new(:, (i - 1) * d + 1:i * d) = 0;
            end
        end

        if isempty(obj.beta_full)
            beta_full_all = cell(n_blocks, 1);
            beta_full_all{1} = phi_hat_full(:, 1:d);
            for i = 2:n_blocks
                beta_full_all{i} = beta_full_all{i - 1} + phi_hat_full(:, (i - 1) * d + 1:i * d);
            end
        else
            beta_full_all = obj.beta_full;
        end

        seg_blocks = [1; sort(loc_block_full(:)); n_blocks + 1];
        beta_hat_list_new = cell(numel(seg_blocks) - 1, 1);
        for s = 1:(numel(seg_blocks) - 1)
            % Use the midpoint block as a representative coefficient matrix
            % for the segment, matching the R implementation.
            idx = floor((seg_blocks(s) + seg_blocks(s + 1)) / 2);
            beta_hat_list_new{s} = beta_full_all{idx};
        end

        % Compute fitted values and residuals for each node over time.
        % This aligns with the R implementation, which reconstructs a
        % piecewise-constant coefficient path and evaluates BIC/HBIC.
        forecast = zeros(p, TT);
        for s = 1:numel(beta_hat_list_new)
            lb = blocks(seg_blocks(s));
            ub = blocks(seg_blocks(s + 1)) - 1;
            for j = 1:p
                idx_minus_j = [1:(j - 1), (j + 1):p];
                X_mj = obj.X(:, idx_minus_j);
                beta_j = beta_hat_list_new{s}(j, :).';
                forecast(j, lb:ub) = (X_mj(lb:ub, :) * beta_j).';
            end
        end
        residual = obj.X.' - forecast;

        BIC_new = 0;
        for j = 1:p
            phi_row = phi_hat_full_new(j, :);
            res_row = residual(j, :);
            if obj.HBIC_step2
                % HBIC uses gamma.val and log(p_x * p_y); for GGM p_y=1.
                BIC_new = BIC_new + bic_scalar(res_row, phi_row, true, obj.gamma_step2);
            else
                BIC_new = BIC_new + bic_scalar(res_row, phi_row, false, obj.gamma_step2);
            end
        end

        BIC_diff = BIC_old - BIC_new;
        BIC_old = BIC_new;

        if BIC_diff <= 0
            % Revert the last addition if the criterion does not improve.
            pts_sel = sort(pts_sel_old);
            break;
        end

        if ~isempty(loc_block)
            % Do not re-select the same boundary in subsequent iterations.
            jumps_work(loc_block) = 0;
        else
            break;
        end
    end

    cp_final = sort(pts_sel(:));
    if ~isempty(cp_final)
        % Drop change points too close to the boundaries. LinearDetect uses
        % a data-dependent guard based on the first/last block sizes.
        left_guard = sum(blocks_size(1:min(3, numel(blocks_size))));
        right_guard = TT - sum(blocks_size(max(1, numel(blocks_size) - 2):numel(blocks_size)));
        cp_final = cp_final(cp_final > left_guard);
        cp_final = cp_final(cp_final < right_guard);
        cp_final = sort(cp_final);
    end

    obj.candidates = cp_final;
    % Cluster candidates into groups for Step III local refinement.
    obj.cp_first = cluster_cp(cp_final, blocks, blocks_size, TT);
end

function [cluster, centers, fit2] = two_means_1d(x)
    %% Deterministic 2-means for a 1D vector.
    %
    % LinearDetect uses kmeans(jumps, centers=2) as a heuristic to separate
    % "large" vs "small" jumps. kmeans is randomized by default. For a 1D
    % vector, the optimal 2-means split can be found deterministically by
    % trying all possible cut points after sorting.
    %
    % - Input:
    %   @x: Vector of nonnegative jump sizes.
    %
    % - Output:
    %   @cluster: Integer labels in {1,2}, aligned to the original order.
    %   @centers: Two cluster means, ordered as [mean(cluster==1); mean(2)].
    %   @fit2:    BetweenSS / TotalSS, matching the R code's criterion.

    x = x(:);
    n = numel(x);
    mu = mean(x);
    totss = sum((x - mu) .^ 2);
    if totss <= 0
        cluster = ones(n, 1);
        centers = [mu; mu];
        fit2 = 0;
        return;
    end

    [xs, order] = sort(x);
    s1 = cumsum(xs);
    s2 = cumsum(xs .^ 2);
    idx = (1:n).';

    best_within = inf;
    best_split = 1;
    for k = 1:(n - 1)
        n1 = k;
        n2 = n - k;
        m1 = s1(k) / n1;
        m2 = (s1(end) - s1(k)) / n2;
        ss1 = s2(k) - n1 * m1^2;
        ss2 = (s2(end) - s2(k)) - n2 * m2^2;
        within = ss1 + ss2;
        if within < best_within
            best_within = within;
            best_split = k;
        end
    end

    cluster_sorted = ones(n, 1);
    cluster_sorted((best_split + 1):end) = 2;
    cluster = zeros(n, 1);
    cluster(order) = cluster_sorted;

    centers = [mean(x(cluster == 1)); mean(x(cluster == 2))];
    fit2 = (totss - best_within) / totss;
end

function val = bic_scalar(residual_row, phi_row, use_hbic, gamma_val)
    %% Compute the per-node BIC / HBIC score used in Step II.
    %
    % This mirrors LinearDetect::BIC() for the univariate residual case
    % (p_y = 1). The model size is the number of nonzeros in phi_row.
    %
    % - Input:
    %   @residual_row: Residuals for one node (length T).
    %   @phi_row:      Step I increment row for one node (length (p-1)*B).
    %   @use_hbic:     If true, compute HBIC; otherwise compute BIC.
    %   @gamma_val:    HBIC gamma parameter (ignored when use_hbic=false).
    %
    % - Output:
    %   @val: Scalar BIC / HBIC value.

    residual_row = residual_row(:);
    T_new = numel(residual_row);

    count = sum(phi_row(:) ~= 0);
    sigma_hat = sum(residual_row .^ 2) / T_new;
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

function pts_list = cluster_cp(cp_final, blocks, blocks_size, TT)
    %% Cluster candidate change points (LinearDetect-compatible).
    %
    % Step III refines each cluster separately. LinearDetect chooses the
    % number of clusters using either a simple gap heuristic (small n) or
    % a Gap statistic heuristic (large n), then splits by the largest gaps.
    %
    % - Input:
    %   @cp_final:    Sorted candidate change points (time indices).
    %   @blocks:      Block boundaries (1-indexed, includes T+1).
    %   @blocks_size: Vector of block lengths.
    %   @TT:          Sample size (T).
    %
    % - Output:
    %   @pts_list: Cell array; each cell contains one cluster of cps.

    if isempty(cp_final)
        pts_list = {};
        return;
    end
    cp_final = sort(cp_final(:));

    if numel(cp_final) == 1
        pts_list = {cp_final};
        return;
    end

    if numel(cp_final) <= 5
        thr = max(3 * mean(blocks_size));
        loc = ones(numel(cp_final), 1);
        cl_number = numel(cp_final);
        for i = 2:numel(cp_final)
            if (cp_final(i) - cp_final(i - 1)) <= thr
                cl_number = cl_number - 1;
                loc(i) = loc(i - 1);
            else
                loc(i) = i;
            end
        end
        loc_u = unique(loc, "stable");
        pts_list = cell(numel(loc_u), 1);
        for k = 1:numel(loc_u)
            pts_list{k} = cp_final(loc == loc_u(k));
        end
        return;
    end

    gap_temp = diff(cp_final);
    med_bs = median(blocks_size);
    sqrt_T = sqrt(TT);

    if med_bs <= sqrt_T / 4
        kappa1 = 9;
        kappa2 = 7;
    elseif med_bs <= sqrt_T / 2
        kappa1 = 7;
        kappa2 = 5;
    else
        kappa1 = 5;
        kappa2 = 3;
    end

    n_cp = numel(cp_final);

    if numel(unique(gap_temp)) > 1
        kmax = min(50, n_cp - 1);
        gap_stat = gap_statistic_1d(cp_final, kmax, 100);
        [~, gap_order] = sort(gap_stat, "descend");

        found = false;
        cl_number = 1;

        for idx = 1:numel(gap_order)
            k = gap_order(idx);
            if k < 1
                continue;
            end

            if k > 1
                [~, ord_gap] = sort(gap_temp, "descend");
                split_pos = sort(ord_gap(1:(k - 1)));
                cluster_pos = [0; split_pos(:); n_cp];
            else
                cluster_pos = [0; n_cp];
            end

            wide = false;
            for c = 1:k
                pts_i = cp_final((cluster_pos(c) + 1):cluster_pos(c + 1));
                [tf, idx_b] = ismember(pts_i, blocks);
                idx_b = idx_b(tf);
                idx_b = idx_b(idx_b >= 1 & idx_b <= numel(blocks_size));
                if isempty(idx_b)
                    block_avg = mean(blocks_size);
                else
                    block_avg = mean(blocks_size(idx_b));
                end

                if max(pts_i) - min(pts_i) > kappa1 * block_avg
                    wide = true;
                    break;
                end
            end

            if ~wide
                cl_number = k;
                found = true;
                break;
            end
        end

        if ~found
            cl_number = numel(gap_temp) + 1;
        end

        if cl_number > 1
            gap_sorted = sort(gap_temp, "descend");
            top_gaps = gap_sorted(1:min(cl_number - 1, numel(gap_sorted)));
            cl_number = sum(top_gaps > kappa2 * med_bs) + 1;
        end
    elseif numel(unique(gap_temp)) == 1 && gap_temp(1) == med_bs
        cl_number = 1;
    else
        cl_number = numel(gap_temp) + 1;
    end

    if cl_number <= 1
        pts_list = {cp_final};
        return;
    end

    [~, ord_gap] = sort(gap_temp, "descend");
    split_pos = sort(ord_gap(1:(cl_number - 1)));
    split_pos = [0; split_pos(:); n_cp];
    pts_list = cell(cl_number, 1);
    for k = 1:cl_number
        pts_list{k} = cp_final((split_pos(k) + 1):split_pos(k + 1));
    end
end

function gap = gap_statistic_1d(x, kmax, nboot)
    %% Compute a 1D Gap statistic (cluster::clusGap-style).
    %
    % LinearDetect calls factoextra::fviz_nbclust(..., method="gap_stat"),
    % which internally uses the Gap statistic with bootstrap reference
    % samples. To avoid heavy dependencies, we implement the same idea for
    % 1D vectors.
    %
    % We use the clusGap within-cluster dispersion definition:
    %   W_k = 0.5 * sum_r ( sum(dist(cluster_r)) / n_r )
    % and then:
    %   Gap(k) = E[log(W_k^*)] - log(W_k)
    % where W_k^* is computed on uniform reference samples.
    %
    % - Input:
    %   @x:     Data vector (change point locations).
    %   @kmax:  Maximum number of clusters to evaluate.
    %   @nboot: Number of bootstrap reference samples.
    %
    % - Output:
    %   @gap: Vector of Gap(k) for k=1..kmax.

    x = x(:);
    n = numel(x);
    if kmax < 1
        gap = zeros(0, 1);
        return;
    end
    if n <= 1
        gap = zeros(kmax, 1);
        return;
    end

    rng_state = rng;
    % Use a fixed seed to match the effective behavior of the R pipeline
    % (and restore the caller RNG state at the end).
    rng(1234, "twister");

    xmin = min(x);
    xmax = max(x);
    if xmax <= xmin
        gap = zeros(kmax, 1);
        rng(rng_state);
        return;
    end

    logW = log_within_dispersion_1d(x, kmax);

    logW_ref = zeros(nboot, kmax);
    for b = 1:nboot
        z = xmin + (xmax - xmin) * rand(n, 1);
        logW_ref(b, :) = log_within_dispersion_1d(z, kmax).';
    end

    gap = mean(logW_ref, 1).' - logW;

    rng(rng_state);
end

function logW = log_within_dispersion_1d(x, kmax)
    %% Compute log within-cluster dispersion for 1D k=1..kmax.
    %
    % This helper computes log(W_k) for the Gap statistic implementation.
    % For a 1D sorted vector, we can find the optimal k-means partitions
    % with dynamic programming and then compute the clusGap dispersion.

    x = sort(x(:));
    n = numel(x);
    kmax = min(kmax, n);

    idx = (1:n).';
    S1 = [0; cumsum(x)];
    S2 = [0; cumsum(x .^ 2)];
    Stx = [0; cumsum(idx .* x)];

    sse_cost = zeros(n, n);
    pair_cost = zeros(n, n);
    for i = 1:n
        for j = i:n
            m = j - i + 1;
            % Prefix sums are 0-indexed:
            % S1(k+1) = sum(x(1:k)), so sum(x(i:j)) = S1(j+1) - S1(i).
            sum_x = S1(j + 1) - S1(i);
            sum_x2 = S2(j + 1) - S2(i);
            sse_cost(i, j) = sum_x2 - (sum_x ^ 2) / m;

            sum_tx = Stx(j + 1) - Stx(i);
            sum_lx = sum_tx - (i - 1) * sum_x;
            pair_sum = 2 * sum_lx - (m + 1) * sum_x;
            pair_cost(i, j) = 0.5 * pair_sum / m;
        end
    end

    dp = inf(kmax, n);
    prev = zeros(kmax, n);
    for j = 1:n
        dp(1, j) = sse_cost(1, j);
        prev(1, j) = 0;
    end
    for k = 2:kmax
        for j = k:n
            best = inf;
            best_t = k - 1;
            for t = (k - 1):(j - 1)
                val = dp(k - 1, t) + sse_cost(t + 1, j);
                if val < best
                    best = val;
                    best_t = t;
                end
            end
            dp(k, j) = best;
            prev(k, j) = best_t;
        end
    end

    logW = zeros(kmax, 1);
    for k = 1:kmax
        j = n;
        c = k;
        W = 0;
        while c >= 1
            t = prev(c, j);
            i = t + 1;
            W = W + pair_cost(i, j);
            j = t;
            c = c - 1;
        end
        if W <= 0
            W = eps;
        end
        logW(k) = log(W);
    end
end
