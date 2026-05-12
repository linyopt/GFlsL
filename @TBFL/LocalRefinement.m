function LocalRefinement(obj)
    %% Step III: Exhaustive local search refinement (LinearDetect-compatible).
    %
    % This method mirrors LinearDetect:::ggm.second.step.search().
    %
    % Each Step II cluster is refined independently. For a cluster:
    % - If it contains multiple candidate points, search the interior
    %   (min+1):(max-1).
    % - If it contains a single point, search a window of width roughly
    %   +/- one block length around the candidate.
    %
    % For each refinement point `num`, compute a local SSE:
    % - Fit the left side using coefficients from the left midpoint block.
    % - Fit the right side using coefficients from the right midpoint block.
    % Coefficients are taken from Step I and kept fixed during the search.
    %
    % The R code uses an evaluation order heuristic and an early stopping
    % rule based on the 20% quantile of visited SSE values. We port these
    % details to reduce differences vs LinearDetect.

    if isempty(obj.cp_first)
        obj.breaks = [];
        obj.beta_hat_list = {};
        return;
    end

    if isempty(obj.beta_full) || isempty(obj.blocks)
        error("TBFL:LocalRefinement:missingState", ...
            "beta_full/blocks must be computed before LocalRefinement().");
    end

    data_y = obj.X;
    data_x = obj.X;
    TT = obj.T;
    p = obj.p;
    d = p - 1;
    blocks = obj.blocks;
    n_blocks = numel(blocks) - 1;
    blocks_size = diff(blocks);

    cp_first = obj.cp_first;
    cl_number = numel(cp_first);

    cp_list = cell(cl_number + 2, 1);
    cp_list{1} = 1;
    cp_list{cl_number + 2} = TT + 1;

    cp_index_list = cell(cl_number + 2, 1);
    cp_index_list{1} = 1;
    cp_index_list{cl_number + 2} = n_blocks + 1;

    for i = 1:cl_number
        cp_list{i + 1} = cp_first{i}(:).';
        cp_index_list{i + 1} = arrayfun(@(x) find(blocks == x, 1), cp_list{i + 1});
    end

    cp_search = zeros(cl_number, 1);
    cp_list_full = cp_list;
    beta_hat_list = cell(cl_number + 1, 1);

    for i = 1:cl_number
        % Use the midpoint block between adjacent clusters as the
        % representative coefficient matrix for the left segment.
        idx = floor((min(cp_index_list{i + 1}) + max(cp_index_list{i})) / 2);
        beta_hat_list{i} = obj.beta_full{idx};

        if numel(cp_list{i + 1}) > 1
            % LinearDetect searches the open interval (first+1):(last-1)
            % when a cluster contains multiple block boundaries.
            cp_list_full{i + 1} = (cp_list{i + 1}(1) + 1):(cp_list{i + 1}(end) - 1);
        else
            bidx = cp_index_list{i + 1}(1);
            span = blocks_size(bidx);
            cp_list_full{i + 1} = (cp_list{i + 1}(1) - span + 1):(cp_list{i + 1}(1) + span - 1);
        end

        cand = cp_list_full{i + 1}(:).';
        cand = cand(cand >= 2 & cand <= TT);
        if isempty(cand)
            cp_search(i) = cp_list{i + 1}(1);
            continue;
        end

        % Decide scan direction by comparing SSE at the two endpoints.
        [sse_left, sse_right] = endpoint_sse(i, cand(1), cp_list, cp_index_list, ...
            obj.beta_full, blocks, blocks_size, data_y, data_x);
        sse1 = sse_left + sse_right;

        [sse_left, sse_right] = endpoint_sse(i, cand(end), cp_list, cp_index_list, ...
            obj.beta_full, blocks, blocks_size, data_y, data_x);
        sse2 = sse_left + sse_right;

        if sse1 <= sse2
            cand_order = cand;
        else
            cand_order = fliplr(cand);
        end

        % Evaluate SSE along the chosen direction. Use the same early-stop
        % heuristic as LinearDetect (20% quantile after >=20 evaluations).
        sse_full = zeros(numel(cand_order), 1);
        n_eval = 0;
        for ii = 1:numel(cand_order)
            num = cand_order(ii);
            [temp1, temp2] = endpoint_sse(i, num, cp_list, cp_index_list, ...
                obj.beta_full, blocks, blocks_size, data_y, data_x);

            n_eval = n_eval + 1;
            sse_full(n_eval) = temp1 + temp2;

            if n_eval >= min(20, numel(cand_order))
                q20 = quantile(sse_full(1:n_eval), 0.20);
                if sse_full(n_eval) >= q20
                    break;
                end
            end
        end

        % LinearDetect breaks ties by the earliest point in scan order.
        [~, best_pos] = min(sse_full(1:n_eval));
        cp_search(i) = cand_order(best_pos);
    end

    idx = floor((min(cp_index_list{cl_number + 2}) + max(cp_index_list{cl_number + 1})) / 2);
    idx = min(max(idx, 1), n_blocks);
    beta_hat_list{cl_number + 1} = obj.beta_full{idx};

    obj.breaks = cp_search(:);
    obj.beta_hat_list = beta_hat_list;
end

function [temp1, temp2] = endpoint_sse(i, num, cp_list, cp_index_list, beta_est, ...
    blocks, blocks_size, data_y, data_x)
    %% Compute SSE for the two adjacent windows around a candidate split.
    %
    % This helper matches the repeated SSE computations in
    % LinearDetect:::ggm.second.step.search(). For a candidate split `num`:
    % - The left window spans [lb1, ub1] and is evaluated with beta1.
    % - The right window spans [lb2, ub2] and is evaluated with beta2.
    %
    % Coefficients beta1 and beta2 are taken from Step I (beta_est) at
    % midpoint blocks on the left and right sides.
    TT = size(data_y, 1);
    p = size(data_y, 2);
    d = p - 1;

    lb1 = min(cp_list{i + 1}) - blocks_size(cp_index_list{i + 1}(1));
    ub1 = num - 1;
    lb1 = max(lb1, 1);
    ub1 = min(ub1, TT);

    idx1 = floor((min(cp_index_list{i + 1}) + max(cp_index_list{i})) / 2);
    idx1 = min(max(idx1, 1), numel(beta_est));
    beta1 = beta_est{idx1};

    temp1 = segment_sse(lb1, ub1, beta1, data_y, data_x, d);

    lb2 = num;
    ub2 = max(cp_list{i + 1}) + blocks_size(cp_index_list{i + 1}(end)) - 1;
    lb2 = max(lb2, 1);
    ub2 = min(ub2, TT);

    idx2 = floor((min(cp_index_list{i + 2}) + max(cp_index_list{i + 1})) / 2);
    idx2 = min(max(idx2, 1), numel(beta_est));
    beta2 = beta_est{idx2};

    temp2 = segment_sse(lb2, ub2, beta2, data_y, data_x, d);
end

function sse = segment_sse(lb, ub, beta_mat, data_y, data_x, d)
    %% Compute SSE for the GGM nodewise regressions on an interval.
    %
    % For each node j:
    %   y_j(t) is predicted by X_{-j}(t) * beta_j,
    % where beta_j is taken from the j-th row of beta_mat.
    %
    % - Input:
    %   @lb:       Left index (inclusive).
    %   @ub:       Right index (inclusive).
    %   @beta_mat: p-by-(p-1) coefficient matrix.
    %   @data_y:   Response matrix (T-by-p).
    %   @data_x:   Predictor matrix (T-by-p). For GGM, data_x=data_y.
    %   @d:        p-1 (kept as an explicit parameter for clarity).
    %
    % - Output:
    %   @sse: Sum of squared residuals over nodes and time indices.
    if lb > ub
        sse = 0;
        return;
    end

    p = size(data_y, 2);
    len = ub - lb + 1;
    forecast = zeros(len, p);

    for j = 1:p
        idx_minus_j = [1:(j - 1), (j + 1):p];
        X_mj = data_x(lb:ub, idx_minus_j);
        beta_j = beta_mat(j, :).';
        forecast(:, j) = X_mj * beta_j;
    end

    residual = data_y(lb:ub, :) - forecast;
    sse = sum(residual .^ 2, "all");
end
