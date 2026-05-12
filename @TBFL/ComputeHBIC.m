function hbic = ComputeHBIC(obj, block_size)
    %% Compute HBIC for a given block size.
    %
    % HBIC (High-dimensional BIC) is used to select a block size when
    % optimal.block=TRUE in LinearDetect. This method provides an HBIC-like
    % score for a candidate `block_size`.
    %
    % Note:
    % The main TBFL pipeline in `@TBFL/run.m` uses a more direct
    % LinearDetect-aligned implementation (`hbic_optimal_ggm`). This method
    % is kept as a standalone utility.
    %
    % HBIC = sum_j sum_b [ n_b * log(RSS_jb / n_b) ]
    %        + K * log(T) * (s + K) * log(p)
    %
    % where:
    %   - RSS_jb is the residual sum of squares for node j in block b
    %   - K is the number of detected change points
    %   - s is the total number of non-zero coefficients
    %   - n_b is the block size
    %
    % - Input:
    %   @block_size: Block size to evaluate.
    %
    % - Output:
    %   @hbic: HBIC value.

    % Fit the Step I neighborhood regressions for the given block size.
    % This populates obj.Beta and obj.Delta.
    obj.SolveNeighborhoodSelection(block_size);

    num_blocks = ceil(obj.T / block_size);

    % Compute log-likelihood term.
    log_lik = 0;
    total_nonzero = 0;

    for j = 1:obj.p
        idx_minus_j = [1:(j-1), (j+1):obj.p];

        for b = 1:num_blocks
            % Block indices.
            start_idx = (b - 1) * block_size + 1;
            end_idx = min(b * block_size, obj.T);
            I_b = start_idx:end_idx;
            n_b = length(I_b);

            if n_b < 2
                continue;
            end

            % Extract data.
            y_b = obj.X(I_b, j);
            X_b = obj.X(I_b, idx_minus_j);

            % Get coefficients.
            if ~isempty(obj.Beta{j}) && size(obj.Beta{j}, 2) >= b
                beta_jb = obj.Beta{j}(:, b);
            else
                beta_jb = zeros(obj.p - 1, 1);
            end

            % Compute RSS.
            residuals = y_b - X_b * beta_jb;
            rss = sum(residuals.^2);

            % Avoid log(0).
            rss = max(rss, 1e-10);

            % Add to log-likelihood.
            log_lik = log_lik + n_b * log(rss / n_b);

            % Count non-zero coefficients.
            total_nonzero = total_nonzero + sum(abs(beta_jb) > 1e-8);
        end
    end

    % Count change points (non-zero differences).
    K = 0;
    for j = 1:obj.p
        if ~isempty(obj.Delta{j})
            for b = 1:size(obj.Delta{j}, 2)
                if norm(obj.Delta{j}(:, b), 2) > 1e-8
                    K = K + 1;
                    % Count each block boundary only once per node.
                    break;
                end
            end
        end
    end
    % K is the number of block boundaries with at least one node changing.
    % Recount properly.
    K = 0;
    if num_blocks > 1
        for b = 1:(num_blocks - 1)
            has_change = false;
            for j = 1:obj.p
                if ~isempty(obj.Delta{j}) && size(obj.Delta{j}, 2) >= b
                    if norm(obj.Delta{j}(:, b), 2) > 1e-8
                        has_change = true;
                        break;
                    end
                end
            end
            if has_change
                K = K + 1;
            end
        end
    end

    % HBIC penalty term.
    % Following the paper's HBIC construction.
    % Penalty = (K + 1) * log(T) * (s / num_blocks + K) * log(p)
    s_avg = total_nonzero / num_blocks;
    penalty = (K + 1) * log(obj.T) * (s_avg + K) * log(obj.p);

    hbic = log_lik + penalty;
end
