function [phi_hat, beta_blocks, flag] = SolveFusedLassoNode(obj, j, blocks, lambda1, lambda2, cv_index, initial_phi)
    %% Solve TBFL Step I for one node (LinearDetect-compatible).
    %
    % This is a MATLAB port of LinearDetect's C++ routine
    % `lm_break_fit_block_new()` as used by `ggm_break_fit_block()`.
    %
    % Key difference vs a standard fused-lasso ADMM:
    % - The optimization is performed over increment parameters `phi`,
    %   with a two-stage soft-thresholding:
    %   (1) soft-threshold `phi` with `lambda1`
    %   (2) soft-threshold the cumulative sums with `lambda2`, then take
    %       differences to return to the increment parameterization.
    %
    % The output `phi_hat` corresponds to `theta` in the paper / R code:
    % each block i contributes a (p-1)-vector of increments.
    %
    % - Input:
    %   @j:           Node index (1..p).
    %   @blocks:      Block boundary vector (1-indexed, includes T+1).
    %   @lambda1:     Penalty for sparse increments.
    %   @lambda2:     Penalty for sparse cumulative coefficients.
    %   @cv_index:    Block indices used for CV (drop last obs in each).
    %   @initial_phi: Optional warm start, size 1-by-((p-1)*n_blocks).
    %
    % - Output:
    %   @phi_hat:     1-by-((p-1)*n_blocks) increment estimates.
    %   @beta_blocks: (p-1)-by-n_blocks coefficient matrix (cumulative sum).
    %   @flag:        0 if converged, 1 if not converged.

    arguments
        obj
        j (1, 1) double {mustBePositive, mustBeInteger}
        blocks (1, :) double {mustBeInteger, mustBePositive}
        lambda1 (1, 1) double {mustBeNonnegative}
        lambda2 (1, 1) double {mustBeNonnegative}
        cv_index (1, :) double {mustBeInteger, mustBeNonnegative} = []
        initial_phi double = []
    end

    T = obj.T;
    p = obj.p;
    d = p - 1;
    n_blocks = numel(blocks) - 1;

    if isempty(initial_phi)
        phi_hat = zeros(1, d * n_blocks);
    else
        phi_hat = reshape(initial_phi, 1, []);
        if numel(phi_hat) ~= d * n_blocks
            error("TBFL:SolveFusedLassoNode:badInitialPhi", ...
                "initial_phi must have %d elements.", d * n_blocks);
        end
    end

    idx_minus_j = [1:(j - 1), (j + 1):p];
    y_all = obj.X(:, j);

    % Predictors for node j are all other nodes (-j).
    % We keep a transposed view (d-by-T) to match the linear algebra in the
    % original C++ implementation.
    X_all = obj.X(:, idx_minus_j).';

    % Split the data into blocks. Block b contains indices
    % I_b = blocks(b):(blocks(b+1)-1).
    y_b = cell(n_blocks, 1);
    X_b = cell(n_blocks, 1);
    for b = 1:n_blocks
        I_b = blocks(b):(blocks(b + 1) - 1);
        y_b{b} = y_all(I_b, :);
        X_b{b} = X_all(:, I_b);
    end

    if ~isempty(cv_index)
        % LinearDetect performs CV by removing the last observation from
        % each CV block.
        for t = 1:numel(cv_index)
            b = cv_index(t);
            ybt = y_b{b};
            Xbt = X_b{b};
            if size(ybt, 1) <= 1
                error("TBFL:SolveFusedLassoNode:cvBlockTooSmall", ...
                    "CV block %d has <=1 observation after blocking.", b);
            end
            y_b{b} = ybt(1:end-1, :);
            X_b{b} = Xbt(:, 1:end-1);
        end
    end

    C = cell(n_blocks, 1);
    D = cell(n_blocks, 1);
    for b = 1:n_blocks
        % Cross-products for block b:
        % C_b = X_b * y_b, size d-by-1
        % D_b = X_b * X_b', size d-by-d
        C{b} = X_b{b} * y_b{b};
        D{b} = X_b{b} * X_b{b}.';
    end

    % Prefix sums allow fast suffix-sum construction.
    C_prefix = cell(n_blocks, 1);
    D_prefix = cell(n_blocks, 1);
    C_prefix{1} = C{1};
    D_prefix{1} = D{1};
    for b = 2:n_blocks
        C_prefix{b} = C_prefix{b - 1} + C{b};
        D_prefix{b} = D_prefix{b - 1} + D{b};
    end

    C_total = C_prefix{n_blocks};
    D_total = D_prefix{n_blocks};

    % Suffix sums:
    % For each i, C_suf{i} and D_suf{i} represent sums over blocks i..end.
    C_suf = cell(n_blocks, 1);
    D_suf = cell(n_blocks, 1);
    C_suf{1} = C_total;
    D_suf{1} = D_total;
    for b = 2:n_blocks
        C_suf{b} = C_total - C_prefix{b - 1};
        D_suf{b} = D_total - D_prefix{b - 1};
    end

    % Precompute a per-block linear operator for Step I.
    %
    % In exact arithmetic, inv(A) * S and A \ S are equivalent. In finite
    % precision, the choice can affect downstream thresholding and thus the
    % selected change points. We keep both modes:
    %   - obj.linear_solver == "inv": explicit inverse (LinearDetect-like)
    %   - obj.linear_solver == "solve": cached factorization + backslash
    use_inverse = (obj.linear_solver == "inv");
    D_suf_op = cell(n_blocks, 1);
    for b = 1:n_blocks
        Db = D_suf{b};
        % LinearDetect uses Armadillo's eig_sym() on D_sum_new to detect
        % non-positive definiteness. D_sum_new is theoretically symmetric,
        % but in floating-point arithmetic it may have tiny asymmetry.
        % Use a symmetric view for the eigen check, but keep the original
        % matrix for the inverse/solve to better match the C++ path.
        Db_eig = (Db + Db.') / 2;
        eigvals = eig(Db_eig);
        min_eig = min(real(eigvals));
        add_pd = 0;
        if min_eig <= 0
            % Ensure invertibility. In the R/C++ implementation this is
            % handled implicitly by numerical tolerances; here we add a
            % small diagonal shift when needed.
            if min_eig < 0
                % Match LinearDetect::lm_break_fit_block_new(): add
                % 10 * |min_eigen| when the matrix is not PD.
                add_pd = 10 * abs(min_eig);
            else
                add_pd = 1e-6;
            end
        end
        A = Db + add_pd * eye(d);
        if use_inverse
            D_suf_op{b} = inv(A);
        else
            % Prefer Cholesky for SPD matrices; fall back to LU if needed.
            try
                D_suf_op{b} = decomposition(A, "chol");
            catch
                D_suf_op{b} = decomposition(A, "lu");
            end
        end
    end

    phi_new = zeros(size(phi_hat));
    flag = 0;
    tol_curr = obj.tol;

    % Match LinearDetect's C++ loop counter exactly:
    %   int l = 2;
    %   while (l < max_iteration) {
    %     if (l == floor(0.5 * max_iteration)) tol *= 2;
    %     if (l == floor(0.75 * max_iteration)) tol *= 2;
    %     l = l + 1;
    %     ...
    %   }
    %
    % Using the same counter is important because LinearDetect increases the
    % tolerance partway through the iterations; an off-by-one shift can
    % change early stopping and the selected CV lambdas.
    l = 2;
    while l < obj.maxiter
        if l == floor(0.5 * obj.maxiter)
            tol_curr = 2 * tol_curr;
        end
        if l == floor(0.75 * obj.maxiter)
            tol_curr = 2 * tol_curr;
        end
        l = l + 1;

        phi_compare = phi_hat;

        for i = 1:n_blocks
            % Coordinate update for block i:
            % Compute S = C_suf{i} - sum_{jj!=i} D_suf{k(jj,i)} * phi_jj,
            % apply lambda1 soft-thresholding, then solve a d-by-d system.
            E_new = zeros(d, 1);
            for jj = 1:n_blocks
                k = max(jj, i);
                Dk = D_suf{k};
                phi_jj = phi_hat((jj - 1) * d + 1:jj * d);
                E_new = E_new + Dk * phi_jj.';
            end
            phi_ii = phi_hat((i - 1) * d + 1:i * d);
            E_new = E_new - D_suf{i} * phi_ii.';

            S = C_suf{i} - E_new;
            S = TBFL.SoftThreshold(S, lambda1);
            if use_inverse
                phi_temp = D_suf_op{i} * S;
            else
                phi_temp = D_suf_op{i} \ S;
            end
            phi_hat((i - 1) * d + 1:i * d) = phi_temp.';
            phi_new((i - 1) * d + 1:i * d) = phi_temp.';
        end

        % Second-stage soft-thresholding in the cumulative domain.
        % This matches the "lambda2" operation in LinearDetect's solver.
        phi_temp_soft = zeros(size(phi_hat));
        temp_1 = phi_hat(1:d);
        phi_temp_soft(1:d) = TBFL.SoftThreshold(temp_1, lambda2);
        for z = 2:n_blocks
            temp_2 = temp_1 + phi_hat((z - 1) * d + 1:z * d);
            temp_1_soft = TBFL.SoftThreshold(temp_1, lambda2);
            temp_2_soft = TBFL.SoftThreshold(temp_2, lambda2);
            phi_temp_soft((z - 1) * d + 1:z * d) = temp_2_soft - temp_1_soft;
            temp_1 = temp_2;
        end
        phi_new = phi_temp_soft;

        max_diff = max(abs(phi_new - phi_compare), [], "all");
        if max_diff < tol_curr
            break;
        end
        if max_diff > tol_curr
            phi_hat = phi_new;
        end

        if max_diff > 1e5
            flag = 1;
            break;
        end
    end


    phi_incr = reshape(phi_hat, d, n_blocks);
    beta_blocks = cumsum(phi_incr, 2);
end
