function dfeas = DualFeas(obj)
    %% Compute dual infeasibility.

    % Positive definite pages have zero cone infeasibility. Screen them with
    % Cholesky and compute eigenvalues only for pages that fail.
    need_eig = false(1, obj.T);
    for tt = 1:obj.T
        [~, chol_flag] = chol(obj.delta_(:, :, tt));
        need_eig(tt) = chol_flag ~= 0;
    end

    min_eigs = zeros(1, obj.T);
    if any(need_eig)
        eigs_delta_ = reshape(pageeig(obj.delta_(:, :, need_eig)), ...
            obj.p, nnz(need_eig));
        min_eigs(need_eig) = abs(min(min(eigs_delta_, [], 1), 0));
    end
    % Compute the norm of delta_.
    norms_delta_ = vecnorm(reshape(obj.delta_, [], obj.T));
    % Compute dfeas1.
    dfeas1 = max(min_eigs ./ (norms_delta_ + 1));

    % Compute the values of Frobenius norms of Z_k's.
    norms_Z_k = vecnorm(reshape(obj.Z_k(:, :, 2:obj.T), [], obj.T-1));
    % Compute dfeas2.
    dfeas2 = max(max(norms_Z_k - obj.lamb2, 0)) / (1 + max(norms_Z_k));

    % Compute absolute values of Y_k.
    abs_Y_k = abs(obj.Y_k);
    % Compute dfeas3.
    dfeas3 = max(max(abs_Y_k - obj.lamb1, 0), [], 'all') / (1 + max(abs_Y_k, [], 'all'));

    % Compute dfeas.
    dfeas = max([dfeas1, dfeas2, dfeas3]);
end