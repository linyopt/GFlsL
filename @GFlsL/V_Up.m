function V_Up(obj)
    %% Update V_k.

    % Skip eigendecomposition for pages that are already feasible.
    S = obj.Theta_k - obj.A_k * (1 / obj.beta_v);
    S = (S + permute(S, [2, 1, 3])) / 2;

    need_proj = false(1, obj.T);
    eye_p = eye(obj.p);
    for tt = 1:obj.T
        [~, chol_flag] = chol(S(:, :, tt) - obj.epsilon * eye_p);
        need_proj(tt) = chol_flag ~= 0;
    end

    if ~any(need_proj)
        obj.V_k = S;
        return
    end

    [Vs_, Ds_] = pageeig(S(:, :, need_proj));
    num_proj = nnz(need_proj);
    diag_idx = (1:(obj.p + 1):obj.p^2)' + obj.p^2 .* (0:num_proj - 1);
    Ds_(diag_idx) = max(Ds_(diag_idx), obj.epsilon);
    V_proj = pagemtimes(pagemtimes(Vs_, Ds_), permute(Vs_, [2, 1, 3]));
    V_proj = (V_proj + permute(V_proj, [2, 1, 3])) / 2;

    V_k = S;
    V_k(:, :, need_proj) = V_proj;
    obj.V_k = V_k;
end
