function V_Up(obj)
    %% Update V_k.

    % Compute eigendecompositions.
    [Vs_, Ds_] = pageeig(obj.Theta_k - obj.A_k * (1 / obj.beta_));
    % Project eigenvalues.
    Ds_(obj.idx) = max(Ds_(obj.idx), obj.epsilon);
    % Recover.
    V_k = pagemtimes(pagemtimes(Vs_, Ds_), permute(Vs_, [2, 1, 3]));
    % Symmetrization.
    obj.V_k = (V_k + permute(V_k, [2, 1, 3])) / 2;
end