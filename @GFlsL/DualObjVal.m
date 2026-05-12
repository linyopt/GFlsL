function DualObjVal(obj)
    %% Compute the dual objective value.

    % Three components:
    % 1. -Tp / 2 ||Wt||_F^2 with Wt = Theta_k - X_t X_t^T.
    % 2. -< W_t, X_t X_t^T >.
    % 3. epsilon * tr(delta_). Here we compute trace by summing its diagonal
    %    elements.
    W_t = (obj.Theta_k - obj.XXt) / obj.T;
    obj.dualval = -obj.T / 2 * sum(W_t .^ 2, 'all') ...
        - sum(W_t .* obj.XXt, "all") ...
        + obj.epsilon * sum(obj.delta_(obj.idx), 'all');
end
