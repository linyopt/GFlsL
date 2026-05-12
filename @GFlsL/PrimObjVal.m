function PrimObjVal(obj)
    %% Compute the primal objective value.

    % Compute norm of first-order difference of Theta.
    obj.norm_diff = vecnorm(reshape(diff(obj.Theta_k, 1, 3), [], obj.T-1));

    % Compute lossval.
    % Here we compute the trace by summing the diagonal elements.
    obj.lossval = sum((obj.Theta_k - obj.XXt) .^ 2, 'all') / 2 / obj.T;

    % Compute primal value.
    % Three components:
    % 1. lossval (squared Frobenius loss).
    % 2. Off-diagonal l1 penalty.
    % 3. Group fused lasso penalty.
    obj.primval = obj.lossval + (sum(obj.lamb1 .* abs(obj.Theta_k), 'all') ...
            - sum(obj.lamb1(obj.idx) .* abs(obj.Theta_k(obj.idx)), 'all')) ...
            + sum(obj.lamb2 .* obj.norm_diff);
end
