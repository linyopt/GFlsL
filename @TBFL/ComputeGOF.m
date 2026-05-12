function gof = ComputeGOF(obj, j, I, beta_j)
    %% Compute goodness-of-fit (RSS) for a segment.
    %
    % For GGM TBFL, each node j is regressed on the remaining nodes -j.
    % This helper computes the residual sum of squares on an interval and
    % is used by HBIC / BIC-style criteria to compare configurations.
    %
    % If beta_j is empty, fits a new regression on the interval.
    % Otherwise, uses the provided coefficients.
    %
    % - Input:
    %   @j:      Node index (1 to p).
    %   @I:      Interval indices (vector).
    %   @beta_j: Coefficient vector (p-1 x 1) or empty.
    %
    % - Output:
    %   @gof: Residual sum of squares.

    if length(I) < 2
        % A single observation carries no meaningful regression fit.
        gof = 0;
        return;
    end

    % Extract response and predictors for node j.
    idx_minus_j = [1:(j-1), (j+1):obj.p];
    y = obj.X(I, j);
    X_mj = obj.X(I, idx_minus_j);

    if isempty(beta_j)
        % Fit an OLS regression for stability checks in the information
        % criteria. This is not the penalized Step I solver.
        XtX = X_mj' * X_mj;
        Xty = X_mj' * y;

        % Use a small ridge correction if XtX is ill-conditioned.
        if rcond(XtX) < 1e-10
            lambda_ridge = 1e-4 * trace(XtX) / size(XtX, 1);
            beta_j = (XtX + lambda_ridge * eye(size(XtX, 1))) \ Xty;
        else
            beta_j = XtX \ Xty;
        end
    end

    % Compute residuals.
    residuals = y - X_mj * beta_j;
    gof = sum(residuals.^2);
end
