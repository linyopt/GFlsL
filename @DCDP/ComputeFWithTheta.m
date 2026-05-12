function F_val = ComputeFWithTheta(obj, I, theta)
    %% Compute F(theta, I) with fixed theta.
    % Used in PLR final refinement step.
    %
    % - Input:
    %   @I:         Interval indices (vector).
    %   @theta:     Fixed parameter.
    %
    % - Output:
    %   @F_val:     Goodness-of-fit value.

    % Check interval length.
    if length(I) < obj.min_interval_length
        F_val = 0;
        return;
    end

    % Compute F based on model type.
    switch obj.model
        case 'mean'
            % F = sum_{i in I} ||X_i - mu||_2^2
            X_I = obj.X(I, :);
            residuals = X_I - theta';
            F_val = sum(residuals(:).^2);

        case 'linear'
            % F = sum_{i in I} (y_i - X_i^T beta)^2
            X_I = obj.X(I, :);
            y_I = obj.y(I);
            predictions = X_I * theta;
            residuals = y_I - predictions;
            F_val = sum(residuals.^2);

        case 'ggm'
            % F = sum_{i in I} Tr(Omega^T X_i X_i^T) - |I| * log|Omega|
            Omega = theta;

            X_I = obj.X(I, :);
            % Sum_i Tr(Omega * X_i X_i^T) = Tr(Omega * (X_I^T X_I)).
            F_val = trace(Omega * (X_I' * X_I));

            % Compute log determinant using eigenvalues for numerical stability.
            eigvals = eig(Omega);
            logdet_Omega = sum(log(max(eigvals, 1e-10)));

            F_val = F_val - length(I) * logdet_Omega;

        otherwise
            error('Unknown model type: %s', obj.model);
    end
end
