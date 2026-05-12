function [F_val, theta_hat] = ComputeF(obj, I)
    %% Compute F(theta_hat, I) for an interval I.
    % Model-specific goodness-of-fit function.
    %
    % - Input:
    %   @I:         Interval indices (vector).
    %
    % - Output:
    %   @F_val:     Goodness-of-fit value.
    %   @theta_hat: Estimated parameter on interval I.

    % Check interval length.
    if length(I) < obj.min_interval_length
        F_val = 0;
        % Return appropriate shape based on model type.
        switch obj.model
            case {'mean', 'linear'}
                theta_hat = zeros(obj.p, 1);
            case 'ggm'
                theta_hat = zeros(obj.p, obj.p);
            otherwise
                error('Unknown model type: %s', obj.model);
        end
        return;
    end

    % Estimate parameter on interval.
    theta_hat = obj.EstimateTheta(I);

    % Compute F based on model type.
    switch obj.model
        case 'mean'
            % F = sum_{i in I} ||X_i - mu_hat||_2^2
            X_I = obj.X(I, :);
            residuals = X_I - theta_hat';
            F_val = sum(residuals(:).^2);

        case 'linear'
            % F = sum_{i in I} (y_i - X_i^T beta_hat)^2
            X_I = obj.X(I, :);
            y_I = obj.y(I);
            predictions = X_I * theta_hat;
            residuals = y_I - predictions;
            F_val = sum(residuals.^2);

        case 'ggm'
            % F = sum_{i in I} Tr(Omega_hat^T X_i X_i^T) - |I| * log|Omega_hat|
            Omega_hat = theta_hat;

            X_I = obj.X(I, :);
            % Sum_i Tr(Omega * X_i X_i^T) = Tr(Omega * (X_I^T X_I)).
            F_val = trace(Omega_hat * (X_I' * X_I));

            % Compute log determinant using eigenvalues.
            % log(det(A)) = sum(log(eigenvalues(A)))
            eigvals = eig(Omega_hat);
            logdet_Omega = sum(log(max(eigvals, 1e-10)));

            F_val = F_val - length(I) * logdet_Omega;

        otherwise
            error('Unknown model type: %s', obj.model);
    end
end
