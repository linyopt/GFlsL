function theta_hat = EstimateTheta(obj, I)
    %% Estimate parameter on interval I using lasso.
    % Model-specific parameter estimation.
    %
    % - Input:
    %   @I:         Interval indices (vector).
    %
    % - Output:
    %   @theta_hat: Estimated parameter.

    % Extract data for interval.
    X_I = obj.X(I, :);
    n_I = length(I);

    switch obj.model
        case 'mean'
            % Lasso for mean with averaged loss:
            % (1/|I|) * sum ||X_i - mu||_2^2 + (lambda / sqrt(|I|)) * ||mu||_1
            % Closed-form solution: soft-threshold of sample mean.
            sample_mean = mean(X_I, 1)';
            lambda_I = obj.lambda / sqrt(n_I);
            theta_hat = DCDP.SoftThreshold(sample_mean, lambda_I / 2);

        case 'linear'
            % Lasso for regression with averaged loss:
            % (1/|I|) * sum (y_i - X_i^T beta)^2 + (lambda / sqrt(|I|)) * ||beta||_1
            % Use coordinate descent or built-in lasso.
            y_I = obj.y(I);
            lambda_I = obj.lambda / sqrt(n_I);
            lambda_cd = n_I * lambda_I;

            % Simple coordinate descent for lasso.
            beta = zeros(obj.p, 1);
            max_iter = 100;
            for iter = 1:max_iter
                for j = 1:obj.p
                    % Compute partial residual.
                    r_j = y_I - X_I * beta + X_I(:, j) * beta(j);
                    % Update beta(j).
                    rho_j = X_I(:, j)' * r_j;
                    z_j = X_I(:, j)' * X_I(:, j);
                    beta(j) = DCDP.SoftThreshold(rho_j / z_j, lambda_cd / (2 * z_j));
                end
            end
            theta_hat = beta;

        case 'ggm'
            % For GGM, estimate precision matrix.
            % Compute sample covariance.
            S_I = (X_I' * X_I) / n_I;
            % Symmetrize before inversion to avoid numerical errors.
            S_I = (S_I + S_I') / 2;
            % The minimizer of F(Omega, I) is the inverse sample covariance.
            % When S_I is nearly singular numerically, clamp only the tiny
            % eigenvalues rather than adding a large ad hoc ridge.
            [V, D] = eig(S_I);
            d = diag(D);
            d = max(d, 1e-7);
            d_inv = 1 ./ d;
            theta_hat = V * diag(d_inv) * V';

        otherwise
            error('Unknown model type: %s', obj.model);
    end
end
