function R_val = ComputeR(obj, theta1, theta2, eta, s_k, e_k)
    %% Compute penalty R for PLR step.
    % Group lasso penalty for sparse mean/linear models.
    %
    % - Input:
    %   @theta1:    Parameter for first segment.
    %   @theta2:    Parameter for second segment.
    %   @eta:       Change point location.
    %   @s_k:       Start of local interval.
    %   @e_k:       End of local interval.
    %
    % - Output:
    %   @R_val:     Penalty value.

    switch obj.model
        case {'mean', 'linear'}
            % Group lasso: R = sum_{j=1..p} sqrt((eta - s) * theta1_j^2 + (e - eta) * theta2_j^2)
            weights1 = eta - s_k;
            weights2 = e_k - eta;
            R_val = sum(sqrt(weights1 * (theta1(:).^2) + weights2 * (theta2(:).^2)));

        case 'ggm'
            % The paper specifies PLR penalty for sparse mean/linear models;
            % for GGM it is left model-specific. Use no penalty by default.
            R_val = 0;

        otherwise
            error('Unknown model type: %s', obj.model);
    end
end
