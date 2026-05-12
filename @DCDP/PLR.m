function PLR(obj)
    %% Penalized Local Refinement (Algorithm 3).
    % Refines each preliminary change point using local penalized optimization.
    %
    % - Usage:
    %   obj.PLR();
    %
    % - Output:
    %   obj.final_cps: refined change points.

    % Number of preliminary change points.
    K_hat = length(obj.prelim_cps);

    % If no change points, return empty.
    if K_hat == 0
        obj.final_cps = [];
        return;
    end

    % Set boundary points.
    eta_hat = [0; obj.prelim_cps; obj.n];

    % Initialize refined change points.
    obj.final_cps = zeros(K_hat, 1);

    % Refine each change point.
    % Note that eta_hat is indeed of length K_hat + 2.
    for k = 1:K_hat
        % Define local search interval.
        s_k = floor((2/3) * eta_hat(k) + (1/3) * eta_hat(k + 1));
        e_k = ceil((1/3) * eta_hat(k + 1) + (2/3) * eta_hat(k + 2));

        % Ensure valid interval.
        s_k = max(1, s_k);
        e_k = min(obj.n, e_k);

        % Skip degenerate local windows.
        if e_k - s_k < 2
            obj.final_cps(k) = eta_hat(k + 1);
            continue;
        end

        % Apply buffer_refine: search range is [s_k + buffer_refine, e_k - buffer_refine]
        search_start = s_k + obj.buffer_refine;
        search_end = e_k - obj.buffer_refine;

        if search_start >= search_end
            % Buffer too large for this window, use preliminary estimate.
            obj.final_cps(k) = eta_hat(k + 1);
            continue;
        end

        eta_candidates = search_start:obj.step_refine:search_end;

        % Step 1: Penalized local 2-segment fit.
        % Search over possible change point locations with step_refine.
        best_cost = inf;
        best_eta = eta_hat(k + 1);
        best_theta1 = [];
        best_theta2 = [];

        for eta = eta_candidates
            % Estimate parameters for two segments.
            I1 = s_k:(eta - 1);
            I2 = eta:e_k;

            if isempty(I1) || isempty(I2)
                continue;
            end

            [F1, theta1] = obj.ComputeF(I1);
            [F2, theta2] = obj.ComputeF(I2);

            % Compute penalty.
            R_val = obj.ComputeR(theta1, theta2, eta, s_k, e_k);

            % Total cost.
            cost = F1 + F2 + obj.zeta * R_val;

            if cost < best_cost
                best_cost = cost;
                best_eta = eta;
                best_theta1 = theta1;
                best_theta2 = theta2;
            end
        end

        if isempty(best_theta1) || isempty(best_theta2)
            % Penalized local search found no valid split; refinement is
            % skipped.
            obj.final_cps(k) = eta_hat(k + 1);
            continue;
        end

        % Step 2: Final refinement (unpenalized in eta).
        % Use the estimated theta1 and theta2, search for best eta.
        best_cost_final = inf;
        best_eta_final = best_eta;

        for eta = eta_candidates
            I1 = s_k:(eta - 1);
            I2 = eta:e_k;

            if isempty(I1) || isempty(I2)
                continue;
            end

            % Compute F with fixed theta.
            F1 = obj.ComputeFWithTheta(I1, best_theta1);
            F2 = obj.ComputeFWithTheta(I2, best_theta2);

            cost = F1 + F2;

            if cost < best_cost_final
                best_cost_final = cost;
                best_eta_final = eta;
            end
        end

        obj.final_cps(k) = best_eta_final;
    end
end
