function DDP(obj)
    %% Divided Dynamic Programming (Algorithm 2).
    % Runs dynamic programming on the coarse grid to obtain preliminary
    % change point estimates.
    %
    % - Usage:
    %   obj.DDP();
    %
    % - Output:
    %   obj.prelim_cps: preliminary change points.
    %   obj.B: DP cost array.
    %   obj.p_back: DP backpointer array.

    % Number of grid points (including endpoints).
    num_grid = length(obj.grid);

    % Initialize DP arrays.
    % B(i) = best cost up to grid point i.
    % p_back(i) = backpointer for grid point i.
    obj.B = inf(num_grid, 1);
    obj.p_back = -ones(num_grid, 1);

    % Base case: B(1) = gamma (cost at starting point 0).
    obj.B(1) = obj.gamma;

    % DP recurrence over grid points.
    % Start from 2 since we include 0.
    for r_idx = 2:num_grid
        r = obj.grid(r_idx);

        % Try all possible previous grid points.
        % End by r_idx - 1 since we need l < r.
        for l_idx = 1:(r_idx - 1)
            l = obj.grid(l_idx);

            % Define interval I = [l+1, r].
            % Change point at l marks the END of a segment, so the next
            % segment starts at l+1. This ensures non-overlapping segments.
            I = (l + 1):r;

            % Skip if interval is empty or shorter than buffer (min_interval_length).
            if isempty(I) || length(I) < obj.min_interval_length
                continue;
            end

            % Compute F(theta_hat_I, I).
            [F_val, ~] = obj.ComputeF(I);

            % Compute cost: b = B(l) + gamma + F(theta_hat_I, I).
            b = obj.B(l_idx) + obj.gamma + F_val;

            % Update if this is better.
            if b < obj.B(r_idx)
                obj.B(r_idx) = b;
                obj.p_back(r_idx) = l_idx;
            end
        end
    end

    % Backtracking to recover change points.
    obj.prelim_cps = [];
    k_idx = num_grid;
    while k_idx > 1
        h_idx = obj.p_back(k_idx);
        if h_idx <= 0
            break;
        end
        h = obj.grid(h_idx);
        if h > 0
            obj.prelim_cps = [h; obj.prelim_cps];
        end
        k_idx = h_idx;
    end
end
