function [weights, gmvp_ret, ann_mean, ann_vol, info_ratio] = evaluate_gmvp_strategy(Theta, returns)
    % Generate rolling GMVP weights from the selected covariance path and
    % return the corresponding annualized performance statistics.
    [p_local, ~, T_local] = size(Theta);
    if size(returns, 1) ~= T_local
        error('evaluate_gmvp_strategy:badLength', ...
            'Theta has T=%d slices but returns has %d rows.', ...
            T_local, size(returns, 1));
    end
    if size(returns, 2) ~= p_local
        error('evaluate_gmvp_strategy:badWidth', ...
            'Theta has p=%d assets but returns has %d columns.', ...
            p_local, size(returns, 2));
    end

    weights = zeros(p_local, T_local);
    gmvp_ret = zeros(T_local, 1);

    for tt = 2:T_local
        % The weight used at time tt leverages Theta from the previous slot
        % (tt-1) to mimic a one-step-ahead allocation rule.
        Sigma_t = (Theta(:, :, tt - 1) + Theta(:, :, tt - 1)') / 2;
        weights(:, tt) = solve_gmvp_weights(Sigma_t);
        gmvp_ret(tt) = weights(:, tt)' * returns(tt, :)';
    end

    % Drop the burn-in observation because the first weight is undefined.
    gmvp_ret = gmvp_ret(2:end);
    weights = weights(:, 2:end);

    ann_mean = 252 * mean(gmvp_ret);
    ann_vol = sqrt(252) * std(gmvp_ret);
    if ann_vol > 0
        info_ratio = ann_mean / ann_vol;
    else
        info_ratio = NaN;
    end
end

function weights = solve_gmvp_weights(Sigma)
    % Solve the unconstrained global minimum variance portfolio weights.
    p_local = size(Sigma, 2);
    ones_vec = ones(p_local, 1);
    Sigma_inv_ones = Sigma \ ones_vec;
    weights = Sigma_inv_ones / (ones_vec' * Sigma_inv_ones);
end
