function [plot_dates, rolling_diff_fro] = rolling_covariance_signals(returns, dates, window, blend_weight)
%% Frobenius norm of consecutive differences of the blended rolling covariance.
%
% The proxy at time t is
%   Sigma_blend = (1 - a) * x_t x_t^T + a * Sigma_roll,
% where Sigma_roll is the sample covariance over [t - window + 1, t] and x_t
% is the current centered return row. The reported signal is
%   ||Sigma_blend,t - Sigma_blend,t-1||_F.
% The weight a trades the responsiveness of the outer product x_t x_t^T,
% which tracks the current observation exactly but is degenerate, against the
% stability of the rolling sample covariance.
%
% Usage:
%   [plot_dates, rolling_diff_fro] = rolling_covariance_signals(returns, dates, 42);
%   [plot_dates, rolling_diff_fro] = rolling_covariance_signals(returns, dates, 42, 0.5);
%
% - Input:
%   @returns: T-by-p centered return matrix.
%   @dates: T-by-1 vector of observation dates aligned with @returns.
%   @window: Rolling-window length in observations.
%   @blend_weight: Optional weight a on the rolling term, in [0, 1].
%       Default 0.95.
%
% - Output:
%   @plot_dates: dates(window:T), the dates carrying a usable signal.
%   @rolling_diff_fro: (T - window + 1)-by-1 Frobenius norms; the first entry
%       is NaN because the first window has no predecessor.

    if nargin < 4 || isempty(blend_weight)
        blend_weight = 0.95;
    end
    if ~isscalar(blend_weight) || blend_weight < 0 || blend_weight > 1
        error('rolling_covariance_signals:badWeight', ...
            'blend_weight must be a scalar in [0, 1], got %g.', blend_weight);
    end

    a = blend_weight;
    T = size(returns, 1);
    if T < window + 1
        error('rolling_covariance_signals:shortSeries', ...
            'Need at least window+1=%d observations but only received %d.', ...
            window + 1, T);
    end

    plot_dates = dates(window:T);
    n_pts = T - window + 1;
    rolling_diff_fro = nan(n_pts, 1);
    Sigma_prev = [];

    for tt = window:T
        X_window = returns(tt - window + 1:tt, :);
        Sigma_roll = cov(X_window);
        % Enforce exact symmetry; floating-point arithmetic can break it.
        Sigma_roll = (Sigma_roll + Sigma_roll') / 2;

        x_t = returns(tt, :)';
        Sigma_blend = (1 - a) * (x_t * x_t') + a * Sigma_roll;

        idx = tt - window + 1;
        if ~isempty(Sigma_prev)
            rolling_diff_fro(idx) = norm(Sigma_blend - Sigma_prev, 'fro');
        end
        Sigma_prev = Sigma_blend;
    end
end
