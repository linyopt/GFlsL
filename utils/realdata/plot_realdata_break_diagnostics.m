function figure_files = plot_realdata_break_diagnostics(returns, dates, method_labels, break_lists, Sigma_paths, varargin)
%% Plot real-data break diagnostics against rolling covariance proxy (a = 0.01).
%
% The estimated covariance path is compared against the rolling blended proxy
% using the Frobenius norm of consecutive differences — large values suggest
% a structural change.
%
% The proxy is:
%   Sigma_blend = (1-a) * x_t x_t^T  +  a * Sigma_roll,
% where Sigma_roll is the sample covariance over [t-window+1, t].
% a = 0.01 keeps the proxy close to the instantaneous outer product so it
% reacts quickly to structural changes, with slight shrinkage to avoid
% being degenerate (rank-1).
%
% Estimated and proxy signals share separate y-axes because their absolute
% scales differ; only relative movement over time is comparable.
%
% - Usage:
%   figure_files = plot_realdata_break_diagnostics( ...
%       returns, dates, method_labels, break_lists, Sigma_paths);
%   figure_files = plot_realdata_break_diagnostics( ...
%       returns, dates, method_labels, break_lists, Sigma_paths, ...
%       window=42, out_dir='results/realdata/diagnostics', ...
%       file_prefix='SP500_100');
%
% - Input:
%   @returns: T-by-p return matrix.
%   @dates: T-by-1 datetime vector aligned with @returns.
%   @method_labels: Labels for the break-detection methods.
%   @break_lists: Cell array of estimated break-index vectors.
%   @Sigma_paths: Cell array of covariance paths, each of size p-by-p-by-T.
%
% - Optional input:
%   @window: Rolling-window length. Default 42.
%   @out_dir: Directory used to save the figures.
%   @file_prefix: Prefix prepended to the saved PNG filenames.
%
% - Output:
%   @figure_files: num_methods-by-1 cell array of saved PNG file paths.

    p_ = inputParser;
    addParameter(p_, 'window', 42, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    addParameter(p_, 'out_dir', fullfile('results', 'realdata', 'diagnostics'), ...
        @(x) ischar(x) || isstring(x));
    addParameter(p_, 'file_prefix', "", @(x) ischar(x) || isstring(x));
    parse(p_, varargin{:});

    window = p_.Results.window;
    out_dir = char(p_.Results.out_dir);
    file_prefix = string(p_.Results.file_prefix);

    labels = cellstr(string(method_labels(:)));
    break_lists = break_lists(:);
    Sigma_paths = Sigma_paths(:);
    num_methods = numel(labels);
    if numel(break_lists) ~= num_methods || numel(Sigma_paths) ~= num_methods
        error('plot_realdata_break_diagnostics:badInput', ...
            'method_labels, break_lists, and Sigma_paths must have the same length.');
    end

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % Rolling blended covariance proxy (a = 0.01), shared across all methods.
    [plot_dates, rolling_diff_fro] = ...
        rolling_covariance_signals(returns, dates, window);

    figure_files = cell(num_methods, 1);
    for kk = 1:num_methods
        Sigma_t = Sigma_paths{kk};
        if size(Sigma_t, 3) ~= size(returns, 1)
            error('plot_realdata_break_diagnostics:badPathLength', ...
                'Sigma path for "%s" has T=%d but returns has T=%d.', ...
                labels{kk}, size(Sigma_t, 3), size(returns, 1));
        end

        est_diff_fro = covariance_path_signals(Sigma_t, window);
        break_idx = normalize_break_indices(break_lists{kk});
        % Break indices mark the last observation of a segment; +1 shifts
        % the plotted line to the first observation of the new regime,
        % aligning the visual marker with the jump in Theta_t - Theta_{t-1}.
        break_idx = break_indices_to_plot_dates(break_idx, numel(dates));
        % Exclude breaks that fall outside the plotting window; break lines
        % before @window have no corresponding diagnostic data to align with.
        break_idx = break_idx(break_idx >= window & break_idx <= numel(dates));
        break_dates = dates(break_idx);

        file_stub = sanitize_label(labels{kk});
        if strlength(file_prefix) > 0
            file_stub = char(file_prefix + "_" + file_stub);
        end

        figure_files{kk} = render_diagnostic_figure( ...
            out_dir, file_stub, labels{kk}, plot_dates, ...
            est_diff_fro, rolling_diff_fro, break_dates);
    end
end

function figure_file = render_diagnostic_figure(out_dir, file_stub, method_label, ...
    plot_dates, est_diff_fro, rolling_diff_fro, break_dates)
%% Render a Frobenius-diff diagnostic figure with dual y-axes.
%
% Left axis = estimated covariance path, right axis = rolling proxy.
% The two signals have different absolute scales; placing them on separate
% y-axes avoids misleading visual comparisons of magnitude.

    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 380]);
    ax = axes(fig);
    hold(ax, 'on');

    title(ax, sprintf('%s  vs  Rolling covariance with a = 0.01', method_label), ...
        'Interpreter', 'none', 'FontSize', 14);
    figure_file = fullfile(out_dir, [file_stub, '.png']);

    legend_labels = {'Estimated covariance (left axis)'; ...
                     'Rolling covariance with a = 0.01 (right axis)'};

    yyaxis(ax, 'left');
    h_est = plot(ax, plot_dates, est_diff_fro, ...
        'Color', [0.20, 0.45, 0.80], 'LineWidth', 1.4);
    ylabel(ax, 'Estimated Frobenius norm of difference');

    yyaxis(ax, 'right');
    h_proxy = plot(ax, plot_dates, rolling_diff_fro, ...
        'Color', [0.10, 0.10, 0.10], 'LineWidth', 1.4, 'LineStyle', '-.');
    ylabel(ax, 'Rolling Frobenius norm of difference');
    % Color-code y-axis labels to match their corresponding lines.
    ax.YAxis(1).Color = h_est.Color;
    ax.YAxis(2).Color = [0.15, 0.15, 0.15];

    add_break_lines(ax, break_dates);
    hold(ax, 'off');
    grid(ax, 'off');
    enforce_zero_baseline(ax);
    % Legend lists estimated first, then proxy — matches handle order.
    legend(ax, [h_est; h_proxy], legend_labels, ...
        'Location', 'northoutside', 'Orientation', 'horizontal', 'FontSize', 11);
    format_month_axis(ax, plot_dates);
    xlabel(ax, 'Date');

    exportgraphics(fig, figure_file, 'Resolution', 300);
    close(fig);
end

function [plot_dates, rolling_diff_fro] = rolling_covariance_signals(returns, dates, window)
%% Evaluate the blended rolling covariance estimator for a = 0.01.
%
% The estimator at time t is:
%   Sigma_blend = (1-a) * x_t x_t^T  +  a * Sigma_roll,
% where Sigma_roll is the sample covariance over [t-window+1, t] and
% x_t is the current (centered) return.

    a = 0.01;
    T = size(returns, 1);
    if T < window + 1
        error('plot_realdata_break_diagnostics:shortSeries', ...
            'Need at least window+1=%d observations but only received %d.', ...
            window + 1, T);
    end

    plot_dates = dates(window:T);
    n_pts = T - window + 1;
    % First difference is NaN: no previous covariance to compare against.
    rolling_diff_fro = nan(n_pts, 1);
    Sigma_prev = [];

    for tt = window:T
        X_window = returns(tt - window + 1:tt, :);
        Sigma_roll = cov(X_window);
        % Enforce exact symmetry; floating-point arithmetic can break it.
        Sigma_roll = (Sigma_roll + Sigma_roll') / 2;

        x_t = returns(tt, :)';
        xt_xt = x_t * x_t';

        Sigma_blend = (1 - a) * xt_xt + a * Sigma_roll;
        idx = tt - window + 1;
        if ~isempty(Sigma_prev)
            rolling_diff_fro(idx) = norm(Sigma_blend - Sigma_prev, 'fro');
        end
        Sigma_prev = Sigma_blend;
    end
end

function sigma_diff_fro = covariance_path_signals(Sigma_t, window)
%% Align the estimated covariance-path Frobenius differences to the proxy.
%
% Aligned to the same time grid as the proxy signal (starting at t = window)
% so they can be plotted on a shared x-axis.
    T = size(Sigma_t, 3);
    sigma_diff_fro = nan(T - window + 1, 1);

    for tt = window:T
        Sigma_curr = (Sigma_t(:, :, tt) + Sigma_t(:, :, tt)') / 2;
        if tt > window
            Sigma_prev = (Sigma_t(:, :, tt - 1) + Sigma_t(:, :, tt - 1)') / 2;
            sigma_diff_fro(tt - window + 1) = norm(Sigma_curr - Sigma_prev, 'fro');
        end
    end
end

function break_idx = normalize_break_indices(break_idx)
%% Convert the stored break list into a clean sorted row vector.
    if isempty(break_idx)
        break_idx = [];
        return
    end
    if iscell(break_idx)
        if isscalar(break_idx)
            break_idx = break_idx{1};
        else
            break_idx = cell2mat(break_idx(:));
        end
    end
    if isempty(break_idx)
        break_idx = [];
        return
    end
    break_idx = unique(round(double(break_idx(:)')));
    break_idx = break_idx(isfinite(break_idx));
end

function break_plot_idx = break_indices_to_plot_dates(break_idx, T)
%% Breaks are stored as the last index of the previous segment.
    if isempty(break_idx)
        break_plot_idx = [];
        return
    end

    break_plot_idx = unique(break_idx + 1);
    break_plot_idx = break_plot_idx(break_plot_idx >= 1 & break_plot_idx <= T);
end

function add_break_lines(ax, break_dates)
%% Draw unlabeled vertical lines at the estimated break dates.
    for kk = 1:numel(break_dates)
        xline(ax, break_dates(kk), '--', 'Color', [0.85, 0.25, 0.20], ...
            'LineWidth', 1.0, 'HandleVisibility', 'off');
    end
end

function enforce_zero_baseline(ax)
%% Force both yy-axis scales to start at zero.
%
% Frobenius norm is non-negative by definition; starting axes at zero
% prevents exaggerated visual differences that a non-zero baseline would
% create.
    yyaxis(ax, 'left');
    yl = ylim(ax);
    ylim(ax, [0, positive_upper(yl(2))]);

    yyaxis(ax, 'right');
    yl = ylim(ax);
    ylim(ax, [0, positive_upper(yl(2))]);
end

function upper = positive_upper(upper)
%% Guard against degenerate all-zero paths.
    if ~isfinite(upper) || upper <= 0
        upper = 1;
    end
end

function format_month_axis(ax, plot_dates)
%% Keep only the monthly date ticks from the background series.
    ax.XAxis.TickLabelFormat = 'yyyy-MM';
    xlim(ax, [plot_dates(1), plot_dates(end)]);
end

function out = sanitize_label(label)
%% Convert a display label into a filesystem-friendly file stem.
    out = lower(char(label));
    out = regexprep(out, '[^a-z0-9]+', '_');
    out = regexprep(out, '^_+|_+$', '');
end