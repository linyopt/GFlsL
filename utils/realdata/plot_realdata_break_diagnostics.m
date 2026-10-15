function figure_files = plot_realdata_break_diagnostics(returns, dates, method_labels, break_lists, step_norms, varargin)
%% Break diagnostics of the estimated covariance paths, one figure per method.
%
% Every figure compares one method against the same rolling blended proxy
% through the Frobenius norm of consecutive covariance differences, where a
% large value suggests a structural change. What the figures need from an
% estimated path is its step norms ||Theta_t - Theta_{t-1}||_F, which is
% what @realdata_step_norms returns.
%
% A GFlsL criterion gets a figure of its own. Each competing detector gets a
% single figure carrying its native path and every one of its refits: those
% members share the change points of the detector and the regime-wise
% covariance path it returns, so their step norms are flat wherever no change
% point falls and differ only in the jump realized at a change point. That
% jump is marked once per member, and the marker shape together with the
% marker colour identifies the member. The curves themselves are drawn in one
% neutral grey, because the members coincide off the change points and a
% coloured curve would show only the colour of the member drawn last.
%
% Estimated and proxy signals sit on separate y-axes because their absolute
% scales differ; only the movement over time is comparable.
%
% Usage:
%   figure_files = plot_realdata_break_diagnostics( ...
%       returns_centered, dates, method_labels, method_breaks, step_norms);
%   figure_files = plot_realdata_break_diagnostics( ...
%       returns_centered, dates, method_labels, method_breaks, step_norms, ...
%       window=42, out_dir='results/realdata/diagnostics/per_method', ...
%       file_prefix='SP100_100');
%
% - Input:
%   @returns: T-by-p centered return matrix.
%   @dates: T-by-1 date vector aligned with @returns.
%   @method_labels: Labels of the candidate methods; every method the figures
%       report is selected from these by name.
%   @break_lists: Cell array of estimated break-index vectors, aligned with
%       @method_labels.
%   @step_norms: (T-1)-by-numel(method_labels) matrix of Frobenius norms of
%       consecutive covariance differences, as returned by
%       @realdata_step_norms.
%
% - Optional input:
%   @window: Rolling-window length of the proxy. Default 42.
%   @out_dir: Directory used to save the figures.
%   @file_prefix: Prefix prepended to the saved PNG filenames.
%
% - Output:
%   @figure_files: numel(diagnostic_groups())-by-1 cell array of saved PNG
%       file paths.

    p_ = inputParser;
    addParameter(p_, 'window', 42, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    addParameter(p_, 'out_dir', ...
        fullfile('results', 'realdata', 'diagnostics', 'per_method'), ...
        @(x) ischar(x) || isstring(x));
    addParameter(p_, 'file_prefix', "", @(x) ischar(x) || isstring(x));
    parse(p_, varargin{:});

    window = p_.Results.window;
    out_dir = char(p_.Results.out_dir);
    file_prefix = string(p_.Results.file_prefix);

    labels = cellstr(string(method_labels(:)));
    break_lists = break_lists(:);
    num_methods = numel(labels);
    if numel(break_lists) ~= num_methods
        error('plot_realdata_break_diagnostics:badInput', ...
            'method_labels and break_lists must have the same length.');
    end
    if size(step_norms, 1) ~= size(returns, 1) - 1 || ...
            size(step_norms, 2) ~= num_methods
        error('plot_realdata_break_diagnostics:badStepNorms', ...
            ['step_norms must be (T-1)-by-numel(method_labels); got ' ...
            '%d-by-%d for T=%d and %d methods.'], ...
            size(step_norms, 1), size(step_norms, 2), ...
            size(returns, 1), num_methods);
    end

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % Rolling blended covariance proxy, shared by every figure.
    [plot_dates, proxy_diff_fro] = ...
        rolling_covariance_signals(returns, dates, window);

    groups = diagnostic_groups();
    figure_files = cell(numel(groups), 1);
    for gg = 1:numel(groups)
        member_idx = resolve_members(labels, groups(gg).member_labels);

        % step_norms(i, kk) is the difference realized at time i+1. The
        % plotted series starts at t = window, where the first difference has
        % no predecessor and is therefore NaN.
        series = [nan(1, numel(member_idx)); step_norms(window:end, member_idx)];

        break_idx = cell(numel(member_idx), 1);
        for mm = 1:numel(member_idx)
            break_idx{mm} = plot_break_indices( ...
                break_lists{member_idx(mm)}, window, numel(dates));
        end

        file_stub = groups(gg).stem;
        if strlength(file_prefix) > 0
            file_stub = char(file_prefix + "_" + file_stub);
        end

        if isscalar(member_idx)
            figure_files{gg} = render_path_figure(out_dir, file_stub, ...
                groups(gg).title, plot_dates, proxy_diff_fro, series, ...
                dates, break_idx{1});
        else
            figure_files{gg} = render_refit_figure(out_dir, file_stub, ...
                groups(gg).title, plot_dates, proxy_diff_fro, series, ...
                dates, window, break_idx, groups(gg).member_names);
        end
    end
end

% =========================================================================

function groups = diagnostic_groups()
%% The reported methods, one figure each, with the members of a detector merged.
%
% The stored method labels are the stable keys, so the method set the figures
% report is assembled from them here rather than by the caller. Every group
% names its members exactly as they appear in the cached labels, and gives
% each member of a detector the short name that labels its marker.

    criteria = ["BIC", "HBIC", "HBICG", "HBICG(0,2)", "HBIC_c2"];
    variants = ["Non-adaptive", "Adaptive"];

    groups = repmat(struct('stem', '', 'title', '', 'member_labels', {{}}, ...
        'member_names', {{}}), 2 * numel(criteria) + 2, 1);
    kk = 0;
    for cc = 1:numel(criteria)
        for vv = 1:numel(variants)
            kk = kk + 1;
            label = char(variants(vv) + " " + criteria(cc));
            groups(kk).stem = sanitize_label(label);
            groups(kk).title = label;
            groups(kk).member_labels = {label};
        end
    end
    groups(kk + 1) = detector_group("BSOP", ...
        {'BSOP + EC2', 'BSOP + Xue-Ma-Zou', 'BSOP + Adaptive+FSPD'}, ...
        {'EC2', 'Xue-Ma-Zou', 'Adaptive+FSPD'});
    groups(kk + 2) = detector_group("TBFL", ...
        {'TBFL-native', 'TBFL + EC2', 'TBFL + Xue-Ma-Zou', ...
         'TBFL + Adaptive+FSPD'}, ...
        {'Native', 'EC2', 'Xue-Ma-Zou', 'Adaptive+FSPD'});
end

function group = detector_group(detector, member_labels, member_names)
%% One figure for a competing detector, carrying its members and their names.
    group.stem = sanitize_label(char(detector));
    group.title = char(detector);
    group.member_labels = member_labels;
    group.member_names = member_names;
end

function member_idx = resolve_members(labels, members)
%% Positions of a group's member labels in the stored label list.
    member_idx = zeros(numel(members), 1);
    for mm = 1:numel(members)
        idx = find(strcmp(labels, members{mm}), 1);
        if isempty(idx)
            error('plot_realdata_break_diagnostics:missingMethod', ...
                'No method labelled "%s" in the supplied results.', ...
                members{mm});
        end
        member_idx(mm) = idx;
    end
end

function break_plot_idx = plot_break_indices(break_idx, window, T)
%% Shift stored break indices onto the plotted series and drop what falls out.
%
% A break index marks the last observation of a segment; +1 shifts the
% plotted position to the first observation of the new regime, aligning the
% visual marker with the jump in Theta_t - Theta_{t-1}. Breaks that precede
% the first window have no diagnostic series to align with.
    break_idx = normalize_break_indices(break_idx);
    break_plot_idx = unique(break_idx + 1);
    break_plot_idx = break_plot_idx( ...
        break_plot_idx >= window & break_plot_idx <= T);
end

% =========================================================================

function figure_file = render_path_figure(out_dir, file_stub, method_label, ...
        plot_dates, proxy_diff_fro, series, dates, break_idx)
%% Render the diagnostic figure of a single method with dual y-axes.
%
% Left axis = estimated covariance path, right axis = rolling proxy. The two
% signals have different absolute scales; placing them on separate y-axes
% avoids misleading visual comparisons of magnitude.

    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 380]);
    ax = axes(fig);
    hold(ax, 'on');

    title(ax, sprintf('%s  vs  %s', method_label, proxy_label()), ...
        'Interpreter', 'none', 'FontSize', 14);
    figure_file = fullfile(out_dir, [file_stub, '.png']);

    yyaxis(ax, 'left');
    h_est = plot(ax, plot_dates, series, ...
        'Color', [0.20, 0.45, 0.80], 'LineWidth', 1.4);
    ylabel(ax, 'Estimated Frobenius norm of difference');
    % Colour-code y-axis labels to match their corresponding lines.
    ax.YAxis(1).Color = h_est.Color;

    h_proxy = draw_proxy(ax, plot_dates, proxy_diff_fro);

    add_break_lines(ax, dates(break_idx));
    hold(ax, 'off');
    grid(ax, 'off');
    enforce_zero_baseline(ax);
    legend(ax, [h_est; h_proxy], ...
        {'Estimated covariance (left axis)'; ...
         sprintf('%s (right axis)', proxy_label())}, ...
        'Location', 'northoutside', 'Orientation', 'horizontal', ...
        'FontSize', 11);
    format_month_axis(ax, plot_dates);

    exportgraphics(fig, figure_file, 'Resolution', 300);
    close(fig);
end

function figure_file = render_refit_figure(out_dir, file_stub, detector_label, ...
        plot_dates, proxy_diff_fro, series, dates, window, break_idx, names)
%% Render the diagnostic figure of a detector, one marker per member.
%
% The members coincide off the change points and separate only at the change
% points, where each realizes its own jump. The jump of a member is therefore
% marked at every change point, and the marker shape and colour are the legend
% key of the member.

    curve_colour = [0.72, 0.72, 0.72];

    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 380]);
    ax = axes(fig);
    hold(ax, 'on');

    title(ax, sprintf('%s  vs  %s', detector_label, proxy_label()), ...
        'Interpreter', 'none', 'FontSize', 14);
    figure_file = fullfile(out_dir, [file_stub, '.png']);

    yyaxis(ax, 'left');
    h_member = gobjects(numel(names), 1);
    for mm = 1:numel(names)
        [marker_, colour] = member_style(names{mm});
        plot(ax, plot_dates, series(:, mm), '-', ...
            'Color', curve_colour, 'LineWidth', 0.8);
        if ~isempty(break_idx{mm})
            % The plotted series starts at the observation indexed by @window.
            jumps = series(break_idx{mm} - window + 1, mm);
            plot(ax, dates(break_idx{mm}), jumps, marker_, 'LineStyle', 'none', ...
                'MarkerSize', 6, 'MarkerFaceColor', colour, ...
                'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');
        end
        % Legend sample: the marker on its own, carrying no data of its own.
        h_member(mm) = plot(ax, nan, nan, marker_, 'LineStyle', 'none', ...
            'MarkerSize', 6, 'MarkerFaceColor', colour, ...
            'MarkerEdgeColor', 'none');
    end
    ylabel(ax, 'Estimated Frobenius norm of difference');
    % The left axis carries one member per colour, so it is left neutral.
    ax.YAxis(1).Color = [0.15, 0.15, 0.15];

    h_proxy = draw_proxy(ax, plot_dates, proxy_diff_fro);

    % The members share their change points, so the line at a date is drawn
    % once over the union of the member break lists.
    add_break_lines(ax, dates(unique(cell2mat(break_idx(:)'))));
    hold(ax, 'off');
    grid(ax, 'off');
    enforce_zero_baseline(ax);
    legend(ax, [h_member; h_proxy], ...
        [names(:); {sprintf('%s (right axis)', proxy_label())}], ...
        'Location', 'northoutside', 'Orientation', 'horizontal', ...
        'FontSize', 11);
    format_month_axis(ax, plot_dates);

    exportgraphics(fig, figure_file, 'Resolution', 300);
    close(fig);
end

function h_proxy = draw_proxy(ax, plot_dates, proxy_diff_fro)
%% Right-axis trace of the shared rolling proxy.
    yyaxis(ax, 'right');
    h_proxy = plot(ax, plot_dates, proxy_diff_fro, ...
        'Color', [0.10, 0.10, 0.10], 'LineWidth', 1.4, 'LineStyle', '-.');
    ylabel(ax, 'Rolling Frobenius norm of difference');
    ax.YAxis(2).Color = [0.15, 0.15, 0.15];
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
%% Keep readable date ticks when dates are available.
    if isdatetime(plot_dates)
        ax.XAxis.TickLabelFormat = 'yyyy-MM';
    end
    xlim(ax, [plot_dates(1), plot_dates(end)]);
end

function label = proxy_label()
%% Name of the shared proxy, as it appears in a title and a legend.
    label = 'rolling blended proxy (a = 0.95)';
end

function [marker_, colour] = member_style(name)
%% Marker shape and colour identifying a member of a detector figure.
%
% The style is keyed by the member name, so a member carries the same marker
% and colour in every detector figure that reports it.
% The shapes and the colours are both distinctive, so that the members remain
% separable in a greyscale print of the figure.
    switch name
        case 'Native',        marker_ = 'd';  colour = [0.49, 0.18, 0.56];
        case 'EC2',           marker_ = 'o';  colour = [0.15, 0.42, 0.78];
        case 'Xue-Ma-Zou',    marker_ = 's';  colour = [0.90, 0.55, 0.10];
        case 'Adaptive+FSPD', marker_ = '^';  colour = [0.20, 0.60, 0.35];
        otherwise
            error('plot_realdata_break_diagnostics:unknownMember', ...
                'No marker is assigned to the detector member "%s".', name);
    end
end

function break_idx = normalize_break_indices(break_idx)
%% Convert the stored break list into a clean sorted row vector.
    if isempty(break_idx)
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
        return
    end
    break_idx = unique(round(double(break_idx(:)')));
    break_idx = break_idx(isfinite(break_idx));
end

function out = sanitize_label(label)
%% Convert a display label into a filesystem-friendly file stem.
    out = lower(char(label));
    out = regexprep(out, '[^a-z0-9]+', '_');
    out = regexprep(out, '^_+|_+$', '');
end
