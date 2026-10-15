function figure_files = replot_realdata_diagnostics(plot_cache_file, window)
%% Redraw the per-method break-diagnostic figures from a cached plot payload.
%
% @runcode_realdata writes @plot_cache_file (plot_data_<portfolio>_<scale>.mat)
% next to its log, holding the step norms of every estimated covariance path
% together with the dates, the break lists and the centered returns. Redrawing
% the per-method figures therefore needs no refit.
%
% Usage:
%   figure_files = replot_realdata_diagnostics( ...
%       'results/realdata/plot_data_SP100_100.mat');
%   figure_files = replot_realdata_diagnostics(cache_file, 63);
%
% - Input:
%   @plot_cache_file: Path of the cached plot payload.
%   @window: Optional rolling-window length. Defaults to the value used when
%       the cache was written.
%
% - Output:
%   @figure_files: Cell array of saved PNG file paths.

    if nargin < 2
        window = [];
    end

    loaded = load(plot_cache_file, 'plot_data');
    if ~isfield(loaded, 'plot_data')
        error('replot_realdata_diagnostics:missingPayload', ...
            'No plot_data struct in %s.', plot_cache_file);
    end
    plot_data = loaded.plot_data;

    if isempty(window)
        window = plot_data.window;
    end

    output_dir = fileparts(plot_cache_file);
    file_stub = sprintf('%s_%g', char(plot_data.portfolio), plot_data.scale);
    figure_files = plot_realdata_break_diagnostics( ...
        plot_data.returns_centered, plot_data.dates, ...
        plot_data.method_labels, plot_data.method_breaks, ...
        plot_data.step_norms, window=window, ...
        out_dir=fullfile(output_dir, 'diagnostics', file_stub, 'per_method'), ...
        file_prefix=file_stub);
end
