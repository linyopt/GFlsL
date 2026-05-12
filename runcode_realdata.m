%% Real data
addpath(genpath(pwd))
clear
clc

data_dir = 'data';

% Choose the dataset and an optional global scale on returns. If the scale
% changes, the tuning grids below should be adjusted proportionally.
portfolio = "SP500";
scale = 100;
[returns_raw, dates, timeframe] = load_realdata_panel(data_dir, portfolio, scale);

% Center the real-data panel once up front and record the removed sample
% mean so the run settings capture the exact estimation preprocessing.
sample_mean_returns = mean(returns_raw, 1);
returns_centered = returns_raw - sample_mean_returns;

% timeframe = 1333:1533;
% returns_raw = returns_raw(timeframe, :);
% returns_centered = returns_centered(timeframe, :);
% dates = dates(timeframe);

% If a short prefix is selected manually above, keep the logged timeframe
% and centering statistics aligned with the truncated sample so the later
% sections reload the same panel.
if size(returns_raw, 1) ~= numel(timeframe) || ...
        size(returns_centered, 1) ~= numel(timeframe) || ...
        numel(dates) ~= numel(timeframe)
    n_keep = min([size(returns_raw, 1), size(returns_centered, 1), ...
        numel(dates), numel(timeframe)]);
    returns_raw = returns_raw(1:n_keep, :);
    dates = dates(1:n_keep, :);
    timeframe = timeframe(1:n_keep);
    sample_mean_returns = mean(returns_raw, 1);
    returns_centered = returns_raw - sample_mean_returns;
end

[T, p] = size(returns_centered);

% Tuning grids for GFlsL (scale-sensitive; adjust proportionally to
% scale^2).
lamb1s = p * (5e-6:1e-5:1e-4) ./ ((100 / scale) ^ 2);
lamb2s = p * (3e-2:5e-3:1e-1) ./ ((100 / scale) ^ 2);
lamb2s2 = p * (4e-3:3e-3:3e-2) ./ ((100 / scale) ^ 2);

mu2 = 1.5;
mu1 = 0.8;
kappa_ = 0.5;
tol = 1e-3;
beta_ = 0.05;

% The threshold for BSOP.
tau_BSOP = 10;

% The delta and the threshold for WBSIP, according to their code.
% WBSIP is not used in realdata, present them for consistency.
delta_WBSIP = 5;
M_WBSIP = 100;
tau_WBSIP = sqrt(p * log(T)) * 1.5;

% Real data does not come with an oracle structure label, so include only
% the structure-agnostic post-refit estimators here. Convex banding is
% omitted because it is only used for the ordered banded simulation design.
post_refit_methods = ["ec2", "xue_ma_zou", "adaptive_threshold_fspd"];
lambda_post = NaN;
diagnostic_window = 42;

% The file prefix.
results_dir = fullfile('results', 'realdata');
filepre = string(fullfile(results_dir, sprintf('%s_%g_', char(portfolio), scale)));

% Ensure results directory exists before writing any output.
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

settings = struct( ...
    'portfolio', portfolio, ...
    'scale', scale, ...
    'timeframe', timeframe, ...
    'center_data', true, ...
    'sample_mean_returns', sample_mean_returns, ...
    'lamb1s', lamb1s, ...
    'lamb2s', lamb2s, ...
    'lamb2s2', lamb2s2, ...
    'tau_BSOP', tau_BSOP, ...
    'post_refit_methods', post_refit_methods, ...
    'lambda_post', lambda_post, ...
    'mu2', mu2, ...
    'mu1', mu1, ...
    'kappa_', kappa_, ...
    'tol', tol, ...
    'beta_', beta_, ...
    'diagnostic_window', diagnostic_window);

% With X_test empty, get_two_stage_est skips the WBSIP and DCDP paths
% automatically and only evaluates the competitors available on real data.
get_two_stage_est(returns_centered, [], [], [], lamb1s, lamb2s, lamb2s2, mu2=mu2, ...
    mu1=mu1, kappa_=kappa_, tol=tol, beta_=beta_, tau_BSOP=tau_BSOP, ...
    delta_WBSIP=delta_WBSIP, tau_WBSIP=tau_WBSIP, M_WBSIP=M_WBSIP, ...
    showbar=1, filepre=filepre, showfig=0, dates=dates, savetxt=1, ...
    savemat=1, merge_breaks=false, save_tbfl_omega=true, ...
    save_post_refit_paths=true, ...
    post_refit_methods=post_refit_methods, lambda_post=lambda_post);

% Persist run settings so downstream analysis/logs know the scale/tuning grid.
save(filepre + ".mat", 'settings', '-append');

%% Portfolio allocation experiment

% Reuse the prior portfolio/scale choice when available. Keep defaults here
% so this section can be run on its own from the Matlab editor.
if ~exist('portfolio', 'var') || isempty(portfolio)
    portfolio = "SP500";
end
if ~exist('scale', 'var') || isempty(scale)
    scale = 100;
end
if ~exist('data_dir', 'var') || isempty(data_dir)
    data_dir = 'data';
end

% Entry point for the portfolio-allocation experiment.  The function loads
% the saved estimation outputs, assembles all competitor covariance paths,
% runs the GMVP backtest, produces diagnostic plots, and saves everything.
run_realdata_portfolio_experiment(data_dir, portfolio, scale);

%% Local functions.
% =========================================================================
function run_realdata_portfolio_experiment(data_dir, portfolio, scale)
    % Main driver: load estimation results, collect all competitor methods,
    % evaluate GMVP portfolios, produce break diagnostics, and save outputs.

    % Build the output-file prefix and check that the estimation MAT file
    % from the first section exists before doing anything else.
    filepre = string(fullfile('results', 'realdata', ...
        sprintf('%s_%g_', char(portfolio), scale)));
    mat_filename = filepre + ".mat";
    if ~isfile(mat_filename)
        error("Result file %s not found. Run the estimation section first.", ...
            mat_filename);
    end

    % Load the saved estimation outputs and reconstruct the return panels.
    [data, settings] = load_realdata_estimation_data(mat_filename, ...
        data_dir, portfolio, scale);

    % Experiment index within the saved tensor arrays.  get_two_stage_est
    % reads num_exps = size(X,3) and appends it as the last dimension to
    % every saved tensor (Thetans, BICns, EstBreaksn, post_paths, ...),
    % so the same MAT-file layout supports multi-experiment simulation
    % runs (num_exps > 1) and real-data panels (num_exps = 1).  Removing
    % this parameter would require hardcoding {1}, (:,:,1), {det,mm,1}
    % etc. across ~50 indexing places in 10 sub-functions, and would also
    % eliminate the multi-dimensional branches in extract_saved_returns
    % and get_saved_num_exps.  Kept as a named variable for clarity and
    % simplification.
    exp_idx = 1;
    
    % Read centered returns and truncate the dates if necessary.
    [returns_centered, dates] = ...
        load_realdata_portfolio_returns(data, settings, data_dir, exp_idx);

    % Open (or create) the append-only results log and guarantee it is
    % closed on exit, even if an error is thrown below.
    log_filename = fullfile('results', 'realdata', 'realdata_results.txt');
    log_fid = fopen(log_filename, 'a');
    if log_fid < 0
        error('runcode_realdata:openLogFailed', ...
            'Cannot open log file: %s', log_filename);
    end
    cleanup_log = onCleanup(@() fclose(log_fid));

    fprintf(log_fid, "\n======== Portfolio: %s ========\n", ...
        char(settings.portfolio));
    write_settings_block(log_fid, settings);

    % Collect every competitor covariance path: GFlsL (6 IC variants),
    % TBFL-native, post-refit × detector, and baselines.
    [gflsl_labels, gflsl_break_labels, gflsl_thetas, gflsl_breaks] = ...
        collect_realdata_gflsl_methods(data, exp_idx);
    [Theta_TBFL, tbfl_breaks] = load_tbfl_realdata_path(data, ...
        mat_filename, returns_centered, exp_idx);
    [post_labels, post_thetas, post_breaks] = collect_realdata_post_refits( ...
        data, settings, returns_centered, exp_idx, tbfl_breaks);

    % Assemble raw and merged break-location tables.  Only GFlsL breaks
    % go through mergeBreaks; baseline breaks are kept as-is.
    baseline_break_labels = {'BSOP'; 'TBFL'};
    baseline_breaks = {data.BScps{exp_idx}; tbfl_breaks};
    break_labels_raw = [gflsl_break_labels; baseline_break_labels];
    break_lists_raw = [gflsl_breaks; baseline_breaks];
    break_labels_merged = break_labels_raw;
    break_lists_merged = [cellfun(@mergeBreaks, gflsl_breaks, ...
        'UniformOutput', false); baseline_breaks];

    write_break_lists(log_fid, 'Break locations (no merge)', ...
        break_labels_raw, break_lists_raw, dates);
    write_break_lists(log_fid, 'Break locations (merged)', ...
        break_labels_merged, break_lists_merged, dates);

    % Concatenate all methods into a single ordering used by the diagnostic
    % plots and the GMVP evaluation table below.
    method_labels = [gflsl_labels; {'TBFL-native'}; post_labels];
    method_thetas = [gflsl_thetas; {Theta_TBFL}; post_thetas];
    method_breaks = [gflsl_breaks; {tbfl_breaks}; post_breaks];

    % Produce break-diagnostic figures and evaluate every method's GMVP
    % portfolio performance against the equal-weight benchmark.
    diagnostic_dir = fullfile('results', 'realdata', 'diagnostics', ...
        sprintf('%s_%g', char(settings.portfolio), settings.scale));
    diagnostic_figures = plot_realdata_break_diagnostics(returns_centered, ...
        dates, method_labels, method_breaks, method_thetas, ...
        window=settings.diagnostic_window, out_dir=diagnostic_dir, ...
        file_prefix=sprintf('%s_%g', char(settings.portfolio), ...
        settings.scale));

    [gmvp_weights, gmvp_returns, ann_mean, ann_vol, info_ratio] = ...
        evaluate_realdata_portfolios(method_thetas, returns_centered);
    portfolio_labels = [method_labels; {'Equal Weight'}];
    portfolio_table = table(ann_mean, ann_vol, info_ratio, ...
        'VariableNames', {'AnnualMean', 'AnnualVol', 'InformationRatio'}, ...
        'RowNames', portfolio_labels);

    fprintf(log_fid, '\n\nPortfolio allocation summary (annualized metrics):\n');
    write_portfolio_table(log_fid, portfolio_labels, ann_mean, ann_vol, ...
        info_ratio);
    fprintf(log_fid, '\nBreak-diagnostic figures saved to: %s\n', ...
        diagnostic_dir);

    % Persist all portfolio results so they can be reloaded for later
    % plotting or analysis without rerunning the GMVP backtest.
    gmvp_labels = method_labels;
    diagnostic_labels = method_labels;
    diagnostic_breaks = method_breaks;
    save(filepre + "_gmvp.mat", 'gmvp_labels', 'gmvp_weights', ...
        'gmvp_returns', 'portfolio_table', 'settings', 'break_labels_raw', ...
        'break_lists_raw', 'break_labels_merged', 'break_lists_merged', ...
        'diagnostic_labels', 'diagnostic_breaks', 'diagnostic_figures', ...
        '-v7.3');
end

% =========================================================================
% Load saved estimation outputs (covariance paths, IC values, breaks) and
% reconstruct a complete settings struct with safe defaults for any missing
% fields so the portfolio section is self-contained.
function [data, settings] = load_realdata_estimation_data(mat_filename, ...
        data_dir, portfolio, scale)
    % Load only the saved variables needed by the portfolio section.
    vars_to_load = {'X', 'Thetans', 'Theta2s', 'EstBreaksn', 'EstBreaks2', ...
                    'BICns', 'HBIC_ns', 'HBICG_ns', 'BIC2s', 'HBIC_2s', ...
                    'HBICG_2s', 'BScps', 'TBFLcps', 'TBFL_Omega', ...
                    'post_detectors', 'post_methods', 'post_paths', ...
                    'tbfl_native_paths', 'settings'};
    available_vars = who('-file', mat_filename);
    vars_to_load = vars_to_load(ismember(vars_to_load, available_vars));
    data = load(mat_filename, vars_to_load{:});

    % Retrieve the saved settings struct.  When it is missing (old MAT
    % files), build a minimal fallback so the portfolio section can still
    % run without the estimation workspace variables.
    if isfield(data, 'settings')
        settings = data.settings;
    else
        [~, ~, timeframe] = load_realdata_panel(data_dir, portfolio, scale);
        settings = struct('portfolio', portfolio, 'scale', scale, ...
            'timeframe', timeframe);
    end

    % Fill in any fields that may be absent from older settings structs
    % so downstream code can always assume they exist.
    if ~isfield(settings, 'portfolio') || isempty(settings.portfolio)
        settings.portfolio = portfolio;
    end
    if ~isfield(settings, 'scale') || isempty(settings.scale)
        settings.scale = scale;
    end
    if ~isfield(settings, 'timeframe') || isempty(settings.timeframe)
        [~, ~, settings.timeframe] = load_realdata_panel(data_dir, ...
            settings.portfolio, settings.scale);
    end
    if ~isfield(settings, 'lambda_post')
        settings.lambda_post = NaN;
    end
    if ~isfield(settings, 'diagnostic_window')
        settings.diagnostic_window = 42;
    end
    if ~isfield(settings, 'center_data')
        settings.center_data = true;
    end
end

% =========================================================================
% Reconstruct the centered estimation returns and truncates the date
% vector to the saved estimation sample length.
function [returns_centered, dates] = ...
        load_realdata_portfolio_returns(data, settings, data_dir, exp_idx)
    % Load the raw returns panel (needed for dates and as fallback when the
    % saved centered returns are unavailable).
    [returns_raw, dates, ~] = load_realdata_panel(data_dir, ...
        settings.portfolio, settings.scale, settings.timeframe);
    returns_centered = extract_saved_returns(data, exp_idx);

    % When the saved centered returns are not available, re-center the raw
    % panel using the stored sample mean (or the overall column mean).
    if isempty(returns_centered)
        if isfield(settings, 'sample_mean_returns') && ...
                ~isempty(settings.sample_mean_returns) && ...
                size(settings.sample_mean_returns, 2) == size(returns_raw, 2)
            returns_centered = returns_raw - settings.sample_mean_returns;
        else
            returns_centered = returns_raw - mean(returns_raw, 1);
        end
    end

    % Truncate the date vector to the estimation sample length so all
    % downstream consumers share the same time axis.
    T = size(returns_centered, 1);
    if size(returns_raw, 1) < T
        error('runcode_realdata:badRealdataLength', ...
            'Loaded returns has T=%d but saved estimation sample has T=%d.', ...
            size(returns_raw, 1), T);
    end
    dates = dates(1:T, :);
end

% =========================================================================
% Collect the six GFlsL methods: non-adaptive × {BIC, HBIC, HBICG} and
% adaptive × {BIC, HBIC, HBICG}.  Each method's Theta path and break list
% are selected by the corresponding information criterion.
function [method_labels, break_labels, method_thetas, method_breaks] = ...
        collect_realdata_gflsl_methods(data, exp_idx)
    % Gather the six GFlsL covariance paths selected by BIC, HBIC, and
    % HBICG for the non-adaptive and adaptive fits.

    % Map each criterion name to its metric-field name for the non-adaptive
    % and adaptive grids stored in the MAT file.
    crit_names = {'BIC', 'HBIC', 'HBICG'};
    metric_fields_na = {'BICns', 'HBIC_ns', 'HBICG_ns'};
    metric_fields_ad = {'BIC2s', 'HBIC_2s', 'HBICG_2s'};

    method_labels = cell(6, 1);
    break_labels = cell(6, 1);
    method_thetas = cell(6, 1);
    method_breaks = cell(6, 1);

    % Each iteration fills slots kk (non-adaptive) and kk+3 (adaptive)
    % by minimizing the corresponding information criterion over its grid.
    for kk = 1:3
        method_labels{kk} = sprintf('Non-adaptive %s', crit_names{kk});
        break_labels{kk} = sprintf('GFlsL NA %s', crit_names{kk});
        method_thetas{kk} = get_nonadaptive_theta( ...
            data.(metric_fields_na{kk}), data.Thetans, exp_idx);
        method_breaks{kk} = get_nonadaptive_breaks( ...
            data.(metric_fields_na{kk}), data.EstBreaksn, exp_idx);

        method_labels{kk + 3} = sprintf('Adaptive %s', crit_names{kk});
        break_labels{kk + 3} = sprintf('GFlsL AD %s', crit_names{kk});
        method_thetas{kk + 3} = get_adaptive_theta( ...
            data.(metric_fields_ad{kk}), data.Theta2s, exp_idx);
        method_breaks{kk + 3} = get_adaptive_breaks( ...
            data.(metric_fields_ad{kk}), data.EstBreaks2, exp_idx);
    end
end

% =========================================================================
% Retrieve the TBFL covariance path via a three-tier fallback:
%   1. Saved native covariance path (fastest).
%   2. Saved regime-wise precision Omega → invert to covariance.
%   3. Re-run TBFL once and persist the results for future reuse.
function [Theta_TBFL, tbfl_breaks] = load_tbfl_realdata_path(data, ...
        mat_filename, returns_centered, exp_idx)
    % Reuse the saved TBFL covariance path when possible. Otherwise rebuild
    % it once and append it to the saved MAT file.

    % Recover the saved TBFL break vector (may be empty when no breaks
    % were detected).
    if isfield(data, 'TBFLcps') && numel(data.TBFLcps) >= exp_idx
        tbfl_breaks = data.TBFLcps{exp_idx};
    else
        tbfl_breaks = [];
    end

    % Tier 1: use the saved native covariance path when available.
    if isfield(data, 'tbfl_native_paths') && ...
            numel(data.tbfl_native_paths) >= exp_idx && ...
            ~isempty(data.tbfl_native_paths{exp_idx})
        Theta_TBFL = data.tbfl_native_paths{exp_idx};
        return
    end

    % Tier 2: reconstruct from the saved regime-wise precision Omega.
    if isfield(data, 'TBFL_Omega') && numel(data.TBFL_Omega) >= exp_idx && ...
            ~isempty(data.TBFL_Omega{exp_idx})
        Theta_TBFL = invert_precision_segments(data.TBFL_Omega{exp_idx}, ...
            size(returns_centered, 1), tbfl_breaks, min_eig=0.01);
        return
    end

    % Tier 3: neither saved path nor saved Omega exists, so re-run TBFL
    % once on the centered returns and persist the results below.
    tbfl_est = TBFL(X=returns_centered, disp_freq=inf);
    tbfl_est.run();
    if isempty(tbfl_est.breaks)
        tbfl_breaks = [];
    else
        tbfl_breaks = sort(tbfl_est.breaks(:));
    end

    [Omega_hat, ~] = tbfl_est.EstimateOmega();
    Theta_TBFL = invert_precision_segments(Omega_hat, size(returns_centered, 1), ...
        tbfl_breaks, min_eig=0.01);

    % Append the freshly computed TBFL outputs to the main MAT file so
    % future reruns can skip this expensive recomputation.
    num_exps = get_saved_num_exps(data, exp_idx);
    TBFLcps = cell(num_exps, 1);
    TBFL_Omega = cell(num_exps, 1);
    tbfl_native_paths = cell(num_exps, 1);

    if isfield(data, 'TBFLcps') && ~isempty(data.TBFLcps)
        n_copy = min(num_exps, numel(data.TBFLcps));
        TBFLcps(1:n_copy) = data.TBFLcps(1:n_copy);
    end
    if isfield(data, 'TBFL_Omega') && ~isempty(data.TBFL_Omega)
        n_copy = min(num_exps, numel(data.TBFL_Omega));
        TBFL_Omega(1:n_copy) = data.TBFL_Omega(1:n_copy);
    end
    if isfield(data, 'tbfl_native_paths') && ~isempty(data.tbfl_native_paths)
        n_copy = min(num_exps, numel(data.tbfl_native_paths));
        tbfl_native_paths(1:n_copy) = data.tbfl_native_paths(1:n_copy);
    end

    TBFLcps{exp_idx} = tbfl_breaks;
    TBFL_Omega{exp_idx} = Omega_hat;
    tbfl_native_paths{exp_idx} = Theta_TBFL;
    save(mat_filename, 'TBFLcps', 'TBFL_Omega', 'tbfl_native_paths', '-append');
end

% =========================================================================
% Build the cross-product of detectors (BSOP, TBFL) × post-refit
% estimators (ec2, xue_ma_zou, adaptive_threshold_fspd).
% Reuse saved paths when available; recompute via post_covariance_refit
% otherwise.
function [labels, thetas, breaks] = collect_realdata_post_refits(data, ...
        settings, returns_centered, exp_idx, tbfl_breaks)
    % Pair each saved detector with each enabled post-refit estimator.

    % Determine which post-refit methods to include: prefer settings, then
    % the saved data, and fall back to the hardcoded default list.
    if isfield(settings, 'post_refit_methods') && ...
            ~isempty(settings.post_refit_methods)
        post_methods = string(settings.post_refit_methods);
    elseif isfield(data, 'post_methods')
        post_methods = string(data.post_methods);
    else
        post_methods = ["ec2", "xue_ma_zou", "adaptive_threshold_fspd"];
    end
    post_methods = post_methods(:);
    post_methods = post_methods(strlength(post_methods) > 0);

    % Build the detector × method cross-product.  Each detector provides
    % its break list; the covariance path is either looked up from saved
    % post_paths or recomputed on the fly.
    detector_names = ["BSOP"; "TBFL"];
    detector_breaks = {data.BScps{exp_idx}; tbfl_breaks};
    labels = cell(numel(detector_names) * numel(post_methods), 1);
    thetas = cell(size(labels));
    breaks = cell(size(labels));
    if isempty(post_methods)
        labels = cell(0, 1);
        thetas = cell(0, 1);
        breaks = cell(0, 1);
        return
    end

    opts_post = struct('center', true, 'min_eig', 0.01, ...
        'lambda', NaN, 'lambda_scale', 1, 'delta', 2, 'rho', 2.5, ...
        'mu', 2.5, 'maxiter', 500, 'tol', 1e-5);
    % Iterate over detectors and post-refit methods; reuse saved paths
    % when present, otherwise recompute via post_covariance_refit.
    row_idx = 0;
    for dd = 1:numel(detector_names)
        for mm = 1:numel(post_methods)
            method = post_methods(mm);
            Theta_t = [];

            if isfield(data, 'post_paths') && isfield(data, 'post_detectors') ...
                    && isfield(data, 'post_methods')
                det_idx = find(string(data.post_detectors) == detector_names(dd), 1);
                method_idx = find(string(data.post_methods) == method, 1);
                if ~isempty(det_idx) && ~isempty(method_idx)
                    Theta_t = data.post_paths{det_idx, method_idx, exp_idx};
                end
            end

            if isempty(Theta_t)
                opts_mm = opts_post;
                opts_mm.lambda = setting_lambda_for_method(settings, ...
                    post_methods, method);
                Theta_t = post_covariance_refit(returns_centered, ...
                    detector_breaks{dd}, method, opts_mm);
            end

            row_idx = row_idx + 1;
            labels{row_idx} = sprintf('%s + %s', char(detector_names(dd)), ...
                pretty_covariance_method_name(method));
            thetas{row_idx} = Theta_t;
            breaks{row_idx} = detector_breaks{dd};
        end
    end
end

% =========================================================================
% Run the GMVP backtest for every covariance path and append an
% equal-weight benchmark as the final row.  Returns annualized mean,
% volatility, and information ratio for each strategy.
function [gmvp_weights, gmvp_returns, ann_mean, ann_vol, info_ratio] = ...
        evaluate_realdata_portfolios(method_thetas, returns_centered)
    % Evaluate GMVP performance for every covariance path and append the
    % equal-weight benchmark as the final row.

    % Preallocate outputs; the last slot is reserved for equal-weight.
    num_methods = numel(method_thetas);
    gmvp_weights = cell(num_methods, 1);
    gmvp_returns = cell(num_methods, 1);
    ann_mean = zeros(num_methods + 1, 1);
    ann_vol = zeros(num_methods + 1, 1);
    info_ratio = zeros(num_methods + 1, 1);

    for mm = 1:num_methods
        [gmvp_weights{mm}, gmvp_returns{mm}, ann_mean(mm), ann_vol(mm), ...
            info_ratio(mm)] = evaluate_gmvp_strategy(method_thetas{mm}, ...
            returns_centered);
    end

    % Equal-weight benchmark: 1/p allocation, no estimation required.
    [~, p_port] = size(returns_centered);
    equal_weights = ones(p_port, 1) / p_port;
    equal_returns = returns_centered(2:end, :) * equal_weights;
    ann_mean(end) = 252 * mean(equal_returns);
    ann_vol(end) = sqrt(252) * std(equal_returns);
    info_ratio(end) = ann_mean(end) / ann_vol(end);
end

% =========================================================================
% Extract the centered returns tensor saved during estimation, handling
% 2D (single experiment), 3D, and 4D storage layouts.
function returns_centered = extract_saved_returns(data, exp_idx)
    % Prefer the exact centered returns saved during estimation so the
    % covariance paths and the plotted returns always share the same T.
    returns_centered = [];
    if ~isfield(data, 'X') || isempty(data.X)
        return
    end

    % Dispatch on tensor dimensionality to extract the slice for exp_idx.
    if ismatrix(data.X)
        returns_centered = data.X;
    elseif ndims(data.X) == 3 && size(data.X, 3) >= exp_idx
        returns_centered = data.X(:, :, exp_idx);
    elseif ndims(data.X) == 4 && size(data.X, 4) >= exp_idx
        returns_centered = data.X(:, :, 1, exp_idx);
    end
end

% =========================================================================
% Infer the number of experiment slots in the saved MAT file from the
% dimensions of data.X or the lengths of TBFL cell arrays.
function num_exps = get_saved_num_exps(data, exp_idx)
    % Infer how many experiment slots the saved MAT file expects.

    % Prefer the returns tensor dimensions when available.
    num_exps = exp_idx;
    if isfield(data, 'X') && ~isempty(data.X)
        if ismatrix(data.X)
            num_exps = 1;
        elseif ndims(data.X) == 3
            num_exps = size(data.X, 3);
        else
            num_exps = size(data.X, 4);
        end
        return
    end

    % Fall back to the length of any saved TBFL cell array.
    candidate_fields = {'TBFLcps', 'TBFL_Omega', 'tbfl_native_paths'};
    for kk = 1:numel(candidate_fields)
        field_name = candidate_fields{kk};
        if isfield(data, field_name) && ~isempty(data.(field_name))
            num_exps = numel(data.(field_name));
            return
        end
    end
end

function lambda_val = setting_lambda_for_method(settings, post_methods, method)
    % Reuse the same lambda-post convention as get_two_stage_est().

    % Missing or empty lambda_post means "let the estimator choose".
    if ~isfield(settings, 'lambda_post') || isempty(settings.lambda_post)
        lambda_val = NaN;
        return
    end

    lambda_post = settings.lambda_post;
    if ~isnumeric(lambda_post)
        error('runcode_realdata:badLambdaPost', ...
            'settings.lambda_post must be numeric.');
    end

    % A scalar applies to all methods; a vector is indexed by method name.
    if isscalar(lambda_post)
        lambda_val = lambda_post;
    else
        lambda_vec = lambda_post(:);
        mm = find(string(post_methods) == string(method), 1);
        if isempty(mm) || mm > numel(lambda_vec)
            lambda_val = NaN;
            return
        end
        lambda_val = lambda_vec(mm);
    end

    % Reject non-finite or negative values as if they were unset.
    if ~isfinite(lambda_val) || lambda_val < 0
        lambda_val = NaN;
        return
    end
end

% =========================================================================
function Theta = get_nonadaptive_theta(metric_grid, Theta_tensor, exp_idx)
    % Extract the Theta sequence at the first minimum of a 2D metric grid.
    [~, linear_idx] = min(metric_grid(:, exp_idx));
    Theta = squeeze(double(Theta_tensor(:, :, :, linear_idx, exp_idx)));
end

% =========================================================================
function Theta = get_adaptive_theta(metric_tensor, Theta_tensor, exp_idx)
    % Extract the Theta sequence at the first minimum of a 4D metric tensor.

    % Minimize over the full 4D grid and recover the subscript indices to
    % address into the Theta tensor.
    metric_slice = metric_tensor(:, :, :, :, exp_idx);
    [~, linear_idx] = min(metric_slice(:));
    [i1, i2, i3, i4] = ind2sub(size(metric_slice), linear_idx);
    Theta = squeeze(double(Theta_tensor(:, :, :, i1, i2, i3, i4, exp_idx)));
end

% =========================================================================
function breaks = get_nonadaptive_breaks(metric_grid, breaks_cell, exp_idx)
    % Extract the break vector at the first minimum of a 2D metric grid.
    [~, linear_idx] = min(metric_grid(:, exp_idx));
    breaks = breaks_cell{linear_idx, exp_idx};

    % Normalize to a row vector; empty breaks are returned as [].
    if isempty(breaks)
        breaks = [];
    else
        breaks = breaks(:)';
    end
end

% =========================================================================
function breaks = get_adaptive_breaks(metric_tensor, breaks_cell, exp_idx)
    % Extract the break vector at the first minimum of a 4D metric tensor.

    % Minimize over the full 4D grid and recover the subscript indices.
    metric_slice = metric_tensor(:, :, :, :, exp_idx);
    [~, linear_idx] = min(metric_slice(:));
    [i1, i2, i3, i4] = ind2sub(size(metric_slice), linear_idx);
    breaks = breaks_cell{i1, i2, i3, i4, exp_idx};

    % Normalize to a row vector; empty breaks are returned as [].
    if isempty(breaks)
        breaks = [];
    else
        breaks = breaks(:)';
    end
end

% =========================================================================
function write_portfolio_table(fid, labels, ann_mean, ann_vol, info_ratio)
    % Write annualized GMVP summary metrics in a compact text table.

    % Header row followed by one formatted line per method.
    fprintf(fid, '  %-20s %14s %14s %18s\n', ...
        'Method', 'AnnualMean', 'AnnualVol', 'InformationRatio');
    for idx = 1:numel(labels)
        label = labels{idx};
        if isstring(label)
            label = char(label);
        end
        fprintf(fid, '  %-20s %14.4f %14.4f %18.4f\n', ...
            label, ann_mean(idx), ann_vol(idx), info_ratio(idx));
    end
end
