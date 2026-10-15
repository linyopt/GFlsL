%% Real data (SP100 super-large panel).
%
% This script avoids utils/get_two_stage_est.m. Instead, the estimation runs
% inlined here keep only the information-criterion-winning covariance paths
% in memory and reduce them to the step norms the break diagnostics plot;
% those step norms, the break lists and the figures are written under the
% results directory, so redrawing a figure costs no refit.
%
% The full-grid Theta tensors (Thetans / Theta2s) are never materialized:
% for SP100 they would require tens to hundreds of GB.  Only the small
% metric grids and break cells are stored during the grid search; the
% winning covariance path for each criterion is recovered by re-running the
% single winning GFlsL configuration per criterion.

addpath(genpath(pwd), '-begin')
clear
clc

% Suppress the warning produced by nearlySingularMatrix comes from TBFL.
warning('off', 'MATLAB:nearlySingularMatrix')

data_dir = 'data';

% Choose the dataset and an optional global scale on returns. If the scale
% changes, the tuning grids below should be adjusted proportionally.
portfolio = "SP100";
scale = 100;
[returns_raw, dates, timeframe] = load_realdata_panel(data_dir, portfolio, scale);

% --- TEST SWITCH ---------------------------------------------------------
% Set test_subset = true to truncate the panel to a small prefix for a fast
% end-to-end smoke test. Set to false for the full run.
test_subset = false;
if test_subset
    n_keep = 250;          % number of time observations to keep
    p_keep = 20;           % number of assets to keep
    returns_raw = returns_raw(1:min(n_keep, end), 1:min(p_keep, end));
    dates = dates(1:min(n_keep, end), :);
    timeframe = timeframe(1:min(n_keep, numel(timeframe)));
    fprintf('[test_subset] truncated panel to T=%d, p=%d\n', ...
        size(returns_raw, 1), size(returns_raw, 2));
end
% ------------------------------------------------------------------------

% Center the real-data panel once up front and record the removed sample
% mean so the run settings capture the exact estimation preprocessing.
sample_mean_returns = mean(returns_raw, 1);
returns_centered = returns_raw - sample_mean_returns;

% If a short prefix is selected manually above, keep the logged timeframe
% and centering statistics aligned with the truncated sample.
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
beta_ = 0.1;
penalty_scale = [0.25, 0.5, 2];
relaxation = 1.61;

% Keep the smoke test genuinely small so it can be run on a workstation.
if test_subset
    lamb1s = p * [5e-6, 3.5e-5, 6.5e-5];
    lamb2s = p * [0.03, 0.06, 0.09];
    lamb2s2 = p * [0.004, 0.012, 0.020];
end

% Number of process workers for chain-parallel grid evaluation. Each chain
% reuses the previous candidate state while independent chains keep all
% workers busy.

% Worker count is set dynamically: the pool is capped from two sides, the
% core count (parcluster profile) and the machine RAM. Each process worker
% carries its own copy of the p-by-p-by-T solver arrays (ten double arrays
% per fit) plus temporaries and the MATLAB worker runtime, so at the SP100
% panel size one worker holds about 3 GB in total. Keep at most half of the
% total RAM for the pool, so the client and the OS are never starved.
per_worker_bytes = 10 * p * p * T * 8 + 1.5 * 2 ^ 30;
max_workers = max(1, floor(0.5 * total_ram_bytes() / per_worker_bytes));
pool_now = gcp('nocreate');

if isempty(pool_now)
    pool_profile = parcluster('Processes');
    n_workers = pool_profile.NumWorkers;
    if ~isempty(max_workers)
        n_workers = min(n_workers, max_workers);
    end
    parpool('Processes', n_workers);
else
    n_workers = pool_now.NumWorkers;
end
fprintf('Parallel pool: %d process workers\n', n_workers);
warm_start_chains = n_workers;

% Progress display for the real-data grid.
show_progress = true;

% The threshold for BSOP.
tau_BSOP = 10;

% Post-refit configuration (structure-agnostic estimators only on real data).
run_post_refits = true;
post_refit_methods = ["ec2", "xue_ma_zou", "adaptive_threshold_fspd"];
lambda_post = NaN;
diagnostic_window = 42;

% The file prefix and output directory. Smoke tests are isolated from the
% production results.
if test_subset
    results_dir = fullfile('results', 'realdata_smoke');
else
    results_dir = fullfile('results', 'realdata');
end
filepre = string(fullfile(results_dir, sprintf('%s_%g_', char(portfolio), scale)));
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Persistent cache of the per-candidate grid metrics. It stores, for every
% evaluated penalty combination, the raw criterion ingredients (loss,
% Gaussian loss, model dimension, break locations) together with the full
% information-criterion family. Adding or replacing an information
% criterion therefore only rescores these stored grids; the expensive fits
% never have to be repeated. No covariance path (Theta) is cached.
ic_cache_dir = fullfile(results_dir, sprintf('ic_cache_%s_%g', ...
    char(portfolio), scale));

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
    'run_post_refits', run_post_refits, ...
    'post_refit_methods', post_refit_methods, ...
    'lambda_post', lambda_post, ...
    'mu2', mu2, ...
    'mu1', mu1, ...
    'kappa_', kappa_, ...
    'tol', tol, ...
    'beta_', beta_, ...
    'penalty_scale', penalty_scale, ...
    'relaxation', relaxation, ...
    'parallel_workers', n_workers, ...
    'grid_schedule', "warm-chain", ...
    'warm_start_chains', warm_start_chains, ...
    'ic_cache_dir', ic_cache_dir, ...
    'diagnostic_window', diagnostic_window);

% ---------------------------------------------------------------------
% Estimation (inlined, keeps only winning covariance paths).
%
% Returns the assembled competitor covariance paths and break lists needed
% by the break diagnostics. Only the small metric grids are persisted (as
% the criterion cache), never the covariance paths.
% ---------------------------------------------------------------------
[method_labels, method_thetas, method_breaks, ...
    break_labels_raw, break_lists_raw] = ...
    run_realdata_estimation(returns_centered, lamb1s, lamb2s, lamb2s2, ...
    mu1, mu2, kappa_, tol, beta_, penalty_scale, relaxation, tau_BSOP, ...
    run_post_refits, post_refit_methods, lambda_post, show_progress, ...
    warm_start_chains, ic_cache_dir);

% ---------------------------------------------------------------------
% Break diagnostics and result logging (writes text + figures).
% ---------------------------------------------------------------------
run_realdata_diagnostics(method_labels, method_thetas, ...
    method_breaks, break_labels_raw, break_lists_raw, returns_centered, ...
    dates, settings, filepre);

% Local functions.
% =========================================================================
function [method_labels, method_thetas, method_breaks, ...
        break_labels_raw, break_lists_raw] = ...
        run_realdata_estimation(X, lamb1s, lamb2s, lamb2s2, mu1, mu2, ...
        kappa_, tol, beta_, penalty_scale, relaxation, tau_BSOP, ...
        run_post_refits, post_refit_methods, lambda_post, show_progress, ...
        warm_start_chains, ic_cache_dir)
    % Inlined two-stage estimation for a single real-data panel.
    %
    % Only the quantities required by the break diagnostics are kept:
    %   - BSOP and TBFL breaks (and TBFL-native covariance path),
    %   - the GFlsL covariance paths selected by every criterion of the
    %     information-criterion family for the non-adaptive and adaptive
    %     fits (recovered by re-running the single winning candidate per
    %     criterion),
    %   - the post-refit covariance paths for each detector x method.
    %
    % The full-grid Theta tensors are never stored; only the small metric
    % grids and break cells are materialized during the grid search. The
    % raw criterion ingredients are cached per candidate under
    % @ic_cache_dir, so a later change of criterion requires only rescoring
    % them.

    [T, p] = size(X);

    % Non-adaptive grid: lamb1s x lamb2s.
    len1 = numel(lamb1s);
    len2 = numel(lamb2s);
    num12 = len1 * len2;
    [temp1, temp2] = meshgrid(1:len1, 1:len2);
    idxcomb = [temp1(:), temp2(:)];

    % Adaptive grid: lamb1s x lamb2s2, layered on the stage-1 lamb2s grid.
    len22 = numel(lamb2s2);
    num122 = len1 * len22;
    [temp12, temp22] = meshgrid(1:len1, 1:len22);
    idxcomb2 = [temp12(:), temp22(:)];

    bb_ = T ^ (-kappa_);

    % Settings shared by every fit of this panel. Each stage attaches its
    % own grids below, because the cached candidates are keyed by absolute
    % penalty and have to stay valid when a grid changes.
    fit_cfg = fit_settings(p, T, beta_, tol, penalty_scale, relaxation, ...
        0.01, X);

    % Metric grids and break cells (small; kept for IC selection). Every
    % criterion in the information-criterion family is a scalar combination
    % of these ingredients, so caching them makes any future criterion
    % rescorable without refitting.
    lossval_ns = inf(num12, 1);
    gauss_ns = inf(num12, 1);
    dim_ns = inf(num12, 1);
    EstBreaksn = cell(num12, 1);

    fprintf('Estimation: %s | T=%d | p=%d\n', 'SP100', T, p);

    % ---------------------------------------------------------------------
    % Competing baselines: BSOP and TBFL.
    % ---------------------------------------------------------------------
    fprintf('Baselines: BSOP ... ');
    temp_BS = BS_cov(X', 0, T, tau_BSOP);
    if isempty(temp_BS.S)
        BScps = [];
    else
        BScps = sort(temp_BS.S);
    end
    fprintf('done (%d breaks).\n', numel(BScps));

    fprintf('Baselines: TBFL ... ');
    temp_TBFL = TBFL(X=X, disp_freq=inf);
    temp_TBFL.run();
    if isempty(temp_TBFL.breaks)
        tbfl_breaks = [];
    else
        tbfl_breaks = sort(temp_TBFL.breaks(:));
    end
    [Omega_hat, ~] = temp_TBFL.EstimateOmega();
    Theta_TBFL = invert_precision_segments(Omega_hat, T, tbfl_breaks, ...
        min_eig=0.01);
    clear temp_TBFL

    % ---------------------------------------------------------------------
    % Non-adaptive GFlsL grid (candidates cached by absolute penalty).
    % ---------------------------------------------------------------------
    na_config = fit_cfg;
    na_config.lamb1s = lamb1s;
    na_config.lamb2s = lamb2s;
    na_cache_file = fullfile(ic_cache_dir, 'ic_metrics_na.mat');
    [na_metrics, na_cached] = load_stage_metrics(na_cache_file, na_config);
    if na_cached
        fprintf('Non-adaptive grid: loaded cached metrics from %s.\n', ...
            na_cache_file);
    else
        if show_progress
            dq_n = parallel.pool.DataQueue;
            update_realdata_progress("start", "Non-adaptive grid", num12);
            afterEach(dq_n, @(~) update_realdata_progress("tick"));
        else
            dq_n = [];
        end
        grid_start = tic;
        candidate_file = fullfile(ic_cache_dir, 'ic_candidates_na.mat');
        store = load_candidate_store(candidate_file, fit_cfg);
        keys = candidate_keys(lamb1s, lamb2s, idxcomb, {});
        missing = find(~isfield(store, cellstr(keys)));
        if show_progress
            for qq = 1:(num12 - numel(missing))
                update_realdata_progress("tick");
            end
        end
        if ~isempty(missing)
            chains_n = build_warm_chains(missing, idxcomb, ...
                warm_start_chains);
            chain_payloads = cell(numel(chains_n), 1);
            parfor cc = 1:numel(chains_n)
                chain_payloads{cc} = run_realdata_warm_chain(chains_n{cc}, ...
                    X, lamb1s, lamb2s, idxcomb, beta_, tol, penalty_scale, ...
                    relaxation, [], [], dq_n);
            end
            fitted = unpack_realdata_chains(chains_n, chain_payloads, num12);
            clear chain_payloads
            for jj = missing(:)'
                store.(keys(jj)) = fitted{jj};
            end
            clear fitted chains_n
            save_candidate_store(candidate_file, store, fit_cfg);
        end
        for jj = 1:num12
            payload = store.(keys(jj));
            lossval_ns(jj) = payload.lossval;
            gauss_ns(jj) = payload.gauss_loss;
            dim_ns(jj) = payload.model_dim;
            EstBreaksn{jj} = payload.EstBreaks;
        end
        clear store payload keys missing dq_n
        if show_progress
            update_realdata_progress("finish", num12);
        end
        fprintf('Non-adaptive grid completed in %s.\n', ...
            duration([0, 0, toc(grid_start)]));
        na_metrics = struct();
        na_metrics.lamb1s = lamb1s;
        na_metrics.lamb2s = lamb2s;
        na_metrics.lossval = lossval_ns;
        na_metrics.gauss_loss = gauss_ns;
        na_metrics.model_dim = dim_ns;
        na_metrics.nbreak = cellfun(@numel, EstBreaksn);
        na_metrics.EstBreaks = EstBreaksn;
        save_stage_metrics(na_cache_file, na_metrics, na_config);
    end

    % The criterion family is code, not cached data: rescore on every path
    % so that a cache hit never reuses grids produced by an older formula.
    na_metrics = score_ic_family(na_metrics, p, T);

    lossval_ns = na_metrics.lossval;
    EstBreaksn = na_metrics.EstBreaks;
    crit_names = na_metrics.ic_names;
    ic_grids_ns = na_metrics.ic_grids;
    num_crit = numel(crit_names);

    % Recover only the unique criterion winners, in parallel. Every
    % criterion is a scalar combination of the recorded per-candidate
    % metrics, so the whole family costs no extra fits here. The
    % non-adaptive grid is a flat candidate vector, so its winners are
    % linear indices into it and @idxcomb decodes them to grid positions.
    na_winners = ic_family_winners(ic_grids_ns, 1);
    unique_na = unique(na_winners, 'stable');
    unique_na_thetas = cell(numel(unique_na), 1);
    if show_progress
        dq_na_recover = parallel.pool.DataQueue;
        update_realdata_progress("start", "Recover non-adaptive", ...
            numel(unique_na));
        afterEach(dq_na_recover, @(~) update_realdata_progress("tick"));
    else
        dq_na_recover = [];
    end
    parfor uu = 1:numel(unique_na)
        ii = unique_na(uu);
        unique_na_thetas{uu} = rerun_gflsl_nonadaptive(X, ...
            lamb1s(idxcomb(ii, 1)), lamb2s(idxcomb(ii, 2)), ...
            beta_, tol, penalty_scale, relaxation);
        if show_progress
            send(dq_na_recover, 1);
        end
    end
    if show_progress
        update_realdata_progress("finish", numel(unique_na));
    end
    clear dq_na_recover
    na_thetas = cell(num_crit, 1);
    na_breaks = cell(num_crit, 1);
    for cc = 1:num_crit
        ii = na_winners(cc);
        uu = find(unique_na == ii, 1);
        na_thetas{cc} = unique_na_thetas{uu};
        na_breaks{cc} = normalize_breaks(EstBreaksn{ii});
    end
    clear unique_na_thetas

    % ---------------------------------------------------------------------
    % Stage-1 + adaptive GFlsL grid (cached per candidate; stage 1 is
    % re-fit only when the grid has to be recomputed).
    % ---------------------------------------------------------------------
    ad_config = fit_cfg;
    ad_config.lamb1s = lamb1s;
    ad_config.lamb2s = lamb2s;
    ad_config.lamb2s2 = lamb2s2;
    ad_config.mu1 = mu1;
    ad_config.mu2 = mu2;
    ad_cache_file = fullfile(ic_cache_dir, 'ic_metrics_ad.mat');
    [ad_metrics, ad_cached] = load_stage_metrics(ad_cache_file, ad_config);
    if ad_cached
        fprintf('Adaptive grid: loaded cached metrics from %s.\n', ...
            ad_cache_file);
    else
        ad_metrics = compute_adaptive_grid(X, lamb1s, lamb2s, lamb2s2, ...
            mu1, mu2, bb_, beta_, tol, penalty_scale, relaxation, ...
            idxcomb2, num122, warm_start_chains, show_progress, ...
            ic_cache_dir, p, T);
        save_stage_metrics(ad_cache_file, ad_metrics, ad_config);
    end

    % The criterion family is code, not cached data: rescore on every path
    % so that a cache hit never reuses grids produced by an older formula.
    ad_metrics = score_ic_family(ad_metrics, p, T);

    EstBreaks2 = ad_metrics.EstBreaks;
    ic_grids_2s = ad_metrics.ic_grids;
    stage1_lamb2_used = ad_metrics.stage1_lamb2_used;

    % Recover only the unique adaptive winners. Re-running stage 1 for at
    % most one winner per criterion is cheap compared with retaining all
    % stage-1 paths.
    ad_winners = ic_family_winners(ic_grids_2s, 4);
    unique_ad = unique(ad_winners, 'rows', 'stable');
    unique_ad_thetas = cell(size(unique_ad, 1), 1);
    if show_progress
        dq_ad_recover = parallel.pool.DataQueue;
        update_realdata_progress("start", "Recover adaptive", ...
            size(unique_ad, 1));
        afterEach(dq_ad_recover, @(~) update_realdata_progress("tick"));
    else
        dq_ad_recover = [];
    end
    parfor uu = 1:size(unique_ad, 1)
        w = unique_ad(uu, :);
        unique_ad_thetas{uu} = rerun_gflsl_adaptive(X, ...
            lamb1s(idxcomb2(w(1), 1)), ...
            lamb2s2(idxcomb2(w(1), 2)), mu1(w(4)), mu2(w(3)), ...
            stage1_lamb2_used(w(2)), bb_, beta_, tol, ...
            penalty_scale, relaxation);
        if show_progress
            send(dq_ad_recover, 1);
        end
    end
    if show_progress
        update_realdata_progress("finish", size(unique_ad, 1));
    end
    clear dq_ad_recover

    ad_thetas = cell(num_crit, 1);
    ad_breaks = cell(num_crit, 1);
    for cc = 1:num_crit
        uu = find(all(unique_ad == ad_winners(cc, :), 2), 1);
        ad_thetas{cc} = unique_ad_thetas{uu};
        w = ad_winners(cc, :);
        ad_breaks{cc} = normalize_breaks( ...
            EstBreaks2{w(1), w(2), w(3), w(4)});
    end
    clear unique_ad_thetas

    % ---------------------------------------------------------------------
    % Post-refit covariance paths (BSOP and TBFL detectors).
    % ---------------------------------------------------------------------
    post_methods = string(post_refit_methods);
    post_methods = post_methods(:);
    post_methods = post_methods(strlength(post_methods) > 0);
    detector_names = ["BSOP"; "TBFL"];
    detector_breaks = {BScps; tbfl_breaks};

    if run_post_refits && ~isempty(post_methods)
        num_det = numel(detector_names);
        num_met = numel(post_methods);
        post_labels = cell(num_det * num_met, 1);
        post_thetas = cell(num_det * num_met, 1);
        post_breaks = cell(num_det * num_met, 1);

        opts_post = struct('center', true, 'min_eig', 0.01, ...
            'lambda', NaN, 'lambda_scale', 1, 'delta', 2, 'rho', 2.5, ...
            'mu', 2.5, 'maxiter', 500, 'tol', 1e-5);
        if show_progress
            dq_post = parallel.pool.DataQueue;
            update_realdata_progress("start", "Post-refit covariance", ...
                num_det * num_met);
            afterEach(dq_post, @(~) update_realdata_progress("tick"));
        else
            dq_post = [];
        end
        row_idx = 0;
        for dd = 1:num_det
            for mm = 1:num_met
                if show_progress
                    update_realdata_progress("detail", sprintf( ...
                        "%s + %s", char(detector_names(dd)), ...
                        pretty_covariance_method_name(post_methods(mm))));
                end
                opts_mm = opts_post;
                opts_mm.lambda = resolve_lambda_post(lambda_post, ...
                    post_methods, post_methods(mm));
                Theta_t = post_covariance_refit(X, detector_breaks{dd}, ...
                    post_methods(mm), opts_mm);
                row_idx = row_idx + 1;
                post_labels{row_idx} = sprintf('%s + %s', ...
                    char(detector_names(dd)), ...
                    pretty_covariance_method_name(post_methods(mm)));
                post_thetas{row_idx} = Theta_t;
                post_breaks{row_idx} = detector_breaks{dd};
                if show_progress
                    send(dq_post, 1);
                end
            end
        end
        if show_progress
            update_realdata_progress("finish", num_det * num_met);
        end
        clear dq_post
    else
        post_labels = cell(0, 1);
        post_thetas = cell(0, 1);
        post_breaks = cell(0, 1);
    end

    % ---------------------------------------------------------------------
    % Assemble the method ordering used by the break diagnostics.
    % ---------------------------------------------------------------------
    % The shipped three criteria stay in their original row order so the
    % existing log and figures remain comparable; the remaining criteria of
    % the family are appended after them.
    gflsl_labels = cell(2 * num_crit, 1);
    gflsl_break_labels = cell(2 * num_crit, 1);
    for cc = 1:3
        gflsl_labels{cc} = sprintf('Non-adaptive %s', crit_names{cc});
        gflsl_break_labels{cc} = sprintf('GFlsL NA %s', crit_names{cc});
        gflsl_labels{cc + 3} = sprintf('Adaptive %s', crit_names{cc});
        gflsl_break_labels{cc + 3} = sprintf('GFlsL AD %s', crit_names{cc});
    end
    for cc = 4:num_crit
        gflsl_labels{cc + 3} = sprintf('Non-adaptive %s', crit_names{cc});
        gflsl_break_labels{cc + 3} = sprintf('GFlsL NA %s', crit_names{cc});
        gflsl_labels{cc + num_crit} = ...
            sprintf('Adaptive %s', crit_names{cc});
        gflsl_break_labels{cc + num_crit} = ...
            sprintf('GFlsL AD %s', crit_names{cc});
    end
    gflsl_thetas = [na_thetas(1:3); ad_thetas(1:3); ...
        na_thetas(4:num_crit); ad_thetas(4:num_crit)];
    gflsl_breaks = [na_breaks(1:3); ad_breaks(1:3); ...
        na_breaks(4:num_crit); ad_breaks(4:num_crit)];

    method_labels = [gflsl_labels; {'TBFL-native'}; post_labels];
    method_thetas = [gflsl_thetas; {Theta_TBFL}; post_thetas];
    method_breaks = [gflsl_breaks; {tbfl_breaks}; post_breaks];

    baseline_break_labels = {'BSOP'; 'TBFL'};
    baseline_breaks = {BScps; tbfl_breaks};
    break_labels_raw = [gflsl_break_labels; baseline_break_labels];
    break_lists_raw = [gflsl_breaks; baseline_breaks];

    fprintf('Estimation complete.\n');
end

% =========================================================================
function payloads = unpack_realdata_chains(chains, chain_payloads, n_candidates)
    payloads = cell(n_candidates, 1);
    for cc = 1:numel(chains)
        for ee = 1:numel(chains{cc})
            payloads{chains{cc}(ee)} = chain_payloads{cc}{ee};
        end
    end
end

% =========================================================================
function ad_metrics = compute_adaptive_grid(X, lamb1s, lamb2s, lamb2s2, ...
        mu1, mu2, bb_, beta_, tol, penalty_scale, relaxation, idxcomb2, ...
        num122, warm_start_chains, show_progress, ic_cache_dir, p, T)
    % Fit the stage-1 + adaptive GFlsL grid and package the metric grids.
    % Stage 1 is re-fit here because its adaptive weights are needed by
    % every second-stage candidate.

    len2 = numel(lamb2s);
    lenmu1 = numel(mu1);
    lenmu2 = numel(mu2);

    lossval_2s = inf(num122, len2, lenmu2, lenmu1);
    gauss_2s = inf(num122, len2, lenmu2, lenmu1);
    dim_2s = inf(num122, len2, lenmu2, lenmu1);
    EstBreaks2 = cell(num122, len2, lenmu2, lenmu1);
    stage1_lamb2_used = nan(len2, 1);

    fit_cfg = fit_settings(p, T, beta_, tol, penalty_scale, relaxation, ...
        0.01, X);
    candidate_file = fullfile(ic_cache_dir, 'ic_candidates_ad.mat');
    store = load_candidate_store(candidate_file, fit_cfg);

    adaptive_total = len2 * (1 + num122 * lenmu1 * lenmu2);
    if show_progress
        dq_a = parallel.pool.DataQueue;
        update_realdata_progress("start", "Adaptive grid", adaptive_total);
        afterEach(dq_a, @(~) update_realdata_progress("tick"));
    else
        dq_a = [];
    end
    adaptive_start = tic;
    stopped_early = false;

    for k = 1:len2
        lamb2_k = lamb2s(k);
        if show_progress
            update_realdata_progress("detail", sprintf( ...
                "stage 1 %d/%d, lambda2=%.4g", k, len2, lamb2_k));
        end
        Est1 = make_realdata_estimator(X, 0, lamb2_k, beta_, tol, ...
            penalty_scale, relaxation);
        Est1.run();
        stage1_breaks = Est1.EstBreaks;

        if isempty(stage1_breaks)
            if k == 1
                % Search a smaller lamb2 to recover at least one break so
                % the adaptive weights are not degenerate.
                if len2 > 1
                    step_size = abs(lamb2s(2) - lamb2s(1));
                else
                    step_size = abs(lamb2s(1)) * 0.1;
                end
                step_size = max(step_size, 1e-6);
                % lamb2_fit tracks the lambda of the fit that produced the
                % current stage-1 solution, so the recorded stage-1 lambda
                % matches the weights even when no fallback finds a break.
                lamb2_fit = lamb2_k;
                lamb2_temp = lamb2_k - step_size;
                iter = 0;
                while lamb2_temp > 0 && iter < 1000
                    if show_progress
                        update_realdata_progress("detail", sprintf( ...
                            "stage 1 fallback, lambda2=%.4g", lamb2_temp));
                    end
                    lamb2_fit = lamb2_temp;
                    Est1 = make_realdata_estimator(X, 0, lamb2_temp, ...
                        beta_, tol, penalty_scale, relaxation);
                    Est1.run();
                    if ~isempty(Est1.EstBreaks)
                        break
                    end
                    lamb2_temp = lamb2_temp - step_size;
                    iter = iter + 1;
                end
                stage1_lamb2_used(k) = lamb2_fit;
            else
                stopped_early = true;
                if show_progress
                    update_realdata_progress("tick");
                end
                break
            end
        else
            stage1_lamb2_used(k) = lamb2_k;
        end

        if show_progress
            update_realdata_progress("tick");
        end
        stage1_Theta = double(Est1.Theta_k);
        stage1_norm_diff = Est1.norm_diff;

        for idx_mu2 = 1:lenmu2
            for idx_mu1 = 1:lenmu1
                xi1 = max(abs(stage1_Theta), bb_) .^ -mu1(idx_mu1);
                xi2 = max(stage1_norm_diff, bb_) .^ -mu2(idx_mu2);
                if show_progress
                    update_realdata_progress("detail", sprintf( ...
                        "stage 2 %d/%d, mu1=%.2g, mu2=%.2g", ...
                        k, len2, mu1(idx_mu1), mu2(idx_mu2)));
                end

                keys = candidate_keys(lamb1s, lamb2s2, idxcomb2, ...
                    {mu1(idx_mu1), mu2(idx_mu2), stage1_lamb2_used(k)});
                missing = find(~isfield(store, cellstr(keys)));
                if show_progress
                    for qq = 1:(num122 - numel(missing))
                        update_realdata_progress("tick");
                    end
                end
                if ~isempty(missing)
                    chains_a = build_warm_chains(missing, idxcomb2, ...
                        warm_start_chains);
                    chain_payloads = cell(numel(chains_a), 1);
                    parfor cc = 1:numel(chains_a)
                        chain_payloads{cc} = run_realdata_warm_chain( ...
                            chains_a{cc}, X, lamb1s, lamb2s2, idxcomb2, ...
                            beta_, tol, penalty_scale, relaxation, ...
                            xi1, xi2, dq_a);
                    end
                    fitted = unpack_realdata_chains(chains_a, ...
                        chain_payloads, num122);
                    clear chain_payloads
                    for jj = missing(:)'
                        store.(keys(jj)) = fitted{jj};
                    end
                    clear fitted chains_a
                    save_candidate_store(candidate_file, store, fit_cfg);
                end
                for jj = 1:num122
                    payload = store.(keys(jj));
                    lossval_2s(jj, k, idx_mu2, idx_mu1) = payload.lossval;
                    gauss_2s(jj, k, idx_mu2, idx_mu1) = payload.gauss_loss;
                    dim_2s(jj, k, idx_mu2, idx_mu1) = payload.model_dim;
                    EstBreaks2{jj, k, idx_mu2, idx_mu1} = payload.EstBreaks;
                end
                clear payload keys missing
            end
        end
        clear stage1_Theta stage1_norm_diff xi1 xi2 Est1
    end

    if show_progress
        if stopped_early
            update_realdata_progress("detail", ...
                "stopped after an empty stage-1 fit");
        end
        update_realdata_progress("finish");
    end
    clear dq_a store
    fprintf('Adaptive grid completed in %s.\n', ...
        duration([0, 0, toc(adaptive_start)]));

    ad_metrics = struct();
    ad_metrics.lamb1s = lamb1s;
    ad_metrics.lamb2s = lamb2s;
    ad_metrics.lamb2s2 = lamb2s2;
    ad_metrics.mu1 = mu1;
    ad_metrics.mu2 = mu2;
    ad_metrics.stage1_lamb2_used = stage1_lamb2_used;
    ad_metrics.lossval = lossval_2s;
    ad_metrics.gauss_loss = gauss_2s;
    ad_metrics.model_dim = dim_2s;
    ad_metrics.nbreak = cellfun(@numel, EstBreaks2);
    ad_metrics.EstBreaks = EstBreaks2;
end

% =========================================================================
function payloads = run_realdata_warm_chain(chain, X, lamb1_grid, ...
        lamb2_grid, idxcomb, beta_, tol, penalty_scale, relaxation, ...
        xi1, xi2, progress_queue)
    % Fit one warm-start chain and return one payload per candidate.
    %
    % The chain is always fitted from its cold-start head, so which
    % candidates an earlier run happened to store cannot change the fitted
    % path of the candidates still missing.

    adaptive = ~isempty(xi1);
    payloads = cell(numel(chain), 1);

    Est = [];
    for ee = 1:numel(chain)
        jj = chain(ee);
        if adaptive
            lamb1 = lamb1_grid(idxcomb(jj, 1)) .* xi1;
            lamb2 = lamb2_grid(idxcomb(jj, 2)) .* reshape(xi2, 1, []);
        else
            lamb1 = lamb1_grid(idxcomb(jj, 1));
            lamb2 = lamb2_grid(idxcomb(jj, 2));
        end
        if isempty(Est)
            Est = make_realdata_estimator(X, lamb1, lamb2, beta_, tol, ...
                penalty_scale, relaxation);
        else
            update_realdata_penalties(Est, lamb1, lamb2);
        end
        Est.run();
        payloads{ee} = realdata_metric_payload(Est);
        if ~isempty(progress_queue)
            send(progress_queue, 1);
        end
    end
end

% =========================================================================
function update_realdata_penalties(Est, lamb1, lamb2)
    % Update candidate-specific penalties while preserving the warm state.
    if isscalar(lamb1)
        Est.lamb1 = lamb1 * ones(Est.p, Est.p, Est.T);
    else
        lamb1_sym = (lamb1 + permute(lamb1, [2, 1, 3])) / 2;
        if max(vecnorm(reshape(lamb1 - permute(lamb1, [2, 1, 3]), ...
                [], Est.T))) > 1e-5
            error('runcode_realdata:lamb1Symmetry', ...
                'The adaptive l1 penalty must be symmetric.');
        end
        Est.lamb1 = lamb1_sym;
        Est.lamb1(Est.idx) = 0;
    end
    if isscalar(lamb2)
        Est.lamb2 = lamb2 * ones(1, Est.T - 1);
    else
        Est.lamb2 = reshape(lamb2, 1, Est.T - 1);
    end
end

% =========================================================================
function Est = make_realdata_estimator(X, lamb1, lamb2, beta_, tol, ...
        penalty_scale, relaxation)
    Est = GFlsL(X=X, lamb1=lamb1, lamb2=lamb2, beta_=beta_, ...
        penalty_scale=penalty_scale, relaxation=relaxation, ...
        epsilon=0.01, tol=tol, maxiter=inf, disp_freq=inf, ...
        merge_breaks=false, termination_check_freq=20);
end

% =========================================================================
function payload = realdata_metric_payload(Est)
    % Every criterion in the information-criterion family is a scalar
    % combination of these quantities, so record them once and rescore any
    % penalty pair afterwards without refitting anything.
    payload = struct('AIC', Est.AIC, 'BIC', Est.BIC, 'HBIC', Est.HBIC, ...
        'HBICG', Est.HBICG, 'lossval', Est.lossval, ...
        'gauss_loss', Est.gauss_loss, 'model_dim', Est.model_dim, ...
        'EstBreaks', normalize_breaks(Est.EstBreaks));
end

% =========================================================================
function Theta = rerun_gflsl_nonadaptive(X, lamb1, lamb2, beta_, tol, ...
        penalty_scale, relaxation)
    % Re-run one selected configuration to recover its covariance path.
    Est = make_realdata_estimator(X, lamb1, lamb2, beta_, tol, ...
        penalty_scale, relaxation);
    Est.run();
    Theta = single(Est.Theta_k);
end

% =========================================================================
function Theta = rerun_gflsl_adaptive(X, lamb1_scalar, lamb2s2_scalar, ...
        mu1_v, mu2_v, stage1_lamb2, bb_, beta_, tol, penalty_scale, ...
        relaxation)
    % Rebuild the selected adaptive weights and recover the covariance path.
    Est1 = make_realdata_estimator(X, 0, stage1_lamb2, beta_, tol, ...
        penalty_scale, relaxation);
    Est1.run();
    xi1 = max(abs(double(Est1.Theta_k)), bb_) .^ -mu1_v;
    xi2 = max(Est1.norm_diff, bb_) .^ -mu2_v;
    Est2 = make_realdata_estimator(X, lamb1_scalar .* xi1, ...
        lamb2s2_scalar .* reshape(xi2, 1, []), beta_, tol, ...
        penalty_scale, relaxation);
    Est2.run();
    Theta = single(Est2.Theta_k);
end

% =========================================================================
function update_realdata_progress(command, varargin)
    % Render one concise progress line for client-side DataQueue callbacks.
    persistent label total completed started detail last_len last_render spinner_idx

    switch string(command)
        case "start"
            label = char(varargin{1});
            total = varargin{2};
            completed = 0;
            started = tic;
            detail = '';
            last_len = 0;
            last_render = -inf;
            spinner_idx = 1;
            render(false, true);
        case "tick"
            completed = min(completed + 1, total);
            render(false, completed == total);
        case "detail"
            detail = char(varargin{1});
            render(false, true);
        case "finish"
            if ~isempty(varargin)
                completed = min(varargin{1}, total);
            end
            render(true, true);
            label = [];
        otherwise
            error('runcode_realdata:progressCommand', ...
                'Unknown progress command: %s', command);
    end

    function render(done, force)
        if isempty(label)
            return
        end
        elapsed = toc(started);
        if ~force && elapsed - last_render < 0.2
            return
        end
        last_render = elapsed;
        fraction = min(completed / max(total, 1), 1);
        width = 20;
        filled = floor(width * fraction);
        bar = [repmat('█', 1, filled), repmat('░', 1, width - filled)];
        if completed > 0 && completed < total
            eta = elapsed * (total - completed) / completed;
            eta_text = format_elapsed(eta);
        elseif completed >= total
            eta_text = '00:00:00';
        else
            eta_text = '--:--:--';
        end
        if done && completed < total
            state = 'stopped';
        elseif done
            state = 'done';
        else
            state = sprintf('ETA %s', eta_text);
        end
        spinner = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏';
        msg = sprintf('  %-20s │%s│ %5.1f%% %s %4d/%-4d %s %-14s', ...
            label, bar, 100 * fraction, spinner(spinner_idx), completed, total, ...
            format_elapsed(elapsed), state);
        spinner_idx = mod(spinner_idx, numel(spinner)) + 1;
        if ~isempty(detail)
            msg = sprintf('%s | %s', msg, detail);
        end
        if last_len > 0
            fprintf(1, repmat('\b', 1, last_len));
        end
        fprintf(1, '%s', msg);
        last_len = strlength(msg);
        if done
            fprintf(1, '\n');
            last_len = 0;
        end
    end
end

% =========================================================================
function text = format_elapsed(seconds)
    seconds = max(0, round(seconds));
    hours = floor(seconds / 3600);
    minutes = floor(mod(seconds, 3600) / 60);
    seconds = mod(seconds, 60);
    text = sprintf('%02d:%02d:%02d', hours, minutes, seconds);
end

% =========================================================================
function breaks = normalize_breaks(breaks)
    if isempty(breaks)
        breaks = [];
    else
        breaks = breaks(:)';
    end
end

% =========================================================================
function metrics = score_ic_family(metrics, p, T)
    % Attach the full information-criterion family to a raw metric package.
    [names, grids] = ic_family_criteria(metrics.lossval, ...
        metrics.gauss_loss, metrics.model_dim, metrics.nbreak, p, T);
    metrics.ic_names = names;
    metrics.ic_grids = grids;
end

% =========================================================================
function [metrics, ok] = load_stage_metrics(path, config)
    % Load a cached metric package, provided the run configuration matches.
    metrics = [];
    ok = false;
    if exist(path, 'file') ~= 2
        return
    end
    try
        loaded = load(path, 'metrics', 'config');
    catch
        warning('runcode_realdata:cacheUnreadable', ...
            'Ignoring unreadable cached metrics: %s', path);
        return
    end
    if ~isfield(loaded, 'metrics') || ~isfield(loaded, 'config')
        return
    end
    if ~isequaln(loaded.config, config)
        warning('runcode_realdata:cacheConfigMismatch', ...
            'Ignoring cached metrics with a different configuration: %s', ...
            path);
        return
    end
    metrics = loaded.metrics;
    ok = true;
end

% =========================================================================
function fp = data_fingerprint(X)
    % Cheap panel fingerprint; guards the metric cache against a swapped
    % dataset that happens to share the same dimensions and grid.
    X = double(X);
    fp = [numel(X), sum(X(:)), sum(X(:) .^ 2), max(abs(X(:)))];
end

% =========================================================================
function save_stage_metrics(path, metrics, config)
    % Persist a metric package and the configuration that produced it.
    parent = fileparts(path);
    if exist(parent, 'dir') ~= 7
        mkdir(parent);
    end
    save(path, 'metrics', 'config', '-v7.3');
end

% =========================================================================
function keys = candidate_keys(lamb1_grid, lamb2_grid, idxcomb, extra)
    % Cache key of every candidate of a grid, aligned with the @idxcomb rows.
    n = size(idxcomb, 1);
    keys = strings(n, 1);
    for jj = 1:n
        keys(jj) = candidate_key(lamb1_grid(idxcomb(jj, 1)), ...
            lamb2_grid(idxcomb(jj, 2)), extra);
    end
end

% =========================================================================
function key = candidate_key(lamb1, lamb2, extra)
    % Cache key of one candidate: its absolute penalties, plus the adaptive
    % exponents it was weighted by, since those too change the fit.
    key = "l1_" + lambda_key(lamb1) + "_l2_" + lambda_key(lamb2);
    for ii = 1:numel(extra)
        key = key + "_x" + ii + "_" + lambda_key(extra{ii});
    end
end

% =========================================================================
function cfg = fit_settings(p, T, beta_, tol, penalty_scale, relaxation, ...
        epsilon, X)
    % Settings that fix what a single fit produces. The penalty grids are
    % deliberately absent: candidates are cached by absolute penalty, so
    % extending a grid has to reuse the fits that are already stored.
    cfg = struct('T', T, 'p', p, 'beta_', beta_, 'tol', tol, ...
        'penalty_scale', penalty_scale, 'relaxation', relaxation, ...
        'epsilon', epsilon, 'data_fingerprint', data_fingerprint(X));
end

% =========================================================================
function store = load_candidate_store(path, config)
    % Payload of every candidate already fitted, keyed by its penalties.
    store = struct();
    if exist(path, 'file') ~= 2
        return
    end
    try
        loaded = load(path, 'store', 'config');
    catch
        warning('runcode_realdata:cacheUnreadable', ...
            'Ignoring unreadable cached candidates: %s', path);
        return
    end
    if ~isfield(loaded, 'store') || ~isfield(loaded, 'config')
        return
    end
    if ~isequaln(loaded.config, config)
        warning('runcode_realdata:cacheConfigMismatch', ...
            'Ignoring cached candidates fitted with other settings: %s', ...
            path);
        return
    end
    store = loaded.store;
end

% =========================================================================
function save_candidate_store(path, store, config)
    % Rewrite the whole store; it is only a few MB even for the full grid.
    parent = fileparts(path);
    if exist(parent, 'dir') ~= 7
        mkdir(parent);
    end
    save(path, 'store', 'config', '-v7.3');
end

% =========================================================================
function lambda_val = resolve_lambda_post(lambda_post, post_methods, method)
    % Resolve one lambda value for the current post-refit method.
    if isempty(lambda_post) || ~isnumeric(lambda_post)
        lambda_val = NaN;
        return
    end
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
    if ~isfinite(lambda_val) || lambda_val < 0
        lambda_val = NaN;
    end
end

% =========================================================================
function run_realdata_diagnostics(method_labels, method_thetas, ...
        method_breaks, break_labels_raw, break_lists_raw, returns_centered, ...
        dates, settings, filepre)
    % Write the run settings, the break lists and the break-diagnostic
    % figures, and cache the figure inputs for a later redraw.

    output_dir = char(fileparts(filepre));
    file_stub = sprintf('%s_%g', char(settings.portfolio), settings.scale);

    % The diagnostics only ever use the Frobenius norm of consecutive
    % covariance differences, so every path is reduced to its step norms
    % before it reaches a figure or the cache.
    step_norms = realdata_step_norms(method_thetas, size(returns_centered, 1));

    % Cache the figure inputs, so a redraw costs a load instead of a refit.
    plot_cache_file = fullfile(output_dir, ['plot_data_', file_stub, '.mat']);
    plot_data = struct( ...
        'portfolio', settings.portfolio, ...
        'scale', settings.scale, ...
        'window', settings.diagnostic_window, ...
        'method_labels', {method_labels}, ...
        'method_breaks', {method_breaks}, ...
        'step_norms', step_norms, ...
        'returns_centered', returns_centered, ...
        'dates', dates);
    save(plot_cache_file, 'plot_data', '-v7.3');

    % Open (or create) the append-only results log and guarantee it is
    % closed on exit, even if an error is thrown below.
    log_filename = fullfile(output_dir, 'realdata_results.txt');
    log_fid = fopen(log_filename, 'a');
    if log_fid < 0
        error('runcode_realdata:openLogFailed', ...
            'Cannot open log file: %s', log_filename);
    end
    cleanup_log = onCleanup(@() fclose(log_fid));

    fprintf(log_fid, "\n======== Dataset: %s ========\n", ...
        char(settings.portfolio));
    write_settings_block(log_fid, settings);

    write_break_lists(log_fid, 'Break locations', ...
        break_labels_raw, break_lists_raw, dates);

    % Break-diagnostic figures: one figure per reported method.
    diagnostic_dir = fullfile(output_dir, 'diagnostics', file_stub);
    plot_realdata_break_diagnostics(returns_centered, dates, ...
        method_labels, method_breaks, step_norms, ...
        window=settings.diagnostic_window, ...
        out_dir=fullfile(diagnostic_dir, 'per_method'), ...
        file_prefix=file_stub);
    fprintf(log_fid, '\nBreak-diagnostic figures saved to: %s\n', ...
        diagnostic_dir);
    fprintf(log_fid, 'Diagnostic plot cache saved to: %s\n', plot_cache_file);
end

% =========================================================================
function bytes = total_ram_bytes()
    % Total physical RAM of the host machine, in bytes.  MATLAB's memory()
    % only works on Windows, so macOS and Linux are queried via sysctl and
    % /proc/meminfo respectively.


    if ispc
        [~, sysmem] = memory;
        bytes = sysmem.PhysicalMemory.Total;
    elseif ismac
        [~, out] = system('sysctl -n hw.memsize');
        bytes = str2double(strtrim(out));
    else
        fid = fopen('/proc/meminfo', 'r');
        if fid < 0
            error('runcode_realdata:meminfo', 'Cannot open /proc/meminfo.');
        end
        cleanup = onCleanup(@() fclose(fid));
        total_kb = NaN;
        while ~feof(fid)
            line = fgetl(fid);
            if ~ischar(line) || isempty(line)
                continue
            end
            if startsWith(strtrim(line), 'MemTotal:')
                parts = strsplit(line);
                total_kb = str2double(parts{2});
                break
            end
        end
        if isnan(total_kb)
            error('runcode_realdata:meminfo', ...
                'Cannot parse MemTotal from /proc/meminfo.');
        end
        bytes = total_kb * 1024;
    end
end
