% Sensitivity analysis for adaptive second-stage exponents (mu1, mu2).
% This script evaluates how different pairs of (mu1, mu2) affect adaptive
% GFlsL performance when kappa is fixed. Second-stage models are selected
% by HFE, lossval, BIC, HBIC, and HBICG. Results are averaged over
% num_exps experiments.

addpath(genpath(pwd));
clear;
clc;
close all;

rng(42, 'twister');
showbar = 1;

% Shared experiment settings
T = 200;
p = 10;
num_exps = 100;
Nsim = 10;
prob = 0.8;
num_breaks_list = [1, 3];

% Hyperparameters
tol = 1e-3;
beta_ = 0.1;
epsilon = 0.01;

% Fixed adaptive-threshold exponent
kappa_ = 0.5;
bb_ = T ^ (-kappa_);

% Sensitivity grid for adaptive exponents
mu1s = [0.5, 0.8, 1.5];
mu2s = [0.5, 0.8, 1.5];
lenmu1 = length(mu1s);
lenmu2 = length(mu2s);

% Selection criteria for second-stage model selection
IC_names = {'HFE', 'lossval', 'BIC', 'HBIC', 'HBICG'};
lenIC = length(IC_names);

% Metric names in storage order
metric_names = {'nbreaks', 'HD', 'F1', 'acc', 'error'};
num_metrics = length(metric_names);

% Helper to find the index of the first minimum element in a given
% array (reuse from get_two_stage_est.m).
findMinIn = @(x) ind2sub(size(x), find(x == min(x, [], 'all'), 1));

% Use the same shared simulation grid as runcode_simu.m for p = 10.
lamb1s = p * (2.0e-4:5.0e-4:3.2e-3);
lamb2s = p * (0.12:0.08:0.44);
lamb2s2 = p * (0.02:0.07:0.58);

len1 = length(lamb1s);
len2 = length(lamb2s);
len22 = length(lamb2s2);

% Index map for second-stage (lamb1, lamb2) pairs
[temp12, temp22] = meshgrid(1:len1, 1:len22);
idxcomb2 = [temp12(:), temp22(:)];
num122 = size(idxcomb2, 1);
total = len2 + lenmu1 * lenmu2 * num122;

% Run separate sensitivity experiments for each true break-count case
for mi = 1:length(num_breaks_list)
    num_breaks = num_breaks_list(mi);

    % Ensure output directory exists
    case_dir = fullfile('results', 'sensitivity', sprintf('case_m%d', num_breaks));
    if ~exist(case_dir, 'dir')
        mkdir(case_dir);
    end

    % Open log file for summaries
    log_path = fullfile(case_dir, sprintf('sensitivity_mu_pairs_results_m%d.txt', ...
                                          num_breaks));
    fid = fopen(log_path, 'w');
    if fid == -1
        error('Cannot open log file: %s', log_path);
    end

    fprintf(fid, 'Sensitivity Analysis for (mu1, mu2)\n');
    fprintf(fid, '====================================\n\n');
    fprintf(fid, 'Settings:\n');
    fprintf(fid, '  T = %d, p = %d, num_breaks = %d\n', T, p, num_breaks);
    fprintf(fid, '  num_exps = %d, Nsim = %d, prob = %.1f\n', num_exps, Nsim, prob);
    fprintf(fid, '  kappa = %.1f, bb = T^{-kappa}\n', kappa_);
    fprintf(fid, '  mu1 candidates: [%s]\n', sprintf('%.1f ', mu1s));
    fprintf(fid, '  mu2 candidates: [%s]\n', sprintf('%.1f ', mu2s));
    fprintf(fid, '  second-stage criteria: %s\n\n', strjoin(IC_names, ', '));

    % One summary struct per DGP
    results_by_dgp = struct('DGP', {}, 'p', {}, 'num_breaks', {}, 'kappa', {}, ...
                            'mu1s', {}, 'mu2s', {}, 'lamb1s', {}, 'lamb2s', {}, ...
                            'lamb2s2', {}, 'IC_names', {}, 'metric_names', {}, ...
                            'results_all', {}, 'results_mean', {}, 'results_std', {}, ...
                            'stage1_lamb2_selected', {}, 'stage1_nbreaks_selected', {});

    dgp_idx = 0;
    for DGP = ["banded", "iid", "factor"]
        dgp_idx = dgp_idx + 1;
        dgp_name = char(DGP);

        fprintf(fid, '\n%s\n', repmat('=', 1, 76));
        fprintf(fid, 'DGP=%s | p=%d | m=%d | T=%d | num_exps=%d\n', dgp_name, p, ...
                num_breaks, T, num_exps);
        fprintf(fid, '%s\n', repmat('=', 1, 76));

        % Use the same single-line progress bar style as the main
        % simulation runner so each replication shows stage-1 and
        % stage-2 fitting progress together.
        dq = [];
        if showbar
            dq = parallel.pool.DataQueue;
            afterEach(dq, @(~) updateProgress(total));
        end

        % Storage: (mu1, mu2, method, metric, experiment)
        results_all = zeros(lenmu1, lenmu2, lenIC, num_metrics, num_exps);

        % Storage for selected first-stage models
        stage1_lamb2_selected = zeros(num_exps, 1);
        stage1_nbreaks_selected = zeros(num_exps, 1);

        % Main replication loop
        for tt = 1:num_exps
            if showbar
                fprintf('\n  Sensitivity %s m=%d exp %02d/%02d ▶ ', ...
                    dgp_name, num_breaks, tt, num_exps);
                updateProgress(total, true);
            end

            % Generate one synthetic replication from the selected DGP
            switch string(DGP)
                case "banded"
                    [X, X_test, trueTheta, trueBreaks] = DGP_banded(T, p, num_breaks, Nsim);
                case "iid"
                    [X, X_test, trueTheta, trueBreaks] = DGP_iid(T, p, num_breaks, prob, Nsim);
                case "factor"
                    [X, X_test, trueTheta, trueBreaks] = DGP_factor(T, p, num_breaks, prob, Nsim);
            end

            % Run first-stage GFlsL over lambda candidates
            stage1_estimators = cell(len2, 1);
            stage1_break_counts = zeros(len2, 1);
            for k = 1:len2
                Est1 = GFlsL(X=X, lamb1=0, lamb2=lamb2s(k), beta_=beta_, ...
                             epsilon=epsilon, tol=tol, maxiter=inf, disp_freq=inf, ...
                             showfig=0, merge_breaks=false);
                Est1.run();

                stage1_break_counts(k) = length(Est1.EstBreaks());
                stage1_estimators{k} = Est1;
                if showbar
                    updateProgress(total);
                end
            end

            % Select first-stage estimator by break-count rule:
            % prefer nbreaks >= true, among them pick smallest,
            % fallback to closest if none qualify.
            candidates = find(stage1_break_counts >= num_breaks);
            if isempty(candidates)
                [~, idx_s1] = min(abs(stage1_break_counts - num_breaks));
            else
                [~, rel_idx] = min(stage1_break_counts(candidates));
                idx_s1 = candidates(rel_idx);
            end

            % Selected first-stage estimator.
            refEst = stage1_estimators{idx_s1};
            % Selected first-stage lambda and estimated breaks.
            stage1_lamb2_selected(tt) = lamb2s(idx_s1);
            stage1_nbreaks_selected(tt) = stage1_break_counts(idx_s1);

            % Evaluate all (mu1, mu2) pairs using the selected first-stage fit.
            for idx_mu1 = 1:lenmu1
                % Adaptive sparsity weights
                xi1_ = max(abs(refEst.Theta_k), bb_) .^ (-mu1s(idx_mu1));

                for idx_mu2 = 1:lenmu2
                    % Adaptive fusion weights
                    xi2_ = max(refEst.norm_diff, bb_) .^ (-mu2s(idx_mu2));
                    xi2_ = reshape(xi2_, 1, []);

                    % Second-stage score arrays
                    loss2s = inf(num122, 1);
                    BIC2s = inf(num122, 1);
                    HBIC_2s = inf(num122, 1);
                    HBICG_2s = inf(num122, 1);

                    HDs2 = inf(num122, 1);
                    F1s2 = inf(num122, 1);
                    accs2 = inf(num122, 1);
                    errors2 = inf(num122, 1);
                    EstBreaks2 = cell(num122, 1);

                    % Run second-stage grid for this (mu1, mu2) pair
                    parfor ii = 1:num122
                        i1 = idxcomb2(ii, 1);
                        i2 = idxcomb2(ii, 2);

                        Est2 = GFlsL(X=X, lamb1=lamb1s(i1) .* xi1_, ...
                            lamb2=lamb2s2(i2) .* xi2_, beta_=beta_, ...
                            epsilon=epsilon, tol=tol, maxiter=inf, ...
                            disp_freq=inf, showfig=0, merge_breaks=false);
                        Est2.run();

                        loss2s(ii) = Est2.lossfunc(X_test);
                        BIC2s(ii) = Est2.BIC;
                        HBIC_2s(ii) = Est2.HBIC;
                        HBICG_2s(ii) = Est2.HBICG;

                        [HD_tmp, F1_tmp, acc_tmp, error_tmp] = compare_true( ...
                                Est2, trueTheta, trueBreaks);
                        HDs2(ii) = HD_tmp;
                        F1s2(ii) = F1_tmp;
                        accs2(ii) = acc_tmp;
                        errors2(ii) = error_tmp;
                        EstBreaks2{ii} = Est2.EstBreaks();
                        if showbar && ~isempty(dq)
                            send(dq, 0);
                        end
                    end

                    % Build metric rows per selection criterion.
                    % Source order: [HD, F1, error, acc, nbreaks].
                    % Storage order: [nbreaks, HD, F1, acc, error].
                    hfe_src = two_stage_HFE(HDs2, F1s2, errors2, ...
                        accs2, EstBreaks2);

                    jj = findMinIn(loss2s);
                    loss_src = [HDs2(jj), F1s2(jj), errors2(jj), ...
                                accs2(jj), length(EstBreaks2{jj})];

                    jj = findMinIn(BIC2s);
                    BIC_src = [HDs2(jj), F1s2(jj), errors2(jj), ...
                               accs2(jj), length(EstBreaks2{jj})];

                    jj = findMinIn(HBIC_2s);
                    HBIC_src = [HDs2(jj), F1s2(jj), errors2(jj), ...
                                accs2(jj), length(EstBreaks2{jj})];

                    jj = findMinIn(HBICG_2s);
                    HBICG_src = [HDs2(jj), F1s2(jj), errors2(jj), ...
                                 accs2(jj), length(EstBreaks2{jj})];

                    % Convert source → storage order
                    method_metrics = zeros(lenIC, num_metrics);
                    method_metrics(1, :) = [hfe_src(5), hfe_src(1), hfe_src(2), ...
                                            hfe_src(4), hfe_src(3)];
                    method_metrics(2, :) = [loss_src(5), loss_src(1), loss_src(2), ...
                                            loss_src(4), loss_src(3)];
                    method_metrics(3, :) = [BIC_src(5), BIC_src(1), BIC_src(2), ...
                                            BIC_src(4), BIC_src(3)];
                    method_metrics(4, :) = [HBIC_src(5), HBIC_src(1), HBIC_src(2), ...
                                            HBIC_src(4), HBIC_src(3)];
                    method_metrics(5, :) = [HBICG_src(5), HBICG_src(1), HBICG_src(2), ...
                                            HBICG_src(4), HBICG_src(3)];

                    results_all(idx_mu1, idx_mu2, :, :, tt) = method_metrics;
                end
            end
        end

        % Aggregate statistics across replications
        results_mean = mean(results_all, 5);
        results_std = std(results_all, 0, 5);

        % DGP-specific output directory
        dgp_dir = fullfile(case_dir, sprintf('dgp_%s', dgp_name));
        if ~exist(dgp_dir, 'dir')
            mkdir(dgp_dir);
        end

        % Save MAT summary for this DGP
        save(fullfile(dgp_dir, sprintf('sensitivity_mu_pairs_%s_m%d.mat', dgp_name, num_breaks)), ...
            'T', 'p', 'num_breaks', 'num_exps', 'Nsim', 'prob', 'tol', 'beta_', ...
            'epsilon', 'kappa_', 'bb_', 'mu1s', 'mu2s', 'lamb1s', 'lamb2s', ...
            'lamb2s2', 'IC_names', 'metric_names', 'results_all', ...
            'results_mean', 'results_std', 'stage1_lamb2_selected', ...
            'stage1_nbreaks_selected');

        % Plot heatmaps for each method with five metrics per figure
        plot_method_heatmaps(dgp_dir, dgp_name, num_breaks, mu1s, mu2s, ...
                             IC_names, metric_names, results_mean);

        % Print text tables in the case log
        print_method_tables(fid, mu1s, mu2s, IC_names, results_mean, ...
                            results_std);

        % Store this DGP block into the case-level summary
        results_by_dgp(dgp_idx).DGP = dgp_name;
        results_by_dgp(dgp_idx).p = p;
        results_by_dgp(dgp_idx).num_breaks = num_breaks;
        results_by_dgp(dgp_idx).kappa = kappa_;
        results_by_dgp(dgp_idx).mu1s = mu1s;
        results_by_dgp(dgp_idx).mu2s = mu2s;
        results_by_dgp(dgp_idx).lamb1s = lamb1s;
        results_by_dgp(dgp_idx).lamb2s = lamb2s;
        results_by_dgp(dgp_idx).lamb2s2 = lamb2s2;
        results_by_dgp(dgp_idx).IC_names = IC_names;
        results_by_dgp(dgp_idx).metric_names = metric_names;
        results_by_dgp(dgp_idx).results_all = results_all;
        results_by_dgp(dgp_idx).results_mean = results_mean;
        results_by_dgp(dgp_idx).results_std = results_std;
        results_by_dgp(dgp_idx).stage1_lamb2_selected = stage1_lamb2_selected;
        results_by_dgp(dgp_idx).stage1_nbreaks_selected = stage1_nbreaks_selected;
    end

    fclose(fid);

    % Save one MAT summary per break-count case
    save(fullfile(case_dir, sprintf('sensitivity_summary_mu_pairs_m%d.mat', num_breaks)), ...
        'results_by_dgp', 'T', 'p', 'num_breaks', 'num_exps', 'Nsim', 'prob', ...
        'tol', 'beta_', 'epsilon', 'kappa_', 'mu1s', 'mu2s', 'IC_names', ...
        'metric_names');
end

%% Local helper functions

function plot_method_heatmaps(out_dir, DGP, m, mu1s, mu2s, IC_names, metric_names, results_mean)
    % Plot five metric heatmaps for each selection method.
    lenIC = length(IC_names);
    num_metrics = length(metric_names);

    for idx_IC = 1:lenIC
        fig = figure('Visible', 'off', 'Position', [100, 100, 2200, 420]);
        tiledlayout(1, num_metrics, 'TileSpacing', 'compact', 'Padding', 'compact');

        for metric_idx = 1:num_metrics
            nexttile;
            data_map = squeeze(results_mean(:, :, idx_IC, metric_idx));
            imagesc(mu2s, mu1s, data_map);
            set(gca, 'YDir', 'normal');
            xticks(mu2s);
            yticks(mu1s);
            xlabel('\mu_2');
            ylabel('\mu_1');
            title(metric_names{metric_idx});
            colorbar;
        end

        sgtitle(sprintf('DGP=%s, m=%d, method=%s', DGP, m, IC_names{idx_IC}));

        out_png = fullfile(out_dir, sprintf('heatmap_%s_m%d_%s.png', ...
            DGP, m, lower(IC_names{idx_IC})));
        exportgraphics(fig, out_png, 'Resolution', 300);
        close(fig);
    end
end

function print_method_tables(fid,  mu1s, mu2s, IC_names, results_mean, results_std)
    % Print per-method tables for all (mu1, mu2) combinations.
    for idx_IC = 1:length(IC_names)
        fprintf(fid, '\nInformation Criterion: %s\n', IC_names{idx_IC});

        headers = {'mu1', 'mu2', 'nbreaks', 'HD', 'F1', 'acc', 'error'};
        num_cols = numel(headers);
        num_rows = length(mu1s) * length(mu2s);
        table_cells = cell(num_rows, num_cols);

        rr = 0;
        for idx_mu1 = 1:length(mu1s)
            for idx_mu2 = 1:length(mu2s)
                rr = rr + 1;
                mean_row = squeeze(results_mean(idx_mu1, idx_mu2, idx_IC, :));
                std_row = squeeze(results_std(idx_mu1, idx_mu2, idx_IC, :));

                table_cells{rr, 1} = sprintf('%.1f', mu1s(idx_mu1));
                table_cells{rr, 2} = sprintf('%.1f', mu2s(idx_mu2));
                table_cells{rr, 3} = format_mean_std(mean_row(1), std_row(1), 2);
                table_cells{rr, 4} = format_mean_std(mean_row(2), std_row(2), 2);
                table_cells{rr, 5} = format_mean_std(mean_row(3), std_row(3), 3);
                table_cells{rr, 6} = format_mean_std(mean_row(4), std_row(4), 3);
                table_cells{rr, 7} = format_mean_std(mean_row(5), std_row(5), 2);
            end
        end

        col_widths = zeros(1, num_cols);
        for cc = 1:num_cols
            col_widths(cc) = numel(headers{cc});
            for rr = 1:num_rows
                col_widths(cc) = max(col_widths(cc), ...
                    numel(table_cells{rr, cc}));
            end
        end

        sep_parts = cell(1, num_cols);
        header_cells = cell(1, num_cols);
        for cc = 1:num_cols
            sep_parts{cc} = repmat('─', 1, col_widths(cc));
            header_cells{cc} = sprintf('%*s', col_widths(cc), headers{cc});
        end

        top_sep = strjoin(sep_parts, '─┬─');
        mid_sep = strjoin(sep_parts, '─┼─');
        bottom_sep = strjoin(sep_parts, '─┴─');
        header_line = strjoin(header_cells, ' │ ');

        fprintf(fid, '┌%s┐\n', top_sep);
        fprintf(fid, '│%s│\n', header_line);
        fprintf(fid, '├%s┤\n', mid_sep);

        for rr = 1:num_rows
            row_cells = cell(1, num_cols);
            for cc = 1:num_cols
                row_cells{cc} = sprintf('%*s', ...
                    col_widths(cc), table_cells{rr, cc});
            end
            fprintf(fid, '│%s│\n', strjoin(row_cells, ' │ '));
        end

        fprintf(fid, '└%s┘\n', bottom_sep);
    end
end

function out = format_mean_std(mean_val, std_val, precision_digits)
% Build a compact "mean(std)" string.
    fmt = sprintf('%%.%df(%%.%df)', precision_digits, precision_digits);
    out = sprintf(fmt, mean_val, std_val);
end
