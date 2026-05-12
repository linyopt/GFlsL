% Computational complexity analysis for GFlsL.
% This script evaluates the wall-clock time of the non-adaptive GFlsL, the
% first-stage estimator, and the second-stage (adaptive) estimator across
% grids of lamb1 and lamb2 values. Results are averaged over num_exps
% experiments.

addpath(genpath(pwd));
clear;
clc;
close all;

rng(42, 'twister');

% Shared experiment settings
T = 200;
p = 10;
num_breaks = 1;
prob = 0.8;
% No validation replications are required
Nsim = 0;
% Number of experiments to average over
num_exps = 100;

% Hyperparameters.
tol = 1e-3;
beta_ = 0.1;
epsilon = 0.01;
mu1 = 0.8;
mu2 = 1.5;
kappa_ = 0.5;
bb_ = T ^ (-kappa_);

% Grids of tuning parameters.
lamb1s = p * (1e-5:1e-5:1e-4);
lamb2s = p * (1e-2:1e-2:1e-1);

len1 = length(lamb1s);
len2 = length(lamb2s);

% Ensure results directory exists before writing any output.
results_dir = fullfile('results', 'computational_complexity');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Storage for all experiments
times_nonadaptive_all = zeros(len1, len2, num_exps);
times_stage1_all = zeros(len2, num_exps);
stage1_break_counts_all = zeros(len2, num_exps);
times_stage2_all = zeros(len1, len2, num_exps);

% Main loop over experiments
for exp_idx = 1:num_exps
    % Generate ONE dataset for this experiment (used by all stages)
    [X, ~, ~, ~] = DGP_iid(T, p, num_breaks, prob, Nsim);
    
    % (a) Non-adaptive GFlsL complexity over (lamb1, lamb2)
    for ii = 1:len1
        for jj = 1:len2
            tic;
            Est = GFlsL(X=X, lamb1=lamb1s(ii), lamb2=lamb2s(jj), ...
                beta_=beta_, epsilon=epsilon, tol=tol, maxiter=inf, ...
                disp_freq=inf, showfig=0, merge_breaks=false);
            Est.run;
            times_nonadaptive_all(ii, jj, exp_idx) = toc;
        end
    end
    
    % (b) First-stage GFlsL complexity over lamb2
    stage1_estimators = cell(1, len2);
    
    for jj = 1:len2
        tic;
        Est1 = GFlsL(X=X, lamb1=0, lamb2=lamb2s(jj), ...
            beta_=beta_, epsilon=epsilon, tol=tol, maxiter=inf, ...
            disp_freq=inf, showfig=0, merge_breaks=false);
        Est1.run;
        times_stage1_all(jj, exp_idx) = toc;
        stage1_break_counts_all(jj, exp_idx) = length(Est1.EstBreaks);
        stage1_estimators{jj} = Est1;
    end
    
    % (c) Second-stage GFlsL complexity over (lamb1, lamb2)
    % Select the first-stage estimator with estimated breaks >= true breaks
    % and closest to the true number of breaks to build adaptive weights.
    stage1_break_counts = stage1_break_counts_all(:, exp_idx);
    candidates = find(stage1_break_counts >= num_breaks);
    if isempty(candidates)
        [~, idx_stage1] = min(abs(stage1_break_counts - num_breaks));
    else
        [~, rel_idx] = min(stage1_break_counts(candidates));
        idx_stage1 = candidates(rel_idx);
    end
    refEst = stage1_estimators{idx_stage1};
    
    % Compute adaptive weights from the selected first-stage estimator.
    xi1_ = max(abs(refEst.Theta_k), bb_) .^ -mu1;
    xi2_ = max(refEst.norm_diff, bb_) .^ -mu2;
    
    for ii = 1:len1
        for jj = 1:len2
            tic;
            Est2 = GFlsL(X=X, lamb1=lamb1s(ii) .* xi1_, ...
                lamb2=lamb2s(jj) .* reshape(xi2_, 1, []), ...
                beta_=beta_, epsilon=epsilon, tol=tol, maxiter=inf, ...
                disp_freq=inf, showfig=0, merge_breaks=false);
            Est2.run;
            times_stage2_all(ii, jj, exp_idx) = toc;
        end
    end
end

% Compute mean and std across experiments
times_nonadaptive_mean = mean(times_nonadaptive_all, 3);
times_nonadaptive_std = std(times_nonadaptive_all, 0, 3);

times_stage1_mean = mean(times_stage1_all, 2)';
times_stage1_std = std(times_stage1_all, 0, 2)';

times_stage2_mean = mean(times_stage2_all, 3);
times_stage2_std = std(times_stage2_all, 0, 3);

% Persist the raw timings
save(fullfile(results_dir, 'complexity_times.mat'), 'T', 'p', 'num_breaks', 'prob', 'lamb1s', 'lamb2s', ...
     'times_nonadaptive_all', 'times_nonadaptive_mean', 'times_nonadaptive_std', ...
     'times_stage1_all', 'times_stage1_mean', 'times_stage1_std', ...
     'times_stage2_all', 'times_stage2_mean', 'times_stage2_std', ...
     'stage1_break_counts_all', 'num_exps');

% Save heatmap results to txt file for easy reading
fid = fopen(fullfile(results_dir, 'heatmap_results.txt'), 'w');
fprintf(fid, 'Computational Complexity Results (num_exps=%d)\n', num_exps);
fprintf(fid, '===================================================\n\n');

fprintf(fid, 'Settings:\n');
fprintf(fid, '  T = %d, p = %d, num_breaks = %d, prob = %.1f\n', T, p, num_breaks, prob);
fprintf(fid, '  lamb1s = p * [%.0e, ..., %.0e] (len = %d)\n', lamb1s(1)/p, lamb1s(end)/p, len1);
fprintf(fid, '  lamb2s = p * [%.0e, ..., %.0e] (len = %d)\n\n', lamb2s(1)/p, lamb2s(end)/p, len2);

fprintf(fid, 'First-stage GFlsL (lamb1=0, varying lamb2):\n');
fprintf(fid, '%-12s %-20s %-15s\n', 'lamb2', 'Time (s)', 'Mean Breaks');
fprintf(fid, '%-12s %-20s %-15s\n', '------', '-------------------', '----------');
for jj = 1:len2
    fprintf(fid, '%-12.4f %-20s %-15.2f\n', lamb2s(jj), ...
        sprintf('%.4f(%.4f)', times_stage1_mean(jj), times_stage1_std(jj)), ...
        mean(stage1_break_counts_all(jj, :)));
end
fprintf(fid, '\n');

fprintf(fid, 'Non-adaptive GFlsL Mean Time (rows: lamb1, cols: lamb2):\n');
fprintf(fid, '%-12s', 'lamb1\\lamb2');
for jj = 1:len2
    fprintf(fid, ' %-10.4f', lamb2s(jj));
end
fprintf(fid, '\n');
for ii = 1:len1
    fprintf(fid, '%-12.4f', lamb1s(ii));
    for jj = 1:len2
        fprintf(fid, ' %-10.4f', times_nonadaptive_mean(ii, jj));
    end
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

fprintf(fid, 'Non-adaptive GFlsL Std Time (rows: lamb1, cols: lamb2):\n');
fprintf(fid, '%-12s', 'lamb1\\lamb2');
for jj = 1:len2
    fprintf(fid, ' %-10.4f', lamb2s(jj));
end
fprintf(fid, '\n');
for ii = 1:len1
    fprintf(fid, '%-12.4f', lamb1s(ii));
    for jj = 1:len2
        fprintf(fid, ' %-10.4f', times_nonadaptive_std(ii, jj));
    end
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

fprintf(fid, 'Second-stage GFlsL Time (rows: lamb1, cols: lamb2), format: mean(std):\n');
fprintf(fid, '%-12s', 'lamb1\\lamb2');
for jj = 1:len2
    fprintf(fid, ' %-18s', sprintf('%.4f', lamb2s(jj)));
end
fprintf(fid, '\n');
for ii = 1:len1
    fprintf(fid, '%-12.4f', lamb1s(ii));
    for jj = 1:len2
        fprintf(fid, ' %-18s', sprintf('%.4f(%.4f)', times_stage2_mean(ii, jj), times_stage2_std(ii, jj)));
    end
    fprintf(fid, '\n');
end
fclose(fid);

% Visualizations

lamb1_labels = lamb1s;
lamb2_labels = lamb2s;
% Heatmap for non-adaptive GFlsL (using mean times)
fig = figure('Visible', 'off');
h_na = heatmap(lamb2_labels, lamb1_labels, times_nonadaptive_mean);
h_na.Title = sprintf('Non-adaptive GFlsL time (T=%d, p=%d, %d exps)', T, p, num_exps);
h_na.XLabel = '\lambda_2';
h_na.YLabel = '\lambda_1';
h_na.ColorbarVisible = 'on';
exportgraphics(fig, fullfile(results_dir, 'comp-time-nonadaptive.png'), 'Resolution', 300);
close(fig);

% Line plot for first-stage GFlsL using boundedline (mean with std bounds)
fig = figure('Visible', 'off');
[hl, ~] = boundedline(lamb2_labels, times_stage1_mean, times_stage1_std, '-o', ...
    'LineWidth', 1.5, 'transparency', 0.3, 'alpha');
xlabel('\lambda', 'FontSize', 12);
ylabel('Computation time (s)', 'FontSize', 12);
title(sprintf('First-stage GFlsL time (T=%d, p=%d, %d exps)', T, p, num_exps), 'FontSize', 12);
grid on;
% Add legend for clarity
legend(hl(1), 'Mean time', 'Location', 'northwest');
exportgraphics(fig, fullfile(results_dir, 'comp-time-stage1.png'), 'Resolution', 300);
close(fig);

% Heatmap for second-stage GFlsL (using mean times)
fig = figure('Visible', 'off');
h_s2 = heatmap(lamb2_labels, lamb1_labels, times_stage2_mean);
h_s2.Title = sprintf('Second-stage GFlsL time (T=%d, p=%d, %d exps)', T, p, num_exps);
h_s2.XLabel = '\lambda_2';
h_s2.YLabel = '\lambda_1';
h_s2.ColorbarVisible = 'on';
exportgraphics(fig, fullfile(results_dir, 'comp-time-stage2.png'), 'Resolution', 300);
close(fig);

%% T and p computational complexity analysis

rng(42, 'twister');

% Number of experiments to average over
num_exps = 100;
% Shared experiment settings
num_breaks = 1;
prob = 0.8;
% No validation replications are required
Nsim = 0;

% Ensure results directory exists before writing any output.
results_dir = fullfile('results', 'computational_complexity');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Hyperparameters.
tol = 1e-3;
beta_ = 0.1;
epsilon = 0.01;
mu1 = 0.8;
mu2 = 1.5;
kappa_ = 0.5;

% Use middle lambda values that correspond to approximately the lowest
% computation times.
% From the previous grid analysis, lambda = 0.05 * p is approximately the
% fastest.
lamb_stage1_fixed = 0.05;
lamb1_fixed = 5e-5;
lamb2_fixed = 0.05;

% Varying T (fix p = 10)
T_values = [50, 100, 150, 200, 250, 300];
lenT = length(T_values);
p_fixed = 10;

times_stage1_varyT_all = zeros(lenT, num_exps);
times_nonadaptive_varyT_all = zeros(lenT, num_exps);
times_stage2_varyT_all = zeros(lenT, num_exps);

for ti = 1:lenT
    T_current = T_values(ti);
    bb_current = T_current ^ (-kappa_);

    for exp_idx = 1:num_exps
        [X_current, ~, ~, ~] = DGP_iid(T_current, p_fixed, num_breaks, prob, Nsim);

        % First-stage timing
        tic;
        Est1 = GFlsL(X=X_current, lamb1=0, lamb2=lamb_stage1_fixed * p_fixed, ...
            beta_=beta_, epsilon=epsilon, tol=tol, maxiter=inf, disp_freq=inf, ...
            showfig=0, merge_breaks=false);
        Est1.run;
        times_stage1_varyT_all(ti, exp_idx) = toc;

        % Non-adaptive timing
        tic;
        Est_na = GFlsL(X=X_current, lamb1=lamb1_fixed * p_fixed, ...
            lamb2=lamb2_fixed * p_fixed, beta_=beta_, epsilon=epsilon, ...
            tol=tol, maxiter=inf, disp_freq=inf, showfig=0, merge_breaks=false);
        Est_na.run;
        times_nonadaptive_varyT_all(ti, exp_idx) = toc;

        % Second-stage timing (using first-stage estimator)
        xi1_ = max(abs(Est1.Theta_k), bb_current) .^ -mu1;
        xi2_ = max(Est1.norm_diff, bb_current) .^ -mu2;

        tic;
        Est2 = GFlsL(X=X_current, lamb1=lamb1_fixed * p_fixed .* xi1_, ...
            lamb2=lamb2_fixed * p_fixed .* reshape(xi2_, 1, []), beta_=beta_, ...
            epsilon=epsilon, tol=tol, maxiter=inf,disp_freq=inf, showfig=0, ...
            merge_breaks=false);
        Est2.run;
        times_stage2_varyT_all(ti, exp_idx) = toc;
    end
end

times_stage1_varyT_mean = mean(times_stage1_varyT_all, 2);
times_stage1_varyT_std = std(times_stage1_varyT_all, 0, 2);
times_nonadaptive_varyT_mean = mean(times_nonadaptive_varyT_all, 2);
times_nonadaptive_varyT_std = std(times_nonadaptive_varyT_all, 0, 2);
times_stage2_varyT_mean = mean(times_stage2_varyT_all, 2);
times_stage2_varyT_std = std(times_stage2_varyT_all, 0, 2);

% Varying p (fix T = 200)
p_values = [5, 10, 15, 20, 25, 30];
lenp = length(p_values);
T_fixed = 200;

times_stage1_varyp_all = zeros(lenp, num_exps);
times_nonadaptive_varyp_all = zeros(lenp, num_exps);
times_stage2_varyp_all = zeros(lenp, num_exps);

for pi = 1:lenp
    p_current = p_values(pi);
    bb_current = T_fixed ^ (-kappa_);

    for exp_idx = 1:num_exps
        [X_current, ~, ~, ~] = DGP_iid(T_fixed, p_current, num_breaks, prob, Nsim);

        % First-stage timing
        tic;
        Est1 = GFlsL(X=X_current, lamb1=0, lamb2=lamb_stage1_fixed * p_current, ...
            beta_=beta_, epsilon=epsilon, tol=tol, maxiter=inf, disp_freq=inf, ...
            showfig=0, merge_breaks=false);
        Est1.run;
        times_stage1_varyp_all(pi, exp_idx) = toc;

        % Non-adaptive timing
        tic;
        Est_na = GFlsL(X=X_current, lamb1=lamb1_fixed * p_current, ...
            lamb2=lamb2_fixed * p_current, beta_=beta_, epsilon=epsilon, ...
            tol=tol, maxiter=inf, disp_freq=inf, showfig=0, merge_breaks=false);
        Est_na.run;
        times_nonadaptive_varyp_all(pi, exp_idx) = toc;

        % Second-stage timing (using first-stage estimator)
        xi1_ = max(abs(Est1.Theta_k), bb_current) .^ -mu1;
        xi2_ = max(Est1.norm_diff, bb_current) .^ -mu2;

        tic;
        Est2 = GFlsL(X=X_current, lamb1=lamb1_fixed * p_current .* xi1_, ...
            lamb2=lamb2_fixed * p_current .* reshape(xi2_, 1, []), ...
            beta_=beta_, epsilon=epsilon, tol=tol, maxiter=inf, ...
            disp_freq=inf, showfig=0, merge_breaks=false);
        Est2.run;
        times_stage2_varyp_all(pi, exp_idx) = toc;
    end
end

times_stage1_varyp_mean = mean(times_stage1_varyp_all, 2);
times_stage1_varyp_std = std(times_stage1_varyp_all, 0, 2);
times_nonadaptive_varyp_mean = mean(times_nonadaptive_varyp_all, 2);
times_nonadaptive_varyp_std = std(times_nonadaptive_varyp_all, 0, 2);
times_stage2_varyp_mean = mean(times_stage2_varyp_all, 2);
times_stage2_varyp_std = std(times_stage2_varyp_all, 0, 2);

% Save T and p variation results
save(fullfile(results_dir, 'complexity_times_Tp.mat'), ...
    'T_values', 'p_values', 'p_fixed', 'T_fixed', ...
    'lamb_stage1_fixed', 'lamb1_fixed', 'lamb2_fixed', ...
    'times_stage1_varyT_all', 'times_stage1_varyT_mean', ...
    'times_stage1_varyT_std', ...
    'times_nonadaptive_varyT_all', 'times_nonadaptive_varyT_mean', ...
    'times_nonadaptive_varyT_std', ...
    'times_stage2_varyT_all', 'times_stage2_varyT_mean', ...
    'times_stage2_varyT_std', ...
    'times_stage1_varyp_all', 'times_stage1_varyp_mean', ...
    'times_stage1_varyp_std', ...
    'times_nonadaptive_varyp_all', 'times_nonadaptive_varyp_mean', ...
    'times_nonadaptive_varyp_std', ...
    'times_stage2_varyp_all', 'times_stage2_varyp_mean', ...
    'times_stage2_varyp_std', 'num_exps');

% Plot T variation results
fig = figure('Visible', 'off');
[hl, ~] = boundedline(T_values, times_stage1_varyT_mean, times_stage1_varyT_std, ...
    '-o', 'LineWidth', 1.5, 'transparency', 0.3, 'alpha');
xlabel('T', 'FontSize', 12);
ylabel('Computation time (s)', 'FontSize', 12);
title(sprintf('First-stage GFlsL time vs T (p=%d, \\lambda=%.2fp, %d exps)', ...
    p_fixed, lamb_stage1_fixed, num_exps), 'FontSize', 12);
grid on;
legend(hl(1), 'Mean time', 'Location', 'northwest');
exportgraphics(fig, fullfile(results_dir, 'comp-time-stage1-varyT.png'), ...
    'Resolution', 300);
close(fig);

fig = figure('Visible', 'off');
[hl, ~] = boundedline(T_values, times_nonadaptive_varyT_mean, times_nonadaptive_varyT_std, ...
    '-o', 'LineWidth', 1.5, 'transparency', 0.3, 'alpha');
xlabel('T', 'FontSize', 12);
ylabel('Computation time (s)', 'FontSize', 12);
title(sprintf('Non-adaptive GFlsL time vs T (p=%d, \\lambda_1=%.5fp, \\lambda_2=%.2fp, %d exps)', ...
    p_fixed, lamb1_fixed, lamb2_fixed, num_exps), 'FontSize', 12);
grid on;
legend(hl(1), 'Mean time', 'Location', 'northwest');
exportgraphics(fig, fullfile(results_dir, 'comp-time-nonadaptive-varyT.png'), ...
    'Resolution', 300);
close(fig);

fig = figure('Visible', 'off');
[hl, ~] = boundedline(T_values, times_stage2_varyT_mean, times_stage2_varyT_std, ...
    '-o', 'LineWidth', 1.5, 'transparency', 0.3, 'alpha');
xlabel('T', 'FontSize', 12);
ylabel('Computation time (s)', 'FontSize', 12);
title(sprintf( ...
    'Second-stage GFlsL time vs T (p=%d, \\lambda_1=%.5fp, \\lambda_2=%.2fp, %d exps)', ...
    p_fixed, lamb1_fixed, lamb2_fixed, num_exps), 'FontSize', 12);
grid on;
legend(hl(1), 'Mean time', 'Location', 'northwest');
exportgraphics(fig, fullfile(results_dir, 'comp-time-stage2-varyT.png'), ...
    'Resolution', 300);
close(fig);

% Plot p variation results
fig = figure('Visible', 'off');
[hl, ~] = boundedline(p_values, times_stage1_varyp_mean, times_stage1_varyp_std, ...
    '-o', 'LineWidth', 1.5, 'transparency', 0.3, 'alpha');
xlabel('p', 'FontSize', 12);
ylabel('Computation time (s)', 'FontSize', 12);
title(sprintf('First-stage GFlsL time vs p (T=%d, \\lambda=%.2fp, %d exps)', ...
    T_fixed, lamb_stage1_fixed, num_exps), 'FontSize', 12);
grid on;
legend(hl(1), 'Mean time', 'Location', 'northwest');
exportgraphics(fig, fullfile(results_dir, 'comp-time-stage1-varyp.png'), ...
    'Resolution', 300);
close(fig);

fig = figure('Visible', 'off');
[hl, ~] = boundedline(p_values, times_nonadaptive_varyp_mean, ...
    times_nonadaptive_varyp_std, '-o', 'LineWidth', 1.5, ...
    'transparency', 0.3, 'alpha');
xlabel('p', 'FontSize', 12);
ylabel('Computation time (s)', 'FontSize', 12);
title(sprintf( ...
    'Non-adaptive GFlsL time vs p (T=%d, \\lambda_1=%.5fp, \\lambda_2=%.2fp, %d exps)', ...
    T_fixed, lamb1_fixed, lamb2_fixed, num_exps), 'FontSize', 12);
grid on;
legend(hl(1), 'Mean time', 'Location', 'northwest');
exportgraphics(fig, fullfile(results_dir, 'comp-time-nonadaptive-varyp.png'), ...
    'Resolution', 300);
close(fig);

fig = figure('Visible', 'off');
[hl, ~] = boundedline(p_values, times_stage2_varyp_mean, ...
    times_stage2_varyp_std, '-o', 'LineWidth', 1.5, ...
    'transparency', 0.3, 'alpha');
xlabel('p', 'FontSize', 12);
ylabel('Computation time (s)', 'FontSize', 12);
title(sprintf( ...
    'Second-stage GFlsL time vs p (T=%d, \\lambda_1=%.5fp, \\lambda_2=%.2fp, %d exps)', ...
    T_fixed, lamb1_fixed, lamb2_fixed, num_exps), 'FontSize', 12);
grid on;
legend(hl(1), 'Mean time', 'Location', 'northwest');
exportgraphics(fig, ...
    fullfile(results_dir, 'comp-time-stage2-varyp.png'), ...
    'Resolution', 300);
close(fig);
