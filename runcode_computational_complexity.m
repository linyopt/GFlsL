% Computational complexity analysis for GFlsL.
% This script evaluates the wall-clock time of the non-adaptive GFlsL, the
% first-stage estimator, and the second-stage (adaptive) estimator across
% grids of lamb1 and lamb2 values. Results are averaged over num_exps
% experiments. Each fit is a cold start timed under the plain block-scheme
% ADMM of Algorithm 1 (identity metric, relaxation 1.61); the timed
% configuration is recorded in all outputs. A fixed block-diagonal
% preconditioning (penalty_scale) leaves the estimator unchanged and only
% accelerates convergence (about 1.7-2.3x at this scale, 1.6-2.7x on large
% panels); it is used only by the real-data pipeline, not timed here.

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

% Solver configuration used for every timed fit: the plain block scheme of
% Algorithm 1 as published (identity block metric, relaxation 1.61), with
% the stopping diagnostics evaluated every 20 iterations.
penalty_scale = [1, 1, 1];
relaxation = 1.61;
termination_check_freq = 20;

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
                disp_freq=inf, merge_breaks=false, ...
                penalty_scale=penalty_scale, relaxation=relaxation, ...
                termination_check_freq=termination_check_freq);
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
            disp_freq=inf, merge_breaks=false, ...
            penalty_scale=penalty_scale, relaxation=relaxation, ...
            termination_check_freq=termination_check_freq);
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
                disp_freq=inf, merge_breaks=false, ...
                penalty_scale=penalty_scale, relaxation=relaxation, ...
                termination_check_freq=termination_check_freq);
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
     'penalty_scale', 'relaxation', 'termination_check_freq', ...
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
fprintf(fid, '  penalty_scale = [%g, %g, %g], relaxation = %.2f\n', ...
    penalty_scale, relaxation);
fprintf(fid, '  termination_check_freq = %d\n', termination_check_freq);
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

lamb2_labels = lamb2s;

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

% LaTeX tables of the (lamb1, lamb2) grids, one per estimator.
write_comp_time_tables(results_dir, lamb1s, lamb2s, ...
    times_nonadaptive_mean, times_stage2_mean);

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
            beta_=beta_, epsilon=epsilon, tol=tol, maxiter=inf, ...
            disp_freq=inf, merge_breaks=false, penalty_scale=penalty_scale, ...
            relaxation=relaxation, termination_check_freq=termination_check_freq);
        Est1.run;
        times_stage1_varyT_all(ti, exp_idx) = toc;

        % Non-adaptive timing
        tic;
        Est_na = GFlsL(X=X_current, lamb1=lamb1_fixed * p_fixed, ...
            lamb2=lamb2_fixed * p_fixed, beta_=beta_, epsilon=epsilon, ...
            tol=tol, maxiter=inf, disp_freq=inf, ...
            merge_breaks=false, penalty_scale=penalty_scale, ...
            relaxation=relaxation, termination_check_freq=termination_check_freq);
        Est_na.run;
        times_nonadaptive_varyT_all(ti, exp_idx) = toc;

        % Second-stage timing (using first-stage estimator)
        xi1_ = max(abs(Est1.Theta_k), bb_current) .^ -mu1;
        xi2_ = max(Est1.norm_diff, bb_current) .^ -mu2;

        tic;
        Est2 = GFlsL(X=X_current, lamb1=lamb1_fixed * p_fixed .* xi1_, ...
            lamb2=lamb2_fixed * p_fixed .* reshape(xi2_, 1, []), beta_=beta_, ...
            epsilon=epsilon, tol=tol, maxiter=inf, disp_freq=inf, ...
            merge_breaks=false, penalty_scale=penalty_scale, ...
            relaxation=relaxation, termination_check_freq=termination_check_freq);
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
            beta_=beta_, epsilon=epsilon, tol=tol, maxiter=inf, ...
            disp_freq=inf, merge_breaks=false, penalty_scale=penalty_scale, ...
            relaxation=relaxation, termination_check_freq=termination_check_freq);
        Est1.run;
        times_stage1_varyp_all(pi, exp_idx) = toc;

        % Non-adaptive timing
        tic;
        Est_na = GFlsL(X=X_current, lamb1=lamb1_fixed * p_current, ...
            lamb2=lamb2_fixed * p_current, beta_=beta_, epsilon=epsilon, ...
            tol=tol, maxiter=inf, disp_freq=inf, ...
            merge_breaks=false, penalty_scale=penalty_scale, ...
            relaxation=relaxation, termination_check_freq=termination_check_freq);
        Est_na.run;
        times_nonadaptive_varyp_all(pi, exp_idx) = toc;

        % Second-stage timing (using first-stage estimator)
        xi1_ = max(abs(Est1.Theta_k), bb_current) .^ -mu1;
        xi2_ = max(Est1.norm_diff, bb_current) .^ -mu2;

        tic;
        Est2 = GFlsL(X=X_current, lamb1=lamb1_fixed * p_current .* xi1_, ...
            lamb2=lamb2_fixed * p_current .* reshape(xi2_, 1, []), ...
            beta_=beta_, epsilon=epsilon, tol=tol, maxiter=inf, ...
            disp_freq=inf, merge_breaks=false, ...
            penalty_scale=penalty_scale, relaxation=relaxation, ...
            termination_check_freq=termination_check_freq);
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
    'penalty_scale', 'relaxation', 'termination_check_freq', ...
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

% =========================================================================
function write_comp_time_tables(results_dir, lamb1s, lamb2s, times_nonadaptive, times_stage2)
%% Write the (lamb1, lamb2) computation-time grids as one LaTeX float body.
%
% The two estimators are emitted side by side, each scaled to fit the text
% block, so that the body can be pasted into a table environment together
% with its caption and label. Rows are indexed by lamb1 and columns by lamb2,
% which keeps the full grid the heatmaps used to carry.

    panels = { ...
        struct('title', 'Non-adaptive', 'times', times_nonadaptive), ...
        struct('title', 'Second-stage (adaptive)', 'times', times_stage2)};

    fid = fopen(fullfile(results_dir, 'comp-time-table.tex'), 'w');
    if fid < 0
        error('runcode_computational_complexity:openTableFailed', ...
            'Cannot write the computation-time table.');
    end
    for kk = 1:numel(panels)
        fprintf(fid, '\\resizebox{0.465\\textwidth}{!}{%%\n');
        write_comp_time_grid(fid, lamb1s, lamb2s, ...
            panels{kk}.title, panels{kk}.times);
        fprintf(fid, '}\n');
        if kk < numel(panels)
            fprintf(fid, '\\hfill\n');
        end
    end
    fclose(fid);
end

function write_comp_time_grid(fid, lamb1s, lamb2s, panel_title, times)
%% Write one (lamb1, lamb2) computation-time grid as a LaTeX tabular.
    fprintf(fid, '\\begin{tabular}{l*{%d}{c}}\n', numel(lamb2s));
    fprintf(fid, '\\toprule\n');
    fprintf(fid, '\\multicolumn{%d}{c}{%s} \\\\\n', ...
        numel(lamb2s) + 1, panel_title);
    fprintf(fid, '$\\lambda_1 \\backslash \\lambda_2$');
    fprintf(fid, ' & %.1f', lamb2s);
    fprintf(fid, ' \\\\\n');
    fprintf(fid, '\\midrule\n');
    for ii = 1:numel(lamb1s)
        fprintf(fid, '%.4f', lamb1s(ii));
        fprintf(fid, ' & %.2f', times(ii, :));
        fprintf(fid, ' \\\\\n');
    end
    fprintf(fid, '\\bottomrule\n');
    fprintf(fid, '\\end{tabular}\n');
end
