addpath(genpath(pwd))

clear;
% clc

% Set random seed for reproducibility.
rng(42, 'twister')

% The variable for determining if save the results to files.
savetxt = 1;
savemat = 1;
% Keep false by default so a resumed run cannot silently overwrite saved
% per-setting MAT files from a previous long simulation.
overwrite_mat = false;
showbar = 1;
run_post_refits = true;
run_gflsl = true;

% Settings.
T = 200;
% Experiment settings.
num_exps = 100;
% Keep the setting axes explicit because they are also used to pre-validate
% the per-setting MAT filenames before the expensive simulation starts.
DGPs = ["banded", "iid", "factor"];
ps = [10, 20];
num_breaks_list = [0, 1, 3];

tune_mu2 = 0;
tune_mu1 = 0;
prob = 0.8;
Nsim = 10;

% Parameters for algorithm.
tol = 1e-3;
beta_ = 0.1;

% hyperparameters for adaptive weights
if tune_mu2
    mu2 = [0.5, 0.8, 1.5];
else
    mu2 = 1.5;
end

if tune_mu1
    mu1 = [0.5, 0.8, 1.5];
else
    mu1 = 0.8;
end

kappa_ = 0.5;

% The threshold for BSOP.
tau_BSOP = 10;

% The delta and the threshold for WBSIP, according to their code.
delta_WBSIP = 5;
M_WBSIP = 100;

% The gamma and zeta for DCDP.
% If gamma_DCDP is a vector, CV will select from candidates.
gamma_DCDP = [100, 200, 300];
zeta_DCDP = 0;
Q_DCDP = 50;
buffer_DCDP = 4;
buffer_refine_DCDP = 4;

% File names for saving results. All settings share one text log, while
% each setting writes its own MAT file.
results_dir = fullfile('results', 'simulations');
txt_filepre = string(fullfile(results_dir, 'simulations_results'));
mat_filepre_base = string(fullfile(results_dir, 'simulations_results'));

% Ensure results directory exists before writing any output.
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

if savemat
    % Build the full list of MAT files up front. This catches naming
    % collisions before any data generation or detector fitting work is done.
    expected_mat_files = strings(numel(DGPs) * numel(ps) * numel(num_breaks_list), 1);
    idx_setting = 0;
    for DGP_check = DGPs
        for p_check = ps
            for num_breaks_check = num_breaks_list
                idx_setting = idx_setting + 1;
                setting_key = sprintf('%s_p%d_m%d', char(DGP_check), p_check, ...
                    num_breaks_check);
                expected_mat_files(idx_setting) = mat_filepre_base + "_" + ...
                    setting_key + ".mat";
            end
        end
    end

    % MAT output uses one file per setting, so existing files likely mean this
    % run would overwrite reusable detector outputs unless explicitly allowed.
    existing_mat_files = expected_mat_files(arrayfun(@(x) exist(char(x), ...
        'file') == 2, expected_mat_files));
    if ~overwrite_mat && ~isempty(existing_mat_files)
        error('runcode_simu:matFilesExist', ...
            ['Per-setting MAT files already exist. Set overwrite_mat=true ', ...
            'to overwrite them. First existing file: %s'], ...
            char(existing_mat_files(1)));
    end
end

for DGP = DGPs
    for p = ps
        % Use one shared lambda grid for each p across all simulation DGPs.
        if p == 10
            lamb1s = p * (2.0e-4:5.0e-4:3.2e-3);
            lamb2s = p * (0.12:0.08:0.44);
            lamb2s2 = p * (0.02:0.07:0.58);
        elseif p == 20
            lamb1s = p * (2.0e-4:4.0e-4:2.6e-3);
            lamb2s = p * (0.12:0.08:0.44);
            lamb2s2 = p * (0.02:0.06:0.38);
        else
            lamb1s = p * (2.0e-4:4.0e-4:2.6e-3);
            lamb2s = p * (0.12:0.08:0.44);
            lamb2s2 = p * (0.02:0.06:0.38);
        end

        for num_breaks = num_breaks_list
            % The text log stays shared, but each setting gets its own MAT
            % prefix so saved detector breaks can be reused later.
            setting_key = sprintf('%s_p%d_m%d', char(DGP), p, num_breaks);
            mat_filepre = mat_filepre_base + "_" + setting_key;

            % tau for WBSIP.
            tau_WBSIP = sqrt(p * log(T)) * 1.5;

            % For storage of generated data.
            X = zeros(T, p, num_exps);
            X_test = zeros(T, p, Nsim, num_exps);
            trueTheta = zeros(p, p, T, num_exps);
            trueBreaks = cell(num_exps, 1);

            % The loops.
            % The outest loop is for experiments.
            for tt = 1:num_exps
                % Data generation.
                if string(DGP) == "iid"
                    % iid DGP.
                    [X(:, :, tt), X_test(:, :, :, tt), trueTheta(:, :, :, tt), ...
                        trueBreaks{tt}] = DGP_iid(T, p, num_breaks, prob, Nsim);
                elseif string(DGP) == "banded"
                    % banded DGP.
                    [X(:, :, tt), X_test(:, :, :, tt), trueTheta(:, :, :, tt), ...
                        trueBreaks{tt}] = DGP_banded(T, p, num_breaks, Nsim);
                elseif string(DGP) == "factor"
                    % factor DGP.
                    [X(:, :, tt), X_test(:, :, :, tt), trueTheta(:, :, :, tt), ...
                        trueBreaks{tt}] = DGP_factor(T, p, num_breaks, prob, Nsim);
                end
            end

            % Set prob parameter based on DGP type.
            % Banded DGP does not use prob, so pass NaN to exclude it from
            % output. iid and factor DGPs use prob.
            if string(DGP) == "iid" || string(DGP) == "factor"
                prob_to_pass = prob;
            else
                prob_to_pass = NaN;
            end

            if ~run_post_refits
                methods_post = string.empty(1, 0);
            end

            if string(DGP) == "banded"
                % The banded convex-banding benchmark uses fixed lambda_scale
                % values inside get_two_stage_est().
                methods_post = ["ec2", "xue_ma_zou", "adaptive_threshold_fspd", ...
                    "convex_banding"];
            else
                methods_post = ["ec2", "xue_ma_zou", "adaptive_threshold_fspd"];
            end

            get_two_stage_est(X, X_test, trueTheta, trueBreaks, lamb1s, lamb2s, ...
                lamb2s2, mu2=mu2, mu1=mu1, kappa_=kappa_, tol=tol, ...
                beta_=beta_, tau_BSOP=tau_BSOP, delta_WBSIP=delta_WBSIP, ...
                tau_WBSIP=tau_WBSIP, M_WBSIP=M_WBSIP, gamma_DCDP=gamma_DCDP, ...
                zeta_DCDP=zeta_DCDP, Q_DCDP=Q_DCDP, buffer_DCDP=buffer_DCDP, ...
                buffer_refine_DCDP=buffer_refine_DCDP, showbar=showbar, savetxt=savetxt, ...
                filepre=txt_filepre, txt_filepre=txt_filepre, mat_filepre=mat_filepre, ...
                merge_breaks=false, DGP=DGP, prob=prob_to_pass, ...
                post_refit_methods=methods_post, savemat=savemat, ...
                run_gflsl=run_gflsl);

        end
    end
end
