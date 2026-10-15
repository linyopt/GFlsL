function get_two_stage_est(X, X_test, trueTheta, trueBreaks, lamb1s, lamb2s, lamb2s2, varargin)
%% Obtain two-stage estimators.
%
% - Usage:
%   get_two_stage_est(X, X_test, trueTheta, trueBreaks, lamb1s, lamb2s, lamb2s2, ...
%       'mu2', 0.5, 'mu1', 0.8, 'kappa_', 0.5, 'tol', 1e-3, 'beta_', 0.1, ...
%       'tau_BSOP', 10, 'delta_WBSIP', 5, 'tau_WBSIP', sqrt(p*log(T))*1.5, ...
%       'gamma_DCDP', 10, 'zeta_DCDP', 1, ...
%       'filepre', "res", 'txt_filepre', "res", 'mat_filepre', "res", ...
%       'savetxt', 0, 'savemat', 0, 'showbar', 0, ...
%       'dates', 1:T, 'run_post_refits', true, 'termination_check_freq', 1)
%   OR
%   get_two_stage_est(X, X_test, trueTheta, trueBreaks, lamb1s, lamb2s, lamb2s2, ...
%       mu2=0.5, mu1=0.8, kappa_=0.5, tol=1e-3, beta_=0.1, tau_BSOP=10, ...
%       delta_WBSIP=5, tau_WBSIP=sqrt(p*log(T))*1.5, gamma_DCDP=10, zeta_DCDP=1, ...
%       filepre="res", txt_filepre="res", mat_filepre="res", ...
%       savetxt=0, savemat=0, showbar=0, dates=1:T, ...
%       run_post_refits=true, termination_check_freq=1)
%
% - Input:
%   @X: The training data.
%   @X_test: The test data. If nonempty, out-of-sample loss is evaluated
%            on X_test and the WBSIP/DCDP competitor paths are enabled.
%   @trueTheta: The true Theta values.
%   @trueBreaks: The true break points.
%   @lamb1s: Tuning parameter candidates for the first stage.
%   @lamb2s: Tuning parameter candidates for the first stage.
%   @lamb2s2: Tuning parameter candidates for the second stage.
%
% - Optional input (name-value pairs):
%   @mu2: Hyperparameter for adaptive weight xi2. Default is 0.5.
%   @mu1: Hyperparameter for adaptive weight xi1. Default is 0.8.
%   @kappa_: Hyperparameter for adaptive weights. Default is 0.5.
%   @tol: Tolerance for the algorithm. Default is 1e-3.
%   @beta_: Parameter for the algorithm. Default is 0.1.
%   @penalty_scale: Relative penalties for the V, Upsilon, and D constraint
%                   blocks. Default is [1, 1, 1].
%   @relaxation: Dual-ascent relaxation in (0, (1+sqrt(5))/2). Default is 1.61.
%   @tau_BSOP: Parameter for the BSOP method. Default is 10.
%   @delta_WBSIP: Parameter for the WBSIP method. Default is 5.
%   @tau_WBSIP: Parameter for the WBSIP method. Default is calculated based on p and T.
%   @gamma_DCDP: Penalty parameter for DCDP divide step. If vector, CV selects
%                from candidates; if scalar, uses fixed value. Default is 10.
%   @zeta_DCDP: Penalty parameter for DCDP conquer step. Default is 1.
%   @Q_DCDP: Grid size for DCDP dynamic programming. Default is 10.
%   @filepre: Prefix for saving files. Default is "res".
%   @txt_filepre: Prefix for the text log. Default falls back to filepre.
%   @mat_filepre: Prefix for the MAT file. Default falls back to filepre.
%   @savetxt: Flag to save results to a text file. Default is 0.
%   @savemat: Flag to save results to a MAT file. Default is 0.
%   @save_candidate_thetas: If true, include every candidate's full Theta
%                           path in MAT output. Default is true.
%   @showbar: Flag to show the progress bar. Default is 0.
%   @dates:   The dates. Default is 1:T.
%   @merge_breaks: Apply mergeBreaks post-processing on GFlsL outputs. Default is 0.
%   @save_tbfl_omega: If true, compute and save TBFL precision matrices (Omega)
%                     using EstimateOmega method. Default is false.
%   @run_gflsl: If false, skip the GFlsL adaptive/non-adaptive fits and
%               only run competing baselines plus optional post-refits.
%   @post_refit_methods: String array of post-refit covariance estimators
%                        applied to competing baselines.
%   @run_post_refits: If false, skip post-refit covariance estimators even
%                     when post_refit_methods is nonempty. Default is true.
%   @include_tbfl_native_cov: If true, also evaluate the inverse of TBFL's
%                             own segment-wise precision estimates.
%   @save_post_refit_paths: If true, save post-refit covariance paths in
%                           the MAT output.
%   @min_eig: Eigenvalue floor shared by PD-enforced post-refit methods.
%             Default is 0.01 to match the epsilon used elsewhere.
%   @delta_adaptive: Threshold multiplier for adaptive-threshold refits.
%   @rho_post: ADMM penalty parameter for EC2 and Xue-Ma-Zou refits.
%   @maxiter_post: Maximum ADMM iterations for EC2 and Xue-Ma-Zou refits.
%   @tol_post: ADMM convergence tolerance for EC2 and Xue-Ma-Zou refits.
%   @lambda_post: Lambda override for post-refit methods. NaN uses each
%                 method's default formula. A nonnegative scalar uses the
%                 same fixed lambda for every post-refit method. A numeric
%                 vector is interpreted in post_refit_methods order;
%                 missing, NaN, or negative entries fall back to each
%                 method's default formula. For convex_banding in the banded
%                 simulation, we uses fixed lambda_scale values to keep that
%                 benchmark reproducible.
%   @cache_dir: Optional directory for per-replicate GFlsL candidate caches.
%               Existing non-adaptive, stage-1, and adaptive candidates are
%               reused when the same tuning values are requested again. A
%               non-default metric or relaxation uses a metric-specific
%               subdirectory.
%   @termination_check_freq: Frequency for GFlsL stopping diagnostics. The
%                            default 1 checks every iteration; larger values
%                            reduce objective and feasibility checks.

    % Derive dimensions from input data
    [T, p, num_exps] = size(X);

    p_ = inputParser;
    addParameter(p_, 'mu2', 0.5);
    addParameter(p_, 'mu1', 0.8);
    addParameter(p_, 'kappa_', 0.5);
    addParameter(p_, 'tol', 1e-3);
    addParameter(p_, 'beta_', 0.1);
    addParameter(p_, 'penalty_scale', [1, 1, 1]);
    addParameter(p_, 'relaxation', 1.61);
    addParameter(p_, 'tau_BSOP', 10);
    addParameter(p_, 'delta_WBSIP', 5);
    addParameter(p_, 'tau_WBSIP', sqrt(p * log(T)) * 1.5);
    addParameter(p_, 'M_WBSIP', 100);
    addParameter(p_, 'gamma_DCDP', 100);
    addParameter(p_, 'zeta_DCDP', 0);
    addParameter(p_, 'Q_DCDP', 50);
    addParameter(p_, 'buffer_DCDP', 4);
    addParameter(p_, 'buffer_refine_DCDP', 4);
    addParameter(p_, 'filepre', "res");
    addParameter(p_, 'txt_filepre', "");
    addParameter(p_, 'mat_filepre', "");
    addParameter(p_, 'savetxt', 0);
    addParameter(p_, 'savemat', 0);
    addParameter(p_, 'save_candidate_thetas', true);
    addParameter(p_, 'showbar', 0);
    addParameter(p_, 'dates', 1:T);
    addParameter(p_, 'merge_breaks', false);
    addParameter(p_, 'DGP', '');
    addParameter(p_, 'prob', NaN);
    addParameter(p_, 'save_tbfl_omega', false);
    addParameter(p_, 'run_gflsl', true);
    addParameter(p_, 'run_post_refits', true);
    addParameter(p_, 'run_baselines', true);
    addParameter(p_, 'post_refit_methods', string.empty(1, 0));
    addParameter(p_, 'include_tbfl_native_cov', true);
    addParameter(p_, 'save_post_refit_paths', false);
    addParameter(p_, 'min_eig', 0.01);
    addParameter(p_, 'delta_adaptive', 2);
    addParameter(p_, 'rho_post', 2.5);
    addParameter(p_, 'maxiter_post', 500);
    addParameter(p_, 'tol_post', 1e-5);
    addParameter(p_, 'lambda_post', NaN);
    addParameter(p_, 'cache_dir', "");
    addParameter(p_, 'termination_check_freq', 1);
    addParameter(p_, 'gflsl_maxiter', inf);
    % Warm start: initialize each GFlsL fit from the converged state of a
    % neighboring lambda candidate. Stage-1 (serial over lamb2s) always
    % uses it when enabled; the non-adaptive and adaptive grids switch
    % from per-candidate parfor to chain-parallel warm starts when
    % warm_start_chains > 0. Results agree with cold starts up to tol.
    addParameter(p_, 'warm_start', false);
    addParameter(p_, 'warm_start_chains', 0);
    parse(p_, varargin{:});

    mu2 = p_.Results.mu2;
    mu1 = p_.Results.mu1;
    kappa_ = p_.Results.kappa_;
    tol = p_.Results.tol;
    beta_ = p_.Results.beta_;
    penalty_scale = p_.Results.penalty_scale;
    if ~isnumeric(penalty_scale) || numel(penalty_scale) ~= 3 || ...
            any(~isfinite(penalty_scale(:))) || any(penalty_scale(:) <= 0)
        error('get_two_stage_est:penaltyScale', ...
            'penalty_scale must contain three finite positive values.');
    end
    penalty_scale = reshape(penalty_scale, 1, 3);
    relaxation = p_.Results.relaxation;
    if ~isnumeric(relaxation) || ~isscalar(relaxation) || ...
            ~isfinite(relaxation) || relaxation <= 0 || ...
            relaxation >= (1 + sqrt(5)) / 2
        error('get_two_stage_est:relaxation', ...
            'relaxation must lie in (0, (1 + sqrt(5)) / 2).');
    end
    make_est = @(varargin) GFlsL(varargin{:}, ...
        penalty_scale=penalty_scale, relaxation=relaxation);
    tau_BSOP = p_.Results.tau_BSOP;
    delta_WBSIP = p_.Results.delta_WBSIP;
    tau_WBSIP = p_.Results.tau_WBSIP;
    M_WBSIP = p_.Results.M_WBSIP;
    gamma_DCDP = p_.Results.gamma_DCDP;
    zeta_DCDP = p_.Results.zeta_DCDP;
    Q_DCDP = p_.Results.Q_DCDP;
    buffer_DCDP = p_.Results.buffer_DCDP;
    buffer_refine_DCDP = p_.Results.buffer_refine_DCDP;
    filepre = string(p_.Results.filepre);
    txt_filepre = string(p_.Results.txt_filepre);
    mat_filepre = string(p_.Results.mat_filepre);
    if isempty(txt_filepre) || strlength(txt_filepre) == 0
        txt_filepre = filepre;
    end
    if isempty(mat_filepre) || strlength(mat_filepre) == 0
        mat_filepre = filepre;
    end
    savetxt = p_.Results.savetxt;
    savemat = p_.Results.savemat;
    save_candidate_thetas = savemat && ...
        logical(p_.Results.save_candidate_thetas);
    showbar = p_.Results.showbar;
    dates = ensure_datetime_array(p_.Results.dates);
    merge_breaks = logical(p_.Results.merge_breaks);
    DGP = p_.Results.DGP;
    prob = p_.Results.prob;
    save_tbfl_omega = logical(p_.Results.save_tbfl_omega);
    run_gflsl = logical(p_.Results.run_gflsl);
    run_post_refits = logical(p_.Results.run_post_refits);
    run_baselines = logical(p_.Results.run_baselines);
    has_test = ~isempty(X_test);
    post_refit_methods = unique(string(p_.Results.post_refit_methods), 'stable');
    post_refit_methods = post_refit_methods(strlength(post_refit_methods) > 0);
    include_tbfl_native_cov = logical(p_.Results.include_tbfl_native_cov);
    save_post_refit_paths = logical(p_.Results.save_post_refit_paths);
    post_refits_enabled = run_post_refits && ~isempty(post_refit_methods);
    lambda_post = p_.Results.lambda_post;
    cache_dir = string(p_.Results.cache_dir);
    if strlength(cache_dir) > 0 && ...
            (any(penalty_scale ~= [1, 1, 1]) || relaxation ~= 1.61)
        metric_key = sprintf('penalty_%s_%s_%s_relax_%s', ...
            lambda_key(penalty_scale(1)), lambda_key(penalty_scale(2)), ...
            lambda_key(penalty_scale(3)), lambda_key(relaxation));
        cache_dir = string(fullfile(cache_dir, metric_key));
    end
    termination_check_freq = p_.Results.termination_check_freq;
    gflsl_maxiter = p_.Results.gflsl_maxiter;
    warm_start = logical(p_.Results.warm_start);
    warm_start_chains = p_.Results.warm_start_chains;
    validate_lambda_post(lambda_post, post_refit_methods);

    % A helper function used to find the index of the first minimum element in
    % a given tensor.
    findMinIn = @(x)ind2sub(size(x), find(x == min(x, [], 'all'), 1));

    % Variant that ignores Inf/NaN entries. Metric arrays are initialized to
    % Inf, so grid extensions leave unevaluated candidates at Inf; the plain
    % argmin above would select those unevaluated slices. Multi-output call
    % sites invoke argmin_finite directly (anonymous handles cannot forward
    % multiple outputs).
    findMinValid = @(x) argmin_finite(x);

    if ~isempty(trueBreaks)
        num_breaks = length(trueBreaks{1});
    else
        num_breaks = NaN;
    end

    filename = mat_filepre + ".mat";
    use_cache = strlength(cache_dir) > 0 && ~savemat;
    if use_cache && exist(cache_dir, 'dir') ~= 7
        mkdir(cache_dir);
    end

    if savetxt
        txt_path = char(txt_filepre + ".txt");
        f = fopen(txt_path, 'a');
        if f < 0
            error('get_two_stage_est:openLogFailed', ...
                'Cannot open log file: %s', txt_path);
        end
    else
        f = 1; % standard output
    end

    % The number of candidates of two tuning parameters.
    len1 = length(lamb1s);
    len2 = length(lamb2s);
    num12 = len1 * len2;

    % Compute the combinations of the two numbers for parfor loop.
    [temp1, temp2] = meshgrid(1:len1, 1:len2);
    idxcomb = [temp1(:), temp2(:)];

    % Combinations for the second stage.
    len22 = length(lamb2s2);
    num122 = len1 * len22;
    [temp12, temp22] = meshgrid(1:len1, 1:len22);
    idxcomb2 = [temp12(:), temp22(:)];

    bb_ = T .^ (-kappa_);

    % Fixed IC sensitivity variants. beta_ic weights the explicit break
    % penalty and is unrelated to the solver parameter beta_. HBICG variants
    % reduce its edge penalty; HBIC variants strengthen penalties in regimes
    % where the original HBIC over-selects.
    fixed_hbic_names = ["HBIC_g0_b8", "HBIC_g2_b8", "HBIC_g4_b1"];
    fixed_hbic_gamma = [0, 2, 4];
    fixed_hbic_beta = [8, 8, 1];
    num_fixed_hbic = numel(fixed_hbic_names);

    fixed_hbicg_names = ["HBICG_g0_b2", "HBICG_g025_b1", ...
        "HBICG_g05_b1"];
    fixed_hbicg_gamma = [0, 0.25, 0.5];
    fixed_hbicg_beta = [2, 1, 1];
    num_fixed_hbicg = numel(fixed_hbicg_names);

    % The numbers of candidates of mu1 and mu2.
    lenmu1 = length(mu1);
    lenmu2 = length(mu2);

    % If showing processbar.
    if showbar && run_gflsl
        total = num12 + num122 * len2 * lenmu2 * lenmu1;
        
        % Create DataQueue instance for non-adaptive version.
        dq = parallel.pool.DataQueue;
        % Callback function: each worker tick updates a shared single-line bar.
        afterEach(dq, @(~) updateProgress(total));
    else
        % Create a dummy DataQueue to avoid undefined variable errors in parfor loops
        dq = [];
    end

    % The change points (cps) detected by BSOP.
    BScps = cell(num_exps, 1);
    % The HD of BSOP.
    HD_BS = zeros(num_exps, 1);

    % The change poitns (cps) detected by WBSIP.
    WBSIPcps = cell(num_exps, 1);
    % The HD of WBSIP.
    HD_WBSIP = zeros(num_exps, 1);

    % The change points (cps) detected by DCDP.
    DCDPcps = cell(num_exps, 1);
    % The HD of DCDP.
    HD_DCDP = nan(num_exps, 1);

    % The change points (cps) detected by TBFL.
    TBFLcps = cell(num_exps, 1);
    % The HD of TBFL.
    HD_TBFL = zeros(num_exps, 1);

    % TBFL precision matrices (Omega) for each regime, if save_tbfl_omega is true.
    % Stored as cell array where each cell contains a p x p x m_hat array.
    if save_tbfl_omega
        TBFL_Omega = cell(num_exps, 1);
    end

    % Delcare metrics.
    % HDs.
    HDs1 = inf(len2, num_exps);
    HDs2 = inf(num122, len2, lenmu2, lenmu1, num_exps);
    HDsn = inf(num12, num_exps);

    % F1s.
    F1s1 = inf(len2, num_exps);
    F1s2 = inf(num122, len2, lenmu2, lenmu1, num_exps);
    F1sn = inf(num12, num_exps);

    % Accuracy.
    accs1 = inf(len2, num_exps);
    accs2 = inf(num122, len2, lenmu2, lenmu1, num_exps);
    accsn = inf(num12, num_exps);

    % Error.
    errors1 = inf(len2, num_exps);
    errors2 = inf(num122, len2, lenmu2, lenmu1, num_exps);
    errorsn = inf(num12, num_exps);

    % AICs
    AIC1s = inf(len2, num_exps);
    AIC2s = inf(num122, len2, lenmu2, lenmu1, num_exps);
    AICns = inf(num12, num_exps);

    % BICs.
    BIC1s = inf(len2, num_exps);
    BIC2s = inf(num122, len2, lenmu2, lenmu1, num_exps);
    BICns = inf(num12, num_exps);

    % HBIC
    HBIC_1s = inf(len2, num_exps);
    HBICG_1s = inf(len2, num_exps);
    HBIC_2s = inf(num122, len2, lenmu2, lenmu1, num_exps);
    HBICG_2s = inf(num122, len2, lenmu2, lenmu1, num_exps);
    HBIC_ns = inf(num12, num_exps);
    HBICG_ns = inf(num12, num_exps);

    % lossvals
    lossval1s = inf(len2, num_exps);
    lossval2s = inf(num122, len2, lenmu2, lenmu1, num_exps);
    lossvalns = inf(num12, num_exps);

    % loss_
    loss1s = inf(len2, num_exps);
    loss2s = inf(num122, len2, lenmu2, lenmu1, num_exps);
    lossns = inf(num12, num_exps);
    
    % Estimators.
    EstBreaks1 = cell(len2, num_exps);
    EstBreaks2 = cell(num122, len2, lenmu2, lenmu1, num_exps);
    EstBreaksn = cell(num12, num_exps);

    % Theta tensors are only needed when MAT output is enabled.
    % For large grids, unconditional allocation of these arrays can consume
    % tens of gigabytes and trigger out-of-memory failures.
    if savemat && run_gflsl && save_candidate_thetas
        % Store Theta tensors in single precision from the beginning.
        % This reduces peak memory and avoids an extra full-size copy
        % created by end-of-function type conversion.
        Theta1s = zeros(p, p, T, len2, num_exps, 'single');
        Theta2s = zeros(p, p, T, num122, len2, lenmu2, lenmu1, num_exps, 'single');
        Thetans = zeros(p, p, T, num12, num_exps, 'single');
    else
        % Keep lightweight placeholders when MAT output is disabled.
        Theta1s = zeros(0, 0, 0, 0, 0, 'single');
        Theta2s = zeros(0, 0, 0, 0, 0, 0, 0, 0, 'single');
        Thetans = zeros(0, 0, 0, 0, 0, 'single');
    end

    % Flat post-refit containers indexed by (detector, method, experiment).
    if post_refits_enabled
        post_detectors = "BSOP";
        if has_test
            post_detectors = [post_detectors, "WBSIP"];
        end
        if has_test
            post_detectors = [post_detectors, "DCDP"];
        end
        post_detectors = [post_detectors, "TBFL"];

        post_methods = post_refit_methods;
        num_det = numel(post_detectors);
        num_met = numel(post_methods);

        post_HD  = zeros(num_det, num_met, num_exps);
        post_F1  = zeros(num_det, num_met, num_exps);
        post_acc = zeros(num_det, num_met, num_exps);
        post_err = zeros(num_det, num_met, num_exps);
        post_nb  = zeros(num_det, num_met, num_exps);
        post_paths = cell(num_det, num_met, num_exps);

        if include_tbfl_native_cov
            tbfl_native_HD    = zeros(num_exps, 1);
            tbfl_native_F1    = zeros(num_exps, 1);
            tbfl_native_acc   = zeros(num_exps, 1);
            tbfl_native_err   = zeros(num_exps, 1);
            tbfl_native_nbr   = zeros(num_exps, 1);
            tbfl_native_paths = cell(num_exps, 1);
        end
    else
        post_detectors = strings(1, 0);
        post_methods = strings(1, 0);
        num_det = 0;
        num_met = 0;
        post_HD = zeros(0, 0, num_exps);
        post_F1 = zeros(0, 0, num_exps);
        post_acc = zeros(0, 0, num_exps);
        post_err = zeros(0, 0, num_exps);
        post_nb = zeros(0, 0, num_exps);
    end

    % Metrics based on different criteria.
    resHFE2 = zeros(num_exps, 5);
    resloss2 = zeros(num_exps, 5);
    resAIC2 = zeros(num_exps, 5);
    resBIC2 = zeros(num_exps, 5);
    resHBIC_2 = zeros(num_exps, 5);
    resHBIC_fixed_2 = zeros(num_exps, 5, num_fixed_hbic);
    resHBICG_2 = zeros(num_exps, 5);
    resHBICG_fixed_2 = zeros(num_exps, 5, num_fixed_hbicg);
    resHBIC_c1_2 = zeros(num_exps, 5);
    resHBIC_c2_2 = zeros(num_exps, 5);

    resHFEn = zeros(num_exps, 5);
    reslossn = zeros(num_exps, 5);
    resAICn = zeros(num_exps, 5);
    resBICn = zeros(num_exps, 5);
    resHBIC_n = zeros(num_exps, 5);
    resHBIC_fixed_n = zeros(num_exps, 5, num_fixed_hbic);
    resHBICG_n = zeros(num_exps, 5);
    resHBICG_fixed_n = zeros(num_exps, 5, num_fixed_hbicg);
    resHBIC_c1_n = zeros(num_exps, 5);
    resHBIC_c2_n = zeros(num_exps, 5);

    % The variable saving the minimal lamb2 used in the first stage.
    min_lamb2s = lamb2s(1) * ones(num_exps, 1);
    stage1_eval_mask = false(len2, num_exps);
    stage1_empty_mask = false(len2, num_exps);
    stage1_stop_index = zeros(num_exps, 1);
    stage1_stop_lamb2 = nan(num_exps, 1);
    stage1_fallback_used = false(num_exps, 1);
    stage1_fallback_lamb2 = nan(num_exps, 1);

    % The loops.
    for tt = 1:num_exps
        
        if showbar && run_gflsl
            if isnan(num_breaks)
                m_label = '-';
            else
                m_label = sprintf('%d', num_breaks);
            end
            fprintf('\n  [%s, T=%d, p=%d, m=%s] %02d/%02d ▶ ', ...
                char(DGP), T, p, m_label, tt, num_exps);
            updateProgress(total, true);
        end
        
        % The competing method: BSOP.
        if run_baselines
            temp_BS = BS_cov(X(:, :, tt)', 0, T, tau_BSOP);
            if isempty(temp_BS.S)
                BScps{tt} = [];
            else
                BScps{tt} = sort(temp_BS.S);
            end
        end
        
        % The competing method: WBSIP.
        if run_baselines && has_test
            [Alpha, Beta] = intervals(M_WBSIP, 0, T);
            temp_WBSIP = WBSIP_cov(X(:, :, tt)', X_test(:, :, 1, tt)', ...
                0, T, Alpha, Beta, tau_WBSIP, delta_WBSIP);
            if isempty(temp_WBSIP.S)
                WBSIPcps{tt} = [];
            else
                WBSIPcps{tt} = sort(temp_WBSIP.S);
            end
        end

        % The competing method: DCDP.
        if run_baselines && has_test
            % Compute C_F from buffer_DCDP (matching GFDtL convention).
            C_F_DCDP = buffer_DCDP / (p * log(max(p, T)));

            % The competing method: DCDP.
            % If gamma_DCDP is a vector, use the held-out X_test samples to
            % select gamma.
            if length(gamma_DCDP) > 1
                [gamma_selected, ~] = dcdp_cv_gamma(X(:, :, tt), gamma_DCDP, ...
                    X_test=X_test(:, :, :, tt), model='ggm', zeta=zeta_DCDP, ...
                    Q=Q_DCDP, C_F=C_F_DCDP, buffer_refine=buffer_refine_DCDP);
            else
                gamma_selected = gamma_DCDP;
            end

            temp_DCDP = DCDP(X=X(:, :, tt), model='ggm', gamma=gamma_selected, ...
                zeta=zeta_DCDP, Q=Q_DCDP, C_F=C_F_DCDP, buffer_refine=buffer_refine_DCDP);
            temp_DCDP.run();
            if isempty(temp_DCDP.final_cps)
                DCDPcps{tt} = [];
            else
                DCDPcps{tt} = sort(temp_DCDP.final_cps);
            end
        end

        % The competing method: TBFL.
        if run_baselines
            temp_TBFL = TBFL(X=X(:, :, tt), disp_freq=inf);
            temp_TBFL.run();
            if isempty(temp_TBFL.breaks)
                TBFLcps{tt} = [];
            else
                TBFLcps{tt} = sort(temp_TBFL.breaks(:));
            end

            % Optionally compute and save TBFL precision matrices using
            % EstimateOmega. The same quantity is also needed when TBFL's
            % native precision-to-covariance baseline is requested.
            if save_tbfl_omega || (post_refits_enabled && include_tbfl_native_cov)
                [Omega_hat, ~] = temp_TBFL.EstimateOmega();
                if save_tbfl_omega
                    TBFL_Omega{tt} = Omega_hat;
                end
            end
        end
        
        % Compute HD.
        if ~isempty(trueBreaks)
            if isempty(trueBreaks{tt})
                if isempty(BScps{tt})
                    HD_BS(tt) = 0;
                else
                    HD_BS(tt) = max(BScps{tt});
                end

                if has_test
                    if isempty(WBSIPcps{tt})
                        HD_WBSIP(tt) = 0;
                    else
                        HD_WBSIP(tt) = max(WBSIPcps{tt});
                    end
                end

                if has_test
                    if isempty(DCDPcps{tt})
                        HD_DCDP(tt) = 0;
                    else
                        HD_DCDP(tt) = max(DCDPcps{tt});
                    end
                end

                if isempty(TBFLcps{tt})
                    HD_TBFL(tt) = 0;
                else
                    HD_TBFL(tt) = max(TBFLcps{tt});
                end
            else
                HD_BS(tt) = compute_HD(trueBreaks{tt}, BScps{tt});

                if has_test
                    HD_WBSIP(tt) = compute_HD(trueBreaks{tt}, WBSIPcps{tt});
                end

                if has_test
                    HD_DCDP(tt) = compute_HD(trueBreaks{tt}, DCDPcps{tt});
                end

                HD_TBFL(tt) = compute_HD(trueBreaks{tt}, TBFLcps{tt});
            end
        end
        
        % Per-experiment slices of the big broadcast arrays. Referencing
        % these inside the parfor loops below keeps the workers from
        % receiving the full num_exps-sized arrays.
        X_tt = X(:, :, tt);
        if has_test
            X_test_tt = X_test(:, :, :, tt);
        else
            X_test_tt = [];
        end
        if ~isempty(trueTheta)
            trueTheta_tt = trueTheta(:, :, :, tt);
        else
            trueTheta_tt = [];
        end
        if ~isempty(trueBreaks)
            trueBreaks_tt = trueBreaks{tt};
        else
            trueBreaks_tt = [];
        end

        % Apply covariance post-refits to the competing break detectors
        % using the same break estimates already computed above.
        if post_refits_enabled

            opts_post = struct( ...
                'center', true, ...
                'min_eig', p_.Results.min_eig, ...
                'lambda', NaN, ...
                'lambda_scale', 1, ...
                'delta', p_.Results.delta_adaptive, ...
                'rho', p_.Results.rho_post, ...
                'mu', p_.Results.rho_post, ...
                'maxiter', p_.Results.maxiter_post, ...
                'tol', p_.Results.tol_post);

            for dd = 1:num_det
                switch post_detectors(dd)
                    case "BSOP";  breaks_tt = BScps{tt};
                    case "WBSIP"; breaks_tt = WBSIPcps{tt};
                    case "DCDP";  breaks_tt = DCDPcps{tt};
                    case "TBFL";  breaks_tt = TBFLcps{tt};
                end

                for mm = 1:num_met
                    opts_mm = opts_post;
                    opts_mm.lambda = lambda_for_method(lambda_post, mm);
                    opts_mm = apply_fixed_banded_convex_banding_settings( ...
                        opts_mm, post_methods(mm), DGP, p);

                    Sigma_post = post_covariance_refit(X_tt, breaks_tt, ...
                        post_methods(mm), opts_mm);

                    if ~isempty(trueTheta) && ~isempty(trueBreaks)
                        [post_HD(dd, mm, tt), post_F1(dd, mm, tt), ...
                            post_acc(dd, mm, tt), post_err(dd, mm, tt), ...
                            post_nb(dd, mm, tt)] = evaluate_covariance_refit( ...
                            Sigma_post, breaks_tt, trueTheta_tt, ...
                            trueBreaks_tt);
                    end

                    if save_post_refit_paths
                        post_paths{dd, mm, tt} = Sigma_post;
                    end
                end
            end

            if include_tbfl_native_cov && run_baselines
                Sigma_tbfl_native = invert_precision_segments(Omega_hat, T, TBFLcps{tt}, ...
                    min_eig=p_.Results.min_eig);

                if ~isempty(trueTheta) && ~isempty(trueBreaks)
                    [tbfl_native_HD(tt), tbfl_native_F1(tt), ...
                        tbfl_native_acc(tt), tbfl_native_err(tt), ...
                        tbfl_native_nbr(tt)] = evaluate_covariance_refit( ...
                        Sigma_tbfl_native, TBFLcps{tt}, ...
                        trueTheta_tt, trueBreaks_tt);
                end

                if save_post_refit_paths
                    tbfl_native_paths{tt} = Sigma_tbfl_native;
                end
            end
        end

        if run_gflsl
            % For the nonadaptive version.
            missing_n = true(num12, 1);
            for ii = 1:num12
                if use_cache
                    cache_file = candidate_cache_file(cache_dir, tt, ...
                        "nonadaptive", lamb1s(idxcomb(ii, 1)), ...
                        lamb2s(idxcomb(ii, 2)));
                    cached = load_candidate_cache(cache_file);
                    if ~isempty(cached)
                        [HDsn(ii, tt), F1sn(ii, tt), accsn(ii, tt), ...
                            errorsn(ii, tt), AICns(ii, tt), BICns(ii, tt), ...
                            HBIC_ns(ii, tt), HBICG_ns(ii, tt), ...
                            lossvalns(ii, tt), lossns(ii, tt), ...
                            EstBreaksn{ii, tt}] = unpack_candidate_cache(cached);
                        missing_n(ii) = false;
                    end
                end
            end

            missing_idx = find(missing_n);
            n_results = cell(numel(missing_idx), 1);
            if warm_start && warm_start_chains > 0 && numel(missing_idx) > 0
                % Chain-parallel warm start: partition the missing
                % candidates into chains, each chain fitted sequentially
                % with every candidate initialized from the previous one.
                chains_n = build_warm_chains(missing_idx, idxcomb, ...
                    warm_start_chains);
                chain_results = cell(numel(chains_n), 1);
                parfor cc = 1:numel(chains_n)
                    chain_results{cc} = run_warm_chain( ...
                        chains_n{cc}, X_tt, lamb1s, lamb2s, idxcomb, ...
                        beta_, tol, gflsl_maxiter, merge_breaks, ...
                        termination_check_freq, make_est, X_test_tt, ...
                        trueTheta_tt, trueBreaks_tt, has_test, ...
                        save_candidate_thetas, [], [], []);
                    end
                for cc = 1:numel(chains_n)
                    for ee = 1:numel(chains_n{cc})
                        % chains hold global candidate indices, while
                        % n_results is indexed by position within
                        % missing_idx, matching the parfor branch and the
                        % unpacking loop below.
                        pos = find(missing_idx == chains_n{cc}(ee), 1);
                        n_results{pos} = chain_results{cc}{ee};
                    end
                end
            else
                parfor jj = 1:numel(missing_idx)
                    ii = missing_idx(jj);
                    Estn = make_est(X=X_tt, lamb1=lamb1s(idxcomb(ii, 1)), ...
                        lamb2=lamb2s(idxcomb(ii, 2)), beta_=beta_, epsilon=0.01, ...
                        tol=tol, maxiter=gflsl_maxiter, disp_freq=inf, ...
                        merge_breaks=merge_breaks, ...
                        termination_check_freq=termination_check_freq);
                    Estn.run
                    n_results{jj} = estimator_cache_payload(Estn, X_test_tt, ...
                        trueTheta_tt, trueBreaks_tt, has_test, ...
                        save_candidate_thetas);
                    if showbar && ~isempty(dq)
                        send(dq, 1);
                    end
                end
            end

            for jj = 1:numel(missing_idx)
                ii = missing_idx(jj);
                payload = n_results{jj};
                [HDsn(ii, tt), F1sn(ii, tt), accsn(ii, tt), ...
                    errorsn(ii, tt), AICns(ii, tt), BICns(ii, tt), ...
                    HBIC_ns(ii, tt), HBICG_ns(ii, tt), ...
                    lossvalns(ii, tt), lossns(ii, tt), ...
                    EstBreaksn{ii, tt}] = unpack_candidate_cache(payload);
                if save_candidate_thetas && isfield(payload, 'Theta_k')
                    Thetans(:, :, :, ii, tt) = single(payload.Theta_k);
                end
                if use_cache
                    cache_file = candidate_cache_file(cache_dir, tt, ...
                        "nonadaptive", lamb1s(idxcomb(ii, 1)), ...
                        lamb2s(idxcomb(ii, 2)));
                    save_candidate_cache(cache_file, payload);
                end
            end

            % For the two-stage experiments.
            stage1_prev_state = [];
            for k = 1:len2
                Est1 = [];
                stage1_cached = [];
                if use_cache
                    stage1_cached = load_candidate_cache( ...
                        candidate_cache_file(cache_dir, tt, "stage1", 0, ...
                        lamb2s(k)));
                end

                if isempty(stage1_cached)
                    if warm_start && ~isempty(stage1_prev_state)
                        Est1 = make_est(X=X_tt, lamb1=0, lamb2=lamb2s(k), beta_=beta_, ...
                            epsilon=0.01, tol=tol, maxiter=gflsl_maxiter, ...
                            disp_freq=inf, merge_breaks=merge_breaks, ...
                            termination_check_freq=termination_check_freq, ...
                            init_state=stage1_prev_state);
                    else
                        Est1 = make_est(X=X_tt, lamb1=0, lamb2=lamb2s(k), beta_=beta_, ...
                            epsilon=0.01, tol=tol, maxiter=gflsl_maxiter, ...
                            disp_freq=inf, merge_breaks=merge_breaks, ...
                            termination_check_freq=termination_check_freq);
                    end
                    Est1.run
                    stage1_prev_state = Est1.solver_state();
                    stage1_payload = estimator_cache_payload(Est1, X_test_tt, ...
                        trueTheta_tt, trueBreaks_tt, has_test, true);
                    stage1_payload.Theta_k = single(Est1.Theta_k);
                    stage1_payload.norm_diff = Est1.norm_diff;
                    if use_cache
                        save_candidate_cache(candidate_cache_file(cache_dir, ...
                            tt, "stage1", 0, lamb2s(k)), stage1_payload);
                    end
                else
                    stage1_payload = stage1_cached;
                end

                stage1_eval_mask(k, tt) = true;
                stage1_empty_mask(k, tt) = isempty(stage1_payload.EstBreaks);

                if isempty(stage1_payload.EstBreaks)
                    if k == 1
                        % Compute step size for lambda2 search.
                        if length(lamb2s) > 1
                            step_size = abs(lamb2s(2) - lamb2s(1));
                        else
                            % Use 10% of lamb2s as default step for single-value grids.
                            step_size = abs(lamb2s(1)) * 0.1;
                        end

                        % Enforce a minimum step so the loop advances.
                        step_size = max(step_size, 1e-6);

                        % If lamb2s(1) is non-positive, there is no smaller
                        % positive value to test, so skip the fallback loop.
                        if lamb2s(1) > 0
                            lamb2_temp = lamb2s(1) - step_size;
                            max_iters = 1000;
                            iter = 0;
                            while lamb2_temp > 0 && iter < max_iters
                                Est1 = make_est(X=X_tt, lamb1=0, lamb2=lamb2_temp, beta_=beta_, ...
                                    epsilon=0.01, tol=tol, maxiter=gflsl_maxiter, ...
                                    disp_freq=inf, merge_breaks=merge_breaks, ...
                                    termination_check_freq=termination_check_freq);
                                Est1.run;
                                stage1_prev_state = Est1.solver_state();
                                if ~isempty(Est1.EstBreaks)
                                    min_lamb2s(tt) = lamb2_temp;
                                    stage1_fallback_used(tt) = true;
                                    stage1_fallback_lamb2(tt) = lamb2_temp;
                                    stage1_payload = estimator_cache_payload( ...
                                        Est1, X_test_tt, trueTheta_tt, ...
                                        trueBreaks_tt, has_test, true);
                                    stage1_payload.norm_diff = Est1.norm_diff;
                                    break
                                end
                                lamb2_temp = lamb2_temp - step_size;
                                iter = iter + 1;
                            end
                        end
                    else
                        stage1_stop_index(tt) = k;
                        stage1_stop_lamb2(tt) = lamb2s(k);
                        break
                    end
                end

                if save_candidate_thetas
                    Theta1s(:, :, :, k, tt) = single(stage1_payload.Theta_k);
                end
                [HDs1(k, tt), F1s1(k, tt), accs1(k, tt), errors1(k, tt), ...
                    AIC1s(k, tt), BIC1s(k, tt), HBIC_1s(k, tt), ...
                    HBICG_1s(k, tt), lossval1s(k, tt), loss1s(k, tt), ...
                    EstBreaks1{k, tt}] = unpack_candidate_cache(stage1_payload);

                for idx_mu2 = 1:lenmu2
                    for idx_mu1 = 1:lenmu1
                        % Build adaptive weights for the current experiment only.
                        % This avoids storing large xi1/xi2 tensors across all
                        % experiments, which are not needed after this loop body.
                        xi1_current = max(abs(double(stage1_payload.Theta_k)), bb_) .^ -mu1(idx_mu1);
                        xi2_current = max(stage1_payload.norm_diff, bb_) .^ -mu2(idx_mu2);

                        missing_a = true(num122, 1);
                        for ii = 1:num122
                            if use_cache
                                cache_file = candidate_cache_file(cache_dir, ...
                                    tt, "adaptive", lamb1s(idxcomb2(ii, 1)), ...
                                    lamb2s(k), lamb2s2(idxcomb2(ii, 2)), ...
                                    mu1(idx_mu1), mu2(idx_mu2));
                                cached = load_candidate_cache(cache_file);
                                if ~isempty(cached)
                                    [HDs2(ii, k, idx_mu2, idx_mu1, tt), ...
                                        F1s2(ii, k, idx_mu2, idx_mu1, tt), ...
                                        accs2(ii, k, idx_mu2, idx_mu1, tt), ...
                                        errors2(ii, k, idx_mu2, idx_mu1, tt), ...
                                        AIC2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                        BIC2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                        HBIC_2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                        HBICG_2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                        lossval2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                        loss2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                        EstBreaks2{ii, k, idx_mu2, idx_mu1, tt}] = ...
                                            unpack_candidate_cache(cached);
                                    missing_a(ii) = false;
                                end
                            end
                        end

                        missing_idx = find(missing_a);
                        a_results = cell(numel(missing_idx), 1);
                        if warm_start && warm_start_chains > 0 && numel(missing_idx) > 0
                            chains_a = build_warm_chains(missing_idx, idxcomb2, ...
                                warm_start_chains);
                            chain_results = cell(numel(chains_a), 1);
                            parfor cc = 1:numel(chains_a)
                                chain_results{cc} = run_warm_chain( ...
                                    chains_a{cc}, X_tt, lamb1s, lamb2s2, idxcomb2, ...
                                    beta_, tol, gflsl_maxiter, merge_breaks, ...
                                    termination_check_freq, make_est, X_test_tt, ...
                                    trueTheta_tt, trueBreaks_tt, has_test, ...
                                    save_candidate_thetas, xi1_current, ...
                                    xi2_current, stage1_prev_state);
                            end
                            for cc = 1:numel(chains_a)
                                for ee = 1:numel(chains_a{cc})
                                    % See the non-adaptive branch: chains
                                    % carry global indices, a_results is
                                    % indexed by position in missing_idx.
                                    pos = find(missing_idx == chains_a{cc}(ee), 1);
                                    a_results{pos} = chain_results{cc}{ee};
                                end
                            end
                        else
                            parfor jj = 1:numel(missing_idx)
                                ii = missing_idx(jj);
                                Est2 = make_est(X=X_tt, ...
                                    lamb1=lamb1s(idxcomb2(ii, 1)) .* xi1_current, ...
                                    lamb2=lamb2s2(idxcomb2(ii, 2)) .* reshape(xi2_current, 1, []), ...
                                    beta_=beta_, epsilon=0.01, tol=tol, maxiter=gflsl_maxiter, ...
                                    disp_freq=inf, merge_breaks=merge_breaks, ...
                                    termination_check_freq=termination_check_freq);
                                Est2.run

                                a_results{jj} = estimator_cache_payload(Est2, ...
                                    X_test_tt, trueTheta_tt, trueBreaks_tt, ...
                                    has_test, save_candidate_thetas);
                                if showbar && ~isempty(dq)
                                    send(dq, 1);
                                end
                            end
                        end

                        for jj = 1:numel(missing_idx)
                            ii = missing_idx(jj);
                            payload = a_results{jj};
                            [HDs2(ii, k, idx_mu2, idx_mu1, tt), ...
                                F1s2(ii, k, idx_mu2, idx_mu1, tt), ...
                                accs2(ii, k, idx_mu2, idx_mu1, tt), ...
                                errors2(ii, k, idx_mu2, idx_mu1, tt), ...
                                AIC2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                BIC2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                HBIC_2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                HBICG_2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                lossval2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                loss2s(ii, k, idx_mu2, idx_mu1, tt), ...
                                EstBreaks2{ii, k, idx_mu2, idx_mu1, tt}] = ...
                                    unpack_candidate_cache(payload);
                            if save_candidate_thetas && isfield(payload, 'Theta_k')
                                Theta2s(:, :, :, ii, k, idx_mu2, idx_mu1, tt) = ...
                                    single(payload.Theta_k);
                            end
                            if use_cache
                                cache_file = candidate_cache_file(cache_dir, ...
                                    tt, "adaptive", lamb1s(idxcomb2(ii, 1)), ...
                                    lamb2s(k), lamb2s2(idxcomb2(ii, 2)), ...
                                    mu1(idx_mu1), mu2(idx_mu2));
                                save_candidate_cache(cache_file, payload);
                            end
                        end
                    end
                end
            end

            if ~isempty(trueTheta) && ~isempty(trueBreaks)
                % Restrict selections to candidates with valid truth-comparison
                % metrics. Stage-1's no-break stop can still run stage-2 fits at
                % the stop lamb2 (those candidates get finite loss/criteria but
                % Inf HD/F1/nbreak); selecting them produces null-metric rows.
                % The F1 array is the reliable evaluated marker: it is finite
                % iff the candidate produced a valid comparison against truth.
                evalmask2 = isfinite(F1s2(:, :, :, :, tt));
                AIC2s(:, :, :, :, tt)   = local_mask_unevaluated(AIC2s(:, :, :, :, tt), evalmask2);
                BIC2s(:, :, :, :, tt)   = local_mask_unevaluated(BIC2s(:, :, :, :, tt), evalmask2);
                HBIC_2s(:, :, :, :, tt) = local_mask_unevaluated(HBIC_2s(:, :, :, :, tt), evalmask2);
                HBICG_2s(:, :, :, :, tt)= local_mask_unevaluated(HBICG_2s(:, :, :, :, tt), evalmask2);
                if has_test
                    loss2s(:, :, :, :, tt) = local_mask_unevaluated(loss2s(:, :, :, :, tt), evalmask2);
                end
                evalmaskn = isfinite(F1sn(:, tt));
                AICns(:, tt)   = local_mask_unevaluated(AICns(:, tt), evalmaskn);
                BICns(:, tt)   = local_mask_unevaluated(BICns(:, tt), evalmaskn);
                HBIC_ns(:, tt) = local_mask_unevaluated(HBIC_ns(:, tt), evalmaskn);
                HBICG_ns(:, tt)= local_mask_unevaluated(HBICG_ns(:, tt), evalmaskn);
                if has_test
                    lossns(:, tt) = local_mask_unevaluated(lossns(:, tt), evalmaskn);
                end

                resHFE2(tt, :) = two_stage_HFE(HDs2(:, :, :, :, tt), ...
                    F1s2(:, :, :, :, tt), errors2(:, :, :, :, tt), ...
                    accs2(:, :, :, :, tt), EstBreaks2(:, :, :, :, tt));
                resHFEn(tt, :) = two_stage_HFE(HDsn(:, tt), F1sn(:, tt), errorsn(:, tt), ...
                    accsn(:, tt), EstBreaksn(:, tt));

                if has_test
                    [i1, i2, i3, i4] = argmin_finite(loss2s(:, :, :, :, tt));
                    resloss2(tt, :) = [HDs2(i1, i2, i3, i4, tt), F1s2(i1, i2, i3, i4, tt), ...
                        errors2(i1, i2, i3, i4, tt), accs2(i1, i2, i3, i4, tt), ...
                        length(EstBreaks2{i1, i2, i3, i4, tt})];

                    jj = findMinValid(lossns(:, tt));
                    reslossn(tt, :) = [HDsn(jj, tt), F1sn(jj, tt), errorsn(jj, tt), ...
                        accsn(jj, tt), length(EstBreaksn{jj, tt})];
                end

                [i1, i2, i3, i4] = argmin_finite(AIC2s(:, :, :, :, tt));
                resAIC2(tt, :) = [HDs2(i1, i2, i3, i4, tt), F1s2(i1, i2, i3, i4, tt), ...
                    errors2(i1, i2, i3, i4, tt), accs2(i1, i2, i3, i4, tt), ...
                    length(EstBreaks2{i1, i2, i3, i4, tt})];

                jj = findMinValid(AICns(:, tt));
                resAICn(tt, :) = [HDsn(jj, tt), F1sn(jj, tt), errorsn(jj, tt), ...
                    accsn(jj, tt), length(EstBreaksn{jj, tt})];

                [i1, i2, i3, i4] = argmin_finite(BIC2s(:, :, :, :, tt));
                resBIC2(tt, :) = [HDs2(i1, i2, i3, i4, tt), F1s2(i1, i2, i3, i4, tt), ...
                    errors2(i1, i2, i3, i4, tt), accs2(i1, i2, i3, i4, tt), ...
                    length(EstBreaks2{i1, i2, i3, i4, tt})];

                jj = findMinValid(BICns(:, tt));
                resBICn(tt, :) = [HDsn(jj, tt), F1sn(jj, tt), errorsn(jj, tt), ...
                    accsn(jj, tt), length(EstBreaksn{jj, tt})];

                [i1, i2, i3, i4] = argmin_finite(HBIC_2s(:, :, :, :, tt));
                resHBIC_2(tt, :) = [HDs2(i1, i2, i3, i4, tt), F1s2(i1, i2, i3, i4, tt), ...
                    errors2(i1, i2, i3, i4, tt), accs2(i1, i2, i3, i4, tt), ...
                    length(EstBreaks2{i1, i2, i3, i4, tt})];

                jj = findMinValid(HBIC_ns(:, tt));
                resHBIC_n(tt, :) = [HDsn(jj, tt), F1sn(jj, tt), errorsn(jj, tt), ...
                    accsn(jj, tt), length(EstBreaksn{jj, tt})];

                [i1, i2, i3, i4] = argmin_finite(HBICG_2s(:, :, :, :, tt));
                resHBICG_2(tt, :) = [HDs2(i1, i2, i3, i4, tt), F1s2(i1, i2, i3, i4, tt), ...
                    errors2(i1, i2, i3, i4, tt), accs2(i1, i2, i3, i4, tt), ...
                    length(EstBreaks2{i1, i2, i3, i4, tt})];

                jj = findMinValid(HBICG_ns(:, tt));
                resHBICG_n(tt, :) = [HDsn(jj, tt), F1sn(jj, tt), errorsn(jj, tt), ...
                    accsn(jj, tt), length(EstBreaksn{jj, tt})];

                aT = log(T) * p / T;
                cT = log(T) * log(p) / T;

                % Fixed non-oracle IC penalty variants.
                for vv = 1:num_fixed_hbic
                    resHBIC_fixed_2(tt, :, vv) = hbic_penalty_select( ...
                        HBIC_2s(:, :, :, :, tt), lossval2s(:, :, :, :, tt), ...
                        EstBreaks2(:, :, :, :, tt), HDs2(:, :, :, :, tt), ...
                        F1s2(:, :, :, :, tt), errors2(:, :, :, :, tt), ...
                        accs2(:, :, :, :, tt), aT, cT, ...
                        fixed_hbic_gamma(vv), fixed_hbic_beta(vv));
                    resHBIC_fixed_n(tt, :, vv) = hbic_penalty_select( ...
                        HBIC_ns(:, tt), lossvalns(:, tt), EstBreaksn(:, tt), ...
                        HDsn(:, tt), F1sn(:, tt), errorsn(:, tt), ...
                        accsn(:, tt), aT, cT, fixed_hbic_gamma(vv), ...
                        fixed_hbic_beta(vv));
                end

                for vv = 1:num_fixed_hbicg
                    resHBICG_fixed_2(tt, :, vv) = hbicg_penalty_select( ...
                        HBICG_2s(:, :, :, :, tt), HBIC_2s(:, :, :, :, tt), ...
                        lossval2s(:, :, :, :, tt), EstBreaks2(:, :, :, :, tt), ...
                        HDs2(:, :, :, :, tt), F1s2(:, :, :, :, tt), ...
                        errors2(:, :, :, :, tt), accs2(:, :, :, :, tt), ...
                        aT, cT, fixed_hbicg_gamma(vv), fixed_hbicg_beta(vv));
                    resHBICG_fixed_n(tt, :, vv) = hbicg_penalty_select( ...
                        HBICG_ns(:, tt), HBIC_ns(:, tt), lossvalns(:, tt), ...
                        EstBreaksn(:, tt), HDsn(:, tt), F1sn(:, tt), ...
                        errorsn(:, tt), accsn(:, tt), aT, cT, ...
                        fixed_hbicg_gamma(vv), fixed_hbicg_beta(vv));
                end

                % --- Plug-in penalty calibration (c_est) ---
                % Compute c_est from the candidate loss landscape, then
                % rescale the HBIC break penalty by c_est and 2*c_est.

                % Adaptive
                [resHBIC_c1_2(tt,:), resHBIC_c2_2(tt,:)] = ...
                    c_est_select(HBIC_2s(:,:,:,:,tt), ...
                    lossval2s(:,:,:,:,tt), EstBreaks2(:,:,:,:,tt), ...
                    HDs2(:,:,:,:,tt), F1s2(:,:,:,:,tt), ...
                    errors2(:,:,:,:,tt), accs2(:,:,:,:,tt), aT, cT);

                % Non-adaptive
                [resHBIC_c1_n(tt,:), resHBIC_c2_n(tt,:)] = ...
                    c_est_select(HBIC_ns(:,tt), ...
                    lossvalns(:,tt), EstBreaksn(:,tt), ...
                    HDsn(:,tt), F1sn(:,tt), ...
                    errorsn(:,tt), accsn(:,tt), aT, cT);
            end
        end
    end

    if ~isempty(trueTheta) && ~isempty(trueBreaks)
        % Write experiment settings header before results
        fprintf(f, "\n\n");
        fprintf(f, '%s\n', repmat('═', 1, 70));
        if ~isempty(DGP)
            % Format settings line
            if isnan(prob)
                fprintf(f, 'DGP=%s | T=%d | p=%d | m=%d | num_exps=%d\n', ...
                    DGP, T, p, num_breaks, num_exps);
            else
                fprintf(f, 'DGP=%s | T=%d | p=%d | m=%d | prob=%.1f | num_exps=%d\n', ...
                    DGP, T, p, num_breaks, prob, num_exps);
            end
        else
            fprintf(f, 'T=%d | p=%d | num_exps=%d\n', T, p, num_exps);
        end
        fprintf(f, '%s\n', repmat('─', 1, 70));

        % Summaries printed in a boxed table for readability.
        bs_rows = {
            'BSOP',  mean(HD_BS),  NaN, NaN, NaN, mean(cellfun(@length, BScps));
            'TBFL',  mean(HD_TBFL),  NaN, NaN, NaN, mean(cellfun(@length, TBFLcps));
            };
        if has_test
            bs_rows(end+1, :) = {'WBSIP', mean(HD_WBSIP), NaN, NaN, NaN, mean(cellfun(@length, WBSIPcps))};
        end
        if has_test
            bs_rows(end+1, :) = {'DCDP', mean(HD_DCDP), NaN, NaN, NaN, mean(cellfun(@length, DCDPcps))};
        end

        combined_rows = [
            build_section_row('Competing baselines');
            bs_rows
            ];
        if run_gflsl
            adaptive_rows = [
                build_metric_row('HFE',  mean(resHFE2, 1));
                build_metric_row('AIC',  mean(resAIC2, 1));
                build_metric_row('BIC',  mean(resBIC2, 1));
                build_metric_row('HBIC', mean(resHBIC_2, 1));
                fixed_metric_table_rows(fixed_hbic_names, resHBIC_fixed_2);
                build_metric_row('HBICG', mean(resHBICG_2, 1));
                fixed_metric_table_rows(fixed_hbicg_names, resHBICG_fixed_2);
                build_metric_row('HBIC_c1', mean(resHBIC_c1_2, 1));
                build_metric_row('HBIC_c2', mean(resHBIC_c2_2, 1));
                ];
            if has_test
                adaptive_rows = [build_metric_row('loss', mean(resloss2, 1)); adaptive_rows];
            end

            nonadaptive_rows = [
                build_metric_row('HFE',  mean(resHFEn, 1));
                build_metric_row('AIC',  mean(resAICn, 1));
                build_metric_row('BIC',  mean(resBICn, 1));
                build_metric_row('HBIC', mean(resHBIC_n, 1));
                fixed_metric_table_rows(fixed_hbic_names, resHBIC_fixed_n);
                build_metric_row('HBICG', mean(resHBICG_n, 1));
                fixed_metric_table_rows(fixed_hbicg_names, resHBICG_fixed_n);
                build_metric_row('HBIC_c1', mean(resHBIC_c1_n, 1));
                build_metric_row('HBIC_c2', mean(resHBIC_c2_n, 1));
                ];
            if has_test
                nonadaptive_rows = [build_metric_row('loss', mean(reslossn, 1)); nonadaptive_rows];
            end

            combined_rows = [
                combined_rows;
                build_section_row('Adaptive version');
                adaptive_rows;
                build_section_row('Non-adaptive version');
                nonadaptive_rows
                ];
        end
        % Table section titles are embedded in combined_rows.
        display_results_table(f, combined_rows);

        if post_refits_enabled
            post_rows = cell(0, 6);
            for dd = 1:num_det
                post_rows(end + 1, :) = { ...
                    sprintf('%s post-refits', post_detectors(dd)), ...
                    nan, nan, nan, nan, nan}; %#ok<AGROW>
                for mm = 1:num_met
                    post_rows(end + 1, :) = {pretty_covariance_method_name(post_methods(mm)), ...
                        mean(post_HD(dd, mm, :),  'omitnan'), ...
                        mean(post_F1(dd, mm, :),  'omitnan'), ...
                        mean(post_err(dd, mm, :), 'omitnan'), ...
                        mean(post_acc(dd, mm, :), 'omitnan'), ...
                        mean(post_nb(dd, mm, :), 'omitnan')}; %#ok<AGROW>
                end
                if post_detectors(dd) == "TBFL" && include_tbfl_native_cov
                    post_rows(end + 1, :) = {'TBFL-native', ...
                        mean(tbfl_native_HD,  'omitnan'), ...
                        mean(tbfl_native_F1,  'omitnan'), ...
                        mean(tbfl_native_err, 'omitnan'), ...
                        mean(tbfl_native_acc, 'omitnan'), ...
                        mean(tbfl_native_nbr, 'omitnan')}; %#ok<AGROW>
                end
            end
            display_results_table(f, post_rows);
        end
    end

    if savetxt
        fclose(f);
    end

    if savemat
        % Save only the core outputs explicitly. This avoids dumping parser
        % objects and loop temporaries into the MAT file.
        vars_to_save = {'X', 'X_test', 'trueTheta', 'trueBreaks', ...
            'DGP', 'prob', 'num_breaks', ...
            'T', 'p', 'num_exps', 'lamb1s', 'lamb2s', 'lamb2s2', ...
            'mu1', 'mu2', 'kappa_', 'tol', 'beta_', 'penalty_scale', ...
            'relaxation', 'termination_check_freq', ...
            'tau_BSOP', 'delta_WBSIP', 'tau_WBSIP', 'M_WBSIP', ...
            'gamma_DCDP', 'zeta_DCDP', 'Q_DCDP', ...
            'buffer_DCDP', 'buffer_refine_DCDP', ...
            'merge_breaks', 'run_gflsl', 'run_post_refits', 'lambda_post', ...
            'save_candidate_thetas', ...
            'BScps', 'HD_BS', 'WBSIPcps', 'HD_WBSIP', ...
            'DCDPcps', 'HD_DCDP', 'TBFLcps', 'HD_TBFL'};

        if save_tbfl_omega
            vars_to_save{end + 1} = 'TBFL_Omega';
        end

        if run_gflsl
            vars_to_save = [vars_to_save, { ...
                'AIC1s', 'AIC2s', 'AICns', ...
                'BIC1s', 'BIC2s', 'BICns', ...
                'HBIC_1s', 'HBIC_2s', 'HBIC_ns', ...
                'HBICG_1s', 'HBICG_2s', 'HBICG_ns', ...
                'lossval1s', 'lossval2s', 'lossvalns', ...
                'loss1s', 'loss2s', 'lossns', ...
                'HDs1', 'HDs2', 'HDsn', ...
                'F1s1', 'F1s2', 'F1sn', ...
                'accs1', 'accs2', 'accsn', ...
                'errors1', 'errors2', 'errorsn', ...
                'EstBreaks1', 'EstBreaks2', 'EstBreaksn', ...
                'resHFE2', 'resloss2', 'resAIC2', 'resBIC2', ...
                'resHBIC_2', 'resHBIC_fixed_2', ...
                'fixed_hbic_names', 'fixed_hbic_gamma', 'fixed_hbic_beta', ...
                'resHBICG_2', 'resHBICG_fixed_2', ...
                'fixed_hbicg_names', 'fixed_hbicg_gamma', 'fixed_hbicg_beta', ...
                'resHBIC_c1_2', 'resHBIC_c2_2', ...
                'resHFEn', 'reslossn', 'resAICn', 'resBICn', ...
                'resHBIC_n', 'resHBIC_fixed_n', ...
                'resHBICG_n', 'resHBICG_fixed_n', ...
                'resHBIC_c1_n', 'resHBIC_c2_n', ...
                'min_lamb2s'}];
        end

        if run_gflsl && save_candidate_thetas
            vars_to_save = [vars_to_save, {'Theta1s', 'Theta2s', 'Thetans'}];
        end

        if post_refits_enabled
            vars_to_save = [vars_to_save, { ...
                'post_detectors', 'post_methods', ...
                'post_HD', 'post_F1', 'post_acc', 'post_err', 'post_nb'}];

            if save_post_refit_paths
                vars_to_save{end + 1} = 'post_paths';
            end

            if include_tbfl_native_cov
                vars_to_save = [vars_to_save, { ...
                    'tbfl_native_HD', 'tbfl_native_F1', 'tbfl_native_acc', ...
                    'tbfl_native_err', 'tbfl_native_nbr'}];
                if save_post_refit_paths
                    vars_to_save{end + 1} = 'tbfl_native_paths';
                end
            end
        end

        save(filename, vars_to_save{:}, '-v7.3');
    end

    if showbar && run_gflsl
        fprintf(' ✓\n');
    end
end

function row = build_metric_row(name, vals)
    % Expand a vector of metrics into a fixed-width cell row.
    if numel(vals) < 5
        vals = [vals(:).' nan(1, 5 - numel(vals))];
    else
        vals = vals(1:5);
    end
    row = {name, vals(1), vals(2), vals(3), vals(4), vals(5)};
end

function rows = fixed_metric_table_rows(names, values)
    rows = cell(numel(names), 6);
    for ii = 1:numel(names)
        rows(ii, :) = build_metric_row(names(ii), ...
            mean(values(:, :, ii), 1, 'omitnan'));
    end
end

function row = build_section_row(label)
    % Row used as an inline section header within the metric table.
    row = {sprintf('%s', label), nan, nan, nan, nan, nan};
end

function path = candidate_cache_file(cache_dir, tt, version, varargin)
    key = string(version) + "_exp" + sprintf('%03d', tt);
    for ii = 1:numel(varargin)
        key = key + "_" + lambda_key(varargin{ii});
    end
    subdir = fullfile(char(cache_dir), char(version), sprintf('exp_%03d', tt));
    path = fullfile(subdir, char(key + ".mat"));
end

function payload = load_candidate_cache(path)
    payload = [];
    if exist(path, 'file') ~= 2
        return
    end
    try
        loaded = load(path, 'payload');
        if isfield(loaded, 'payload') && isstruct(loaded.payload)
            payload = loaded.payload;
        end
    catch
        payload = [];
    end
end

function save_candidate_cache(path, payload)
    parent = fileparts(path);
    if exist(parent, 'dir') ~= 7
        mkdir(parent);
    end
    save(path, 'payload', '-v7.3');
end

function [payloads, dq_count] = run_warm_chain(chain, X_tt, lamb1s, ...
    lamb2grid, idxcomb, beta_, tol, gflsl_maxiter, merge_breaks, ...
    termination_check_freq, make_est, X_test_tt, trueTheta_tt, trueBreaks_tt, ...
    has_test, include_theta, xi1_current, xi2_current, seed_state)
% Fit one warm-start chain sequentially: candidate ee is initialized from
% the converged solver state of candidate ee-1. make_est is the estimator
% factory carrying the penalty_scale/relaxation metric. xi1_current/xi2_current
% are the adaptive weights (omit or pass [] for the non-adaptive grid).
% seed_state, when provided, initializes the FIRST candidate of the chain
% from the stage-1 solver state (cross-stage warm start) instead of a
% cold start. Progress bars are not updated inside chains (DataQueue is
% not reliably transmitted into nested functions on parfor workers).
    if nargin < 17
        xi1_current = [];
        xi2_current = [];
        seed_state = [];
    elseif nargin < 18
        seed_state = [];
    end
    payloads = cell(numel(chain), 1);
    prev_state = [];
    for ee = 1:numel(chain)
        ii = chain(ee);
        if isempty(xi1_current)
            lamb1_ii = lamb1s(idxcomb(ii, 1));
            lamb2_ii = lamb2grid(idxcomb(ii, 2));
        else
            lamb1_ii = lamb1s(idxcomb(ii, 1)) .* xi1_current;
            lamb2_ii = lamb2grid(idxcomb(ii, 2)) .* reshape(xi2_current, 1, []);
        end
        if isempty(prev_state)
            Est = make_est(X=X_tt, lamb1=lamb1_ii, lamb2=lamb2_ii, ...
                beta_=beta_, epsilon=0.01, tol=tol, maxiter=gflsl_maxiter, ...
                disp_freq=inf, merge_breaks=merge_breaks, ...
                termination_check_freq=termination_check_freq, ...
                init_state=seed_state);
        else
            Est = make_est(X=X_tt, lamb1=lamb1_ii, lamb2=lamb2_ii, ...
                beta_=beta_, epsilon=0.01, tol=tol, maxiter=gflsl_maxiter, ...
                disp_freq=inf, merge_breaks=merge_breaks, ...
                termination_check_freq=termination_check_freq, ...
                init_state=prev_state);
        end
        Est.run
        prev_state = Est.solver_state();
        payloads{ee} = estimator_cache_payload(Est, X_test_tt, ...
            trueTheta_tt, trueBreaks_tt, has_test, include_theta);
    end
end

function payload = estimator_cache_payload(Est, X_test_tt, trueTheta_tt, ...
    trueBreaks_tt, has_test, include_theta)
    % X_test_tt / trueTheta_tt / trueBreaks_tt are the per-experiment slices
    % prepared by the caller (already restricted to experiment tt).
    payload = struct();
    payload.EstBreaks = Est.EstBreaks;
    payload.AIC = Est.AIC;
    payload.BIC = Est.BIC;
    payload.HBIC = Est.HBIC;
    payload.HBICG = Est.HBICG;
    payload.lossval = Est.lossval;
    payload.loss = inf;
    if has_test
        payload.loss = Est.lossfunc(X_test_tt);
    end
    payload.HD = inf;
    payload.F1 = inf;
    payload.acc = inf;
    payload.error = inf;
    % An empty trueBreaks_tt is the valid no-change case (m=0).
    if ~isempty(trueTheta_tt)
        [payload.HD, payload.F1, payload.acc, payload.error] = ...
            compare_true(Est, trueTheta_tt, trueBreaks_tt);
    end
    if include_theta
        payload.Theta_k = single(Est.Theta_k);
    end
end

function [HD, F1, acc, err, AIC, BIC, HBIC, HBICG, lossval, loss, breaks] = ...
    unpack_candidate_cache(payload)
    HD = payload.HD;
    F1 = payload.F1;
    acc = payload.acc;
    err = payload.error;
    AIC = payload.AIC;
    BIC = payload.BIC;
    HBIC = payload.HBIC;
    HBICG = payload.HBICG;
    lossval = payload.lossval;
    loss = payload.loss;
    breaks = payload.EstBreaks;
end

function dt = ensure_datetime_array(val)
    if isdatetime(val)
        dt = val;
    else
        dt = datetime(val, 'ConvertFrom', 'datenum');
    end
end

function validate_lambda_post(lambda_post, post_methods)
    % Accept numeric lambda overrides. Missing entries are handled later by
    % lambda_for_method(), which falls back to method defaults.
    if isempty(post_methods)
        return
    end

    if isempty(lambda_post)
        return
    end

    if isnumeric(lambda_post)
        return
    end

    error('get_two_stage_est:badLambdaPost', ...
        'lambda_post must be empty or numeric.');
end

function lambda_val = lambda_for_method(lambda_post, method_idx)
    % Resolve one lambda value for the current post-refit method.
    if isempty(lambda_post)
        lambda_val = NaN;
        return
    end

    if ~isnumeric(lambda_post)
        error('get_two_stage_est:badLambdaPost', ...
            'lambda_post must be numeric.');
    end

    if isscalar(lambda_post)
        lambda_val = lambda_post;
    else
        lambda_vec = lambda_post(:);
        if method_idx > numel(lambda_vec)
            lambda_val = NaN;
            return
        end
        lambda_val = lambda_vec(method_idx);
    end

    if ~isfinite(lambda_val) || lambda_val < 0
        lambda_val = NaN;
        return
    end
end

function opts = apply_fixed_banded_convex_banding_settings(opts, method, DGP, p)
    % Keep baseline post-refits on their defaults, except for the
    % banded-DGP convex-banding benchmark where one-time simulation tuning
    % fixed the theory-scale multiplier by dimension.
    if string(method) ~= "convex_banding" || string(DGP) ~= "banded"
        return
    end

    opts.lambda = NaN;
    if p == 10
        opts.lambda_scale = 6.0;
    elseif p == 20
        opts.lambda_scale = 5.0;
    elseif p == 50
        opts.lambda_scale = 5.0;
    end
end

function x = local_mask_unevaluated(x, evaluated)
    % Force criterion values at unevaluated cells to Inf without touching
    % evaluated cells (Inf + Inf*0 would produce NaN, so assignment is used
    % instead of arithmetic masking).
    x(~evaluated) = Inf;
end
