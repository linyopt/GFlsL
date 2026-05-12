function get_two_stage_est(X, X_test, trueTheta, trueBreaks, lamb1s, lamb2s, lamb2s2, varargin)
%% Obtain two-stage estimators.
%
% - Usage:
%   get_two_stage_est(X, X_test, trueTheta, trueBreaks, lamb1s, lamb2s, lamb2s2, ...
%       'mu2', 0.5, 'mu1', 0.8, 'kappa_', 0.5, 'tol', 1e-3, 'beta_', 0.1, ...
%       'tau_BSOP', 10, 'delta_WBSIP', 5, 'tau_WBSIP', sqrt(p*log(T))*1.5, ...
%       'gamma_DCDP', 10, 'zeta_DCDP', 1, ...
%       'filepre', "res", 'txt_filepre', "res", 'mat_filepre', "res", ...
%       'savetxt', 0, 'savemat', 0, 'showbar', 0, 'showfig', 0,
%       'dates', 1:T)
%   OR
%   get_two_stage_est(X, X_test, trueTheta, trueBreaks, lamb1s, lamb2s, lamb2s2, ...
%       mu2=0.5, mu1=0.8, kappa_=0.5, tol=1e-3, beta_=0.1, tau_BSOP=10, ...
%       delta_WBSIP=5, tau_WBSIP=sqrt(p*log(T))*1.5, gamma_DCDP=10, zeta_DCDP=1, ...
%       filepre="res", txt_filepre="res", mat_filepre="res", ...
%       savetxt=0, savemat=0, showbar=0, showfig=0, dates=1:T)
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
%   @showbar: Flag to show the progress bar. Default is 0.
%   @showfig: Flag to show figures. Default is 0.
%   @dates:   The dates. Default is 1:T.
%   @merge_breaks: Apply mergeBreaks post-processing on GFlsL outputs. Default is 0.
%   @save_tbfl_omega: If true, compute and save TBFL precision matrices (Omega)
%                     using EstimateOmega method. Default is false.
%   @run_gflsl: If false, skip the GFlsL adaptive/non-adaptive fits and
%               only run competing baselines plus optional post-refits.
%   @post_refit_methods: String array of post-refit covariance estimators
%                        applied to competing baselines.
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

    % Derive dimensions from input data
    [T, p, num_exps] = size(X);

    p_ = inputParser;
    addParameter(p_, 'mu2', 0.5);
    addParameter(p_, 'mu1', 0.8);
    addParameter(p_, 'kappa_', 0.5);
    addParameter(p_, 'tol', 1e-3);
    addParameter(p_, 'beta_', 0.1);
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
    addParameter(p_, 'showbar', 0);
    addParameter(p_, 'showfig', 0);
    addParameter(p_, 'dates', 1:T);
    addParameter(p_, 'merge_breaks', false);
    addParameter(p_, 'DGP', '');
    addParameter(p_, 'prob', NaN);
    addParameter(p_, 'save_tbfl_omega', false);
    addParameter(p_, 'run_gflsl', true);
    addParameter(p_, 'post_refit_methods', string.empty(1, 0));
    addParameter(p_, 'include_tbfl_native_cov', true);
    addParameter(p_, 'save_post_refit_paths', false);
    addParameter(p_, 'min_eig', 0.01);
    addParameter(p_, 'delta_adaptive', 2);
    addParameter(p_, 'rho_post', 2.5);
    addParameter(p_, 'maxiter_post', 500);
    addParameter(p_, 'tol_post', 1e-5);
    addParameter(p_, 'lambda_post', NaN);
    parse(p_, varargin{:});

    mu2 = p_.Results.mu2;
    mu1 = p_.Results.mu1;
    kappa_ = p_.Results.kappa_;
    tol = p_.Results.tol;
    beta_ = p_.Results.beta_;
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
    showbar = p_.Results.showbar;
    showfig = p_.Results.showfig;
    dates = ensure_datetime_array(p_.Results.dates);
    merge_breaks = logical(p_.Results.merge_breaks);
    DGP = p_.Results.DGP;
    prob = p_.Results.prob;
    save_tbfl_omega = logical(p_.Results.save_tbfl_omega);
    run_gflsl = logical(p_.Results.run_gflsl);
    has_test = ~isempty(X_test);
    post_refit_methods = unique(string(p_.Results.post_refit_methods), 'stable');
    post_refit_methods = post_refit_methods(strlength(post_refit_methods) > 0);
    include_tbfl_native_cov = logical(p_.Results.include_tbfl_native_cov);
    save_post_refit_paths = logical(p_.Results.save_post_refit_paths);
    post_refits_enabled = ~isempty(post_refit_methods);
    lambda_post = p_.Results.lambda_post;
    validate_lambda_post(lambda_post, post_refit_methods);

    % A helper function used to find the index of the first minimum element in
    % a given tensor.
    findMinIn = @(x)ind2sub(size(x), find(x == min(x, [], 'all'), 1));

    if ~isempty(trueBreaks)
        num_breaks = length(trueBreaks{1});
    else
        num_breaks = NaN;
    end

    filename = mat_filepre + ".mat";

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
    if savemat && run_gflsl
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
    end

    % Metrics based on different criteria.
    resHFE2 = zeros(num_exps, 5);
    resloss2 = zeros(num_exps, 5);
    resAIC2 = zeros(num_exps, 5);
    resBIC2 = zeros(num_exps, 5);
    resHBIC_2 = zeros(num_exps, 5);
    resHBICG_2 = zeros(num_exps, 5);

    resHFEn = zeros(num_exps, 5);
    reslossn = zeros(num_exps, 5);
    resAICn = zeros(num_exps, 5);
    resBICn = zeros(num_exps, 5);
    resHBIC_n = zeros(num_exps, 5);
    resHBICG_n = zeros(num_exps, 5);

    % The variable saving the minimal lamb2 used in the first stage.
    min_lamb2s = lamb2s(1) * ones(num_exps, 1);

    % The loops.
    for tt = 1:num_exps
        
        if showbar && run_gflsl
            fprintf('\n  Experiment %02d/%02d ▶ ', tt, num_exps);
            updateProgress(total, true);
        end
        
        % The competing method: BSOP.
        temp_BS = BS_cov(X(:, :, tt)', 0, T, tau_BSOP);
        if isempty(temp_BS.S)
            BScps{tt} = [];
        else
            BScps{tt} = sort(temp_BS.S);
        end
        
        % The competing method: WBSIP.
        if has_test
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
        if has_test
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
        
        % Apply covariance post-refits to the competing break detectors
        % using the same break estimates already computed above.
        if post_refits_enabled
            X_tt = X(:, :, tt);

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
                            Sigma_post, breaks_tt, trueTheta(:, :, :, tt), ...
                            trueBreaks{tt});
                    end

                    if save_post_refit_paths
                        post_paths{dd, mm, tt} = Sigma_post;
                    end
                end
            end

            if include_tbfl_native_cov
                Sigma_tbfl_native = invert_precision_segments(Omega_hat, T, TBFLcps{tt}, ...
                    min_eig=p_.Results.min_eig);

                if ~isempty(trueTheta) && ~isempty(trueBreaks)
                    [tbfl_native_HD(tt), tbfl_native_F1(tt), ...
                        tbfl_native_acc(tt), tbfl_native_err(tt), ...
                        tbfl_native_nbr(tt)] = evaluate_covariance_refit( ...
                        Sigma_tbfl_native, TBFLcps{tt}, ...
                        trueTheta(:, :, :, tt), trueBreaks{tt});
                end

                if save_post_refit_paths
                    tbfl_native_paths{tt} = Sigma_tbfl_native;
                end
            end
        end

        if run_gflsl
            % For the nonadaptive version.
            parfor ii = 1:num12
                Estn = GFlsL(X=X(:, :, tt), lamb1=lamb1s(idxcomb(ii, 1)), ...
                    lamb2=lamb2s(idxcomb(ii, 2)), beta_=beta_, epsilon=0.01, ...
                    tol=tol, maxiter=inf, disp_freq=inf, showfig=0, ...
                    merge_breaks=merge_breaks);
                Estn.run

                if savemat
                    Thetans(:, :, :, ii, tt) = single(Estn.Theta_k);
                end
                EstBreaksn{ii, tt} = Estn.EstBreaks;
                AICns(ii, tt) = Estn.AIC;
                BICns(ii, tt) = Estn.BIC;
                HBIC_ns(ii, tt) = Estn.HBIC;
                HBICG_ns(ii, tt) = Estn.HBICG;
                lossvalns(ii, tt) = Estn.lossval;
                if has_test
                    lossns(ii, tt) = Estn.lossfunc(X_test(:, :, :, tt));
                end

                if ~isempty(trueTheta) && ~isempty(trueBreaks)
                    [HDsn(ii, tt), F1sn(ii, tt), accsn(ii, tt), errorsn(ii, tt)] = ...
                        compare_true(Estn, trueTheta(:, :, :, tt), trueBreaks{tt});
                end

                if showbar && ~isempty(dq)
                    send(dq, 1);
                end
            end

            % For the two-stage experiments.
            for k = 1:len2
                Est1 = GFlsL(X=X(:, :, tt), lamb1=0, lamb2=lamb2s(k), beta_=beta_, ...
                    epsilon=0.01, tol=tol, maxiter=inf, disp_freq=inf, ...
                    showfig=0, merge_breaks=merge_breaks);
                Est1.run

                if isempty(Est1.EstBreaks)
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
                                Est1 = GFlsL(X=X(:, :, tt), lamb1=0, lamb2=lamb2_temp, beta_=beta_, ...
                                    epsilon=0.01, tol=tol, maxiter=inf, disp_freq=inf, ...
                                    showfig=0, merge_breaks=merge_breaks);
                                Est1.run;
                                if ~isempty(Est1.EstBreaks)
                                    min_lamb2s(tt) = lamb2_temp;
                                    break
                                end
                                lamb2_temp = lamb2_temp - step_size;
                                iter = iter + 1;
                            end
                        end
                    else
                        break
                    end
                end

                if savemat
                    Theta1s(:, :, :, k, tt) = single(Est1.Theta_k);
                end
                EstBreaks1{k, tt} = Est1.EstBreaks;
                AIC1s(k, tt) = Est1.AIC;
                BIC1s(k, tt) = Est1.BIC;
                HBIC_1s(k, tt) = Est1.HBIC;
                HBICG_1s(k, tt) = Est1.HBICG;
                lossval1s(k, tt) = Est1.lossval;
                if has_test
                    loss1s(k, tt) = Est1.lossfunc(X_test(:, :, :, tt));
                end

                if ~isempty(trueTheta) && ~isempty(trueBreaks)
                    [HDs1(k, tt), F1s1(k, tt), accs1(k, tt), errors1(k, tt)] = ...
                        compare_true(Est1, trueTheta(:, :, :, tt), trueBreaks{tt});
                end

                for idx_mu2 = 1:lenmu2
                    for idx_mu1 = 1:lenmu1
                        % Build adaptive weights for the current experiment only.
                        % This avoids storing large xi1/xi2 tensors across all
                        % experiments, which are not needed after this loop body.
                        xi1_current = max(abs(Est1.Theta_k), bb_) .^ -mu1(idx_mu1);
                        xi2_current = max(Est1.norm_diff, bb_) .^ -mu2(idx_mu2);

                        parfor ii = 1:num122
                            Est2 = GFlsL(X=X(:, :, tt), ...
                                lamb1=lamb1s(idxcomb2(ii, 1)) .* xi1_current, ...
                                lamb2=lamb2s2(idxcomb2(ii, 2)) .* reshape(xi2_current, 1, []), ...
                                beta_=beta_, epsilon=0.01, tol=tol, maxiter=inf, ...
                                disp_freq=inf, showfig=0, merge_breaks=merge_breaks);
                            Est2.run

                            if savemat
                                Theta2s(:, :, :, ii, k, idx_mu2, idx_mu1, tt) = single(Est2.Theta_k);
                            end
                            EstBreaks2{ii, k, idx_mu2, idx_mu1, tt} = Est2.EstBreaks;
                            AIC2s(ii, k, idx_mu2, idx_mu1, tt) = Est2.AIC;
                            BIC2s(ii, k, idx_mu2, idx_mu1, tt) = Est2.BIC;
                            HBIC_2s(ii, k, idx_mu2, idx_mu1, tt) = Est2.HBIC;
                            HBICG_2s(ii, k, idx_mu2, idx_mu1, tt) = Est2.HBICG;
                            lossval2s(ii, k, idx_mu2, idx_mu1, tt) = Est2.lossval;
                            if has_test
                                loss2s(ii, k, idx_mu2, idx_mu1, tt) = Est2.lossfunc(X_test(:, :, :, tt));
                            end

                            if ~isempty(trueTheta) && ~isempty(trueBreaks)
                                [HDs2(ii, k, idx_mu2, idx_mu1, tt), F1s2(ii, k, idx_mu2, idx_mu1, tt), ...
                                    accs2(ii, k, idx_mu2, idx_mu1, tt), errors2(ii, k, idx_mu2, idx_mu1, tt)] = ...
                                    compare_true(Est2, trueTheta(:, :, :, tt), trueBreaks{tt});
                            end

                            if showbar && ~isempty(dq)
                                send(dq, 1);
                            end
                        end
                    end
                end
            end

            if ~isempty(trueTheta) && ~isempty(trueBreaks)
                resHFE2(tt, :) = two_stage_HFE(HDs2(:, :, :, :, tt), ...
                    F1s2(:, :, :, :, tt), errors2(:, :, :, :, tt), ...
                    accs2(:, :, :, :, tt), EstBreaks2(:, :, :, :, tt));
                resHFEn(tt, :) = two_stage_HFE(HDsn(:, tt), F1sn(:, tt), errorsn(:, tt), ...
                    accsn(:, tt), EstBreaksn(:, tt));

                if has_test
                    [i1, i2, i3, i4] = findMinIn(loss2s(:, :, :, :, tt));
                    resloss2(tt, :) = [HDs2(i1, i2, i3, i4, tt), F1s2(i1, i2, i3, i4, tt), ...
                        errors2(i1, i2, i3, i4, tt), accs2(i1, i2, i3, i4, tt), ...
                        length(EstBreaks2{i1, i2, i3, i4, tt})];

                    jj = findMinIn(lossns(:, tt));
                    reslossn(tt, :) = [HDsn(jj, tt), F1sn(jj, tt), errorsn(jj, tt), ...
                        accsn(jj, tt), length(EstBreaksn{jj, tt})];
                end

                [i1, i2, i3, i4] = findMinIn(AIC2s(:, :, :, :, tt));
                resAIC2(tt, :) = [HDs2(i1, i2, i3, i4, tt), F1s2(i1, i2, i3, i4, tt), ...
                    errors2(i1, i2, i3, i4, tt), accs2(i1, i2, i3, i4, tt), ...
                    length(EstBreaks2{i1, i2, i3, i4, tt})];

                jj = findMinIn(AICns(:, tt));
                resAICn(tt, :) = [HDsn(jj, tt), F1sn(jj, tt), errorsn(jj, tt), ...
                    accsn(jj, tt), length(EstBreaksn{jj, tt})];

                [i1, i2, i3, i4] = findMinIn(BIC2s(:, :, :, :, tt));
                resBIC2(tt, :) = [HDs2(i1, i2, i3, i4, tt), F1s2(i1, i2, i3, i4, tt), ...
                    errors2(i1, i2, i3, i4, tt), accs2(i1, i2, i3, i4, tt), ...
                    length(EstBreaks2{i1, i2, i3, i4, tt})];

                jj = findMinIn(BICns(:, tt));
                resBICn(tt, :) = [HDsn(jj, tt), F1sn(jj, tt), errorsn(jj, tt), ...
                    accsn(jj, tt), length(EstBreaksn{jj, tt})];

                [i1, i2, i3, i4] = findMinIn(HBIC_2s(:, :, :, :, tt));
                resHBIC_2(tt, :) = [HDs2(i1, i2, i3, i4, tt), F1s2(i1, i2, i3, i4, tt), ...
                    errors2(i1, i2, i3, i4, tt), accs2(i1, i2, i3, i4, tt), ...
                    length(EstBreaks2{i1, i2, i3, i4, tt})];

                jj = findMinIn(HBIC_ns(:, tt));
                resHBIC_n(tt, :) = [HDsn(jj, tt), F1sn(jj, tt), errorsn(jj, tt), ...
                    accsn(jj, tt), length(EstBreaksn{jj, tt})];

                [i1, i2, i3, i4] = findMinIn(HBICG_2s(:, :, :, :, tt));
                resHBICG_2(tt, :) = [HDs2(i1, i2, i3, i4, tt), F1s2(i1, i2, i3, i4, tt), ...
                    errors2(i1, i2, i3, i4, tt), accs2(i1, i2, i3, i4, tt), ...
                    length(EstBreaks2{i1, i2, i3, i4, tt})];

                jj = findMinIn(HBICG_ns(:, tt));
                resHBICG_n(tt, :) = [HDsn(jj, tt), F1sn(jj, tt), errorsn(jj, tt), ...
                    accsn(jj, tt), length(EstBreaksn{jj, tt})];
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
                build_metric_row('HBICG', mean(resHBICG_2, 1));
                ];
            if has_test
                adaptive_rows = [build_metric_row('loss', mean(resloss2, 1)); adaptive_rows];
            end

            nonadaptive_rows = [
                build_metric_row('HFE',  mean(resHFEn, 1));
                build_metric_row('AIC',  mean(resAICn, 1));
                build_metric_row('BIC',  mean(resBICn, 1));
                build_metric_row('HBIC', mean(resHBIC_n, 1));
                build_metric_row('HBICG', mean(resHBICG_n, 1));
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
            'mu1', 'mu2', 'kappa_', 'tol', 'beta_', ...
            'tau_BSOP', 'delta_WBSIP', 'tau_WBSIP', 'M_WBSIP', ...
            'gamma_DCDP', 'zeta_DCDP', 'Q_DCDP', ...
            'buffer_DCDP', 'buffer_refine_DCDP', ...
            'merge_breaks', 'run_gflsl', 'lambda_post', ...
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
                'Theta1s', 'Theta2s', 'Thetans', ...
                'resHFE2', 'resloss2', 'resAIC2', 'resBIC2', ...
                'resHBIC_2', 'resHBICG_2', ...
                'resHFEn', 'reslossn', 'resAICn', 'resBICn', ...
                'resHBIC_n', 'resHBICG_n', ...
                'min_lamb2s'}];
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

    if showfig && run_gflsl
        % Plot figures with datetime-based labels (datestr is deprecated).
        format_date = @(d) string(d, 'yyyy-MM-dd');

        for tt = 1:num_exps
            figure('Name', "BSOP/WBSIP for Experiment " + tt);
            legend;
            
            BSdates = dates(BScps{tt});
            if ~isempty(BSdates)
                labels_BS = arrayfun(@(d) sprintf('BSOP break time %s', format_date(d)), BSdates, 'UniformOutput', false);
                xline(BSdates, 'r--', labels_BS);
            end
            hold on;
            
            if has_test
                WBSIPdates = dates(WBSIPcps{tt});
                labels_WBSIP = arrayfun(@(d) sprintf('WBSIP break time %s', format_date(d)), WBSIPdates, 'UniformOutput', false);
                xline(WBSIPdates, 'b.-', labels_WBSIP);
                xlim(gca, [dates(1), dates(end)])
                title('Breaks detected by BSOP/WBSIP');
            else
                title('Breaks detected by BSOP');
            end
            
            xlabel('Date');
            legend;
            hold off;

            BICnj = findMinIn(BICns(:, tt));
            HBICnj = findMinIn(HBIC_ns(:, tt));
            HBICGnj = findMinIn(HBICG_ns(:, tt));

            [BIC2i1, BIC2i2, BIC2i3, BIC2i4] = findMinIn(BIC2s(:, :, :, :, tt));
            [HBIC2i1, HBIC2i2, HBIC2i3, HBIC2i4] = findMinIn(HBIC_2s(:, :, :, :, tt));
            [HBICG2i1, HBICG2i2, HBICG2i3, HBICG2i4] = findMinIn(HBICG_2s(:, :, :, :, tt));

            breaks_BICn = EstBreaksn{BICnj, tt};
            breaks_HBICn = EstBreaksn{HBICnj, tt};
            breaks_HBICGn = EstBreaksn{HBICGnj, tt};

            breaks_BIC2 = EstBreaks2{BIC2i1, BIC2i2, BIC2i3, BIC2i4, tt};
            breaks_HBIC2 = EstBreaks2{HBIC2i1, HBIC2i2, HBIC2i3, HBIC2i4, tt};
            breaks_HBICG2 = EstBreaks2{HBICG2i1, HBICG2i2, HBICG2i3, HBICG2i4, tt};

            figure('Name', "Experiment " + tt);
            subplot(2, 3, 1);
            
            % Non-adaptive plots
            subplot(2, 3, 1);
            dates_BICn = dates(breaks_BICn);
            labels_BICn = arrayfun(@(d) sprintf('Non-adaptive BIC break %s', format_date(d)), dates_BICn, 'UniformOutput', false);
            xline(dates_BICn, 'k--', labels_BICn)
            ax = gca;
            ax.XAxis.TickLabelFormat = 'yyyy-MM-dd';
            xlim(ax, [dates(1), dates(end)])
            title('Non-adaptive BIC');

            subplot(2, 3, 2);
            dates_HBICn = dates(breaks_HBICn);
            labels_HBICn = arrayfun(@(d) sprintf('Non-adaptive HBIC break %s', format_date(d)), dates_HBICn, 'UniformOutput', false);
            xline(dates_HBICn, 'k--', labels_HBICn)
            ax = gca;
            ax.XAxis.TickLabelFormat = 'yyyy-MM-dd';
            xlim(ax, [dates(1), dates(end)])
            title('Non-adaptive HBIC');

            subplot(2, 3, 3);
            dates_HBICGn = dates(breaks_HBICGn);
            labels_HBICGn = arrayfun(@(d) sprintf('Non-adaptive HBICG break %s', format_date(d)), dates_HBICGn, 'UniformOutput', false);
            xline(dates_HBICGn, 'k--', labels_HBICGn)
            ax = gca;
            ax.XAxis.TickLabelFormat = 'yyyy-MM-dd';
            xlim(ax, [dates(1), dates(end)])
            title('Non-adaptive HBICG');

            % Adaptive plots
            subplot(2, 3, 4);
            dates_BIC2 = dates(breaks_BIC2);
            labels_BIC2 = arrayfun(@(d) sprintf('Non-adaptive BIC break %s', format_date(d)), dates_BIC2, 'UniformOutput', false);
            xline(dates_BIC2, 'k--', labels_BIC2)
            ax = gca;
            ax.XAxis.TickLabelFormat = 'yyyy-MM-dd';
            xlim(ax, [dates(1), dates(end)])
            title('Adaptive BIC');

            subplot(2, 3, 5);
            dates_HBIC2 = dates(breaks_HBIC2);
            labels_HBIC2 = arrayfun(@(d) sprintf('Non-adaptive HBIC break %s', format_date(d)), dates_HBIC2, 'UniformOutput', false);
            xline(dates_HBIC2, 'k--', labels_HBIC2)
            ax = gca;
            ax.XAxis.TickLabelFormat = 'yyyy-MM-dd';
            xlim(ax, [dates(1), dates(end)])
            title('Adaptive HBIC');

            subplot(2, 3, 6);
            dates_HBICG2 = dates(breaks_HBICG2);
            labels_HBICG2 = arrayfun(@(d) sprintf('Non-adaptive HBICG break %s', format_date(d)), dates_HBICG2, 'UniformOutput', false);
            xline(dates_HBICG2, 'k--', labels_HBICG2)
            ax = gca;
            ax.XAxis.TickLabelFormat = 'yyyy-MM-dd';
            xlim(ax, [dates(1), dates(end)])
            title('Adaptive HBICG');
        end
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

function row = build_section_row(label)
    % Row used as an inline section header within the metric table.
    row = {sprintf('%s', label), nan, nan, nan, nan, nan};
end

% Convert numeric date inputs (e.g., MATLAB serial date numbers) to datetime so
% plotting and string formatting consistently rely on datetime operations.
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
    end
end
