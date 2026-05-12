function [gamma_cv, cv_scores] = dcdp_cv_gamma(X, gamma_grid, opts)
%% Cross-validation for DCDP gamma parameter selection.
% Supports two modes:
% 1. Odd/even split CV (default): Split X into odd (train) and even (test) indices.
% 2. Independent samples CV: Use separate X_test if provided.
%
% - Usage:
%   [gamma_cv, cv_scores] = dcdp_cv_gamma(X, gamma_grid);
%   [gamma_cv, cv_scores] = dcdp_cv_gamma(X, gamma_grid, X_test=X_test);
%   [gamma_cv, cv_scores] = dcdp_cv_gamma(X, gamma_grid, model='ggm', zeta=1);
%
% - Input:
%   @X:           Training data matrix, size (n, p).
%   @gamma_grid:  Vector of candidate gamma values to evaluate. If empty,
%                 use a default grid tuned for the simulation settings.
%   @X_test:      Optional test data, size (n, p) or (n, p, Nsim). If
%                 provided, use independent-sample CV. Otherwise, use
%                 odd/even split.
%   @model:       Model type: 'mean', 'linear', or 'ggm'. Default 'ggm'.
%   @zeta:        Penalty for conquer step. Default 0.
%   @Q:           Grid size for DCDP. Default: 10 for odd/even, 50 for independent.
%   @lambda:      Lasso penalty. Default: auto-computed.
%   @s:           Sparsity level. Default 5.
%   @C_lambda:    Constant for lambda. Default 1.
%   @C_F:         Constant for min interval length. Default 1.
%   @buffer_refine: Buffer for local refinement. Default: auto.
%   @step_refine: Step size for refinement. Default 1.
%
% - Output:
%   @gamma_cv:    Selected gamma value (minimizes CV error).
%   @cv_scores:   Vector of CV scores for each gamma in gamma_grid.

arguments (Input)
    X double
    gamma_grid double = []
    opts.X_test double = []
    opts.model char {mustBeMember(opts.model, {'mean', 'linear', 'ggm'})} = 'ggm'
    opts.zeta (1, 1) double {mustBeNonnegative} = 0
    opts.Q (1, 1) double {mustBeNonnegative, mustBeInteger} = 0
    opts.lambda (1, 1) double {mustBeNonnegative} = 0
    opts.s (1, 1) double {mustBePositive, mustBeInteger} = 5
    opts.C_lambda (1, 1) double {mustBePositive} = 1
    opts.C_F (1, 1) double {mustBePositive} = 1
    opts.buffer_refine (1, 1) double {mustBeNonnegative, mustBeInteger} = 0
    opts.step_refine (1, 1) double {mustBePositive, mustBeInteger} = 1
end

[n, ~] = size(X);
has_test = ~isempty(opts.X_test);

if isempty(gamma_grid)
    gamma_grid = [100, 500, 700, 800, 900, 1000];
end

% Build the training sample and the validation sample stack.
if has_test
    X_train = X;
    X_test_stack = opts.X_test;

    if ismatrix(X_test_stack)
        X_test_stack = reshape(X_test_stack, size(X_test_stack, 1), size(X_test_stack, 2), 1);
    end

    if size(X_test_stack, 1) ~= n
        error('dcdp_cv_gamma:InvalidXTestRows', ...
            'X_test must have the same number of rows as X.');
    end
    if size(X_test_stack, 2) ~= size(X, 2)
        error('dcdp_cv_gamma:InvalidXTestCols', ...
            'X_test must have the same number of columns as X.');
    end

    m = n;

    if opts.Q == 0
        Q_cv = 50;
    else
        Q_cv = opts.Q;
    end
else
    % Mode 1: Odd/even split CV.
    % Split data into odd (train) and even (test) indices.
    if mod(n, 2) == 1
        n_use = n - 1;
    else
        n_use = n;
    end
    train_idx = 1:2:n_use;
    test_idx = 2:2:n_use;
    m = length(train_idx);

    X_train = X(train_idx, :);
    X_test_stack = reshape(X(test_idx, :), m, size(X, 2), 1);

    % Cap Q for CV to avoid overly dense grids on short series.
    if opts.Q == 0
        Q_cv = min(10, max(2, m - 1));
    else
        Q_cv = opts.Q;
    end
end

% Initialize CV scores.
num_gamma = length(gamma_grid);
cv_scores = zeros(num_gamma, 1);
num_test_samples = size(X_test_stack, 3);

% Evaluate each gamma.
for g = 1:num_gamma
    gamma = gamma_grid(g);

    % Build training DCDP object.
    train_dcdp = DCDP(X=X_train, model=opts.model, gamma=gamma, zeta=opts.zeta, ...
                      Q=Q_cv, lambda=opts.lambda, s=opts.s, ...
                      C_lambda=opts.C_lambda, C_F=opts.C_F, ...
                      buffer_refine=opts.buffer_refine, step_refine=opts.step_refine);

    % Run DCDP on training data.
    train_dcdp.run();

    % Get change points from training.
    cps = train_dcdp.final_cps;

    % Build segments on [1, m].
    if isempty(cps)
        % No change points - entire data is one segment.
        segment_starts = 1;
        segment_ends = m;
    else
        segment_starts = [1; cps + 1];
        segment_ends = [cps; m];
    end

    % Evaluate on the validation sample(s). If multiple independent
    % validation samples are provided, average their segment scores.
    cv_score = 0;

    for ss = 1:num_test_samples
        X_test_single = X_test_stack(:, :, ss);

        test_dcdp = DCDP(X=X_test_single, model=opts.model, gamma=gamma, zeta=opts.zeta, ...
                         Q=Q_cv, lambda=opts.lambda, s=opts.s, ...
                         C_lambda=opts.C_lambda, C_F=opts.C_F, ...
                         buffer_refine=opts.buffer_refine, step_refine=opts.step_refine);

        for k = 1:length(segment_starts)
            I_k = segment_starts(k):segment_ends(k);

            % Estimate parameter on training segment.
            theta_hat_k = train_dcdp.EstimateTheta(I_k);

            % Evaluate F on test segment with training parameter.
            F_k = test_dcdp.ComputeFWithTheta(I_k, theta_hat_k);

            cv_score = cv_score + F_k;
        end
    end

    cv_scores(g) = cv_score / num_test_samples;
end

% Select gamma with minimum CV score.
[~, min_idx] = min(cv_scores);
gamma_cv = gamma_grid(min_idx);

end
