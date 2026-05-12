classdef TBFL < handle
    properties
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% The data-related properties.                                %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The sample size.
        T double {mustBePositive, mustBeInteger}

        % The dimension of the data.
        p double {mustBePositive, mustBeInteger}

        % The data matrix of size (T, p).
        % X_t ~ N(0, Sigma_t*), where Sigma_t* = Omega_t*^{-1}.
        X double {mustBeNumeric}

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% The hyperparameters and settings.                           %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Grid of lambda1 values for cross-validation.
        % Default: auto-computed based on data (10 values from warm start).
        lambda1_cv double = []

        % Grid of lambda2 values for cross-validation.
        % Default: [10, 1, 0.1] * sqrt(2*log(p)/T) (matching R).
        lambda2_cv double = []

        % Block size for block fused lasso.
        % If empty, selected via HBIC over a grid.
        % Default: empty (HBIC selection).
        block_size double = []

        % Grid of block sizes to search over when using HBIC selection.
        % Default: computed as in R LinearDetect.
        block_size_grid double = []

        % Gamma value for HBIC penalty.
        % Default 1.5 (matching R optimal.gamma.val).
        gamma_val (1, 1) double {mustBePositive} = 1.5

        % Use HBIC (vs BIC) inside Step II selection loop.
        % Default true (matching the MATLAB->R wrapper usage).
        HBIC_step2 (1, 1) logical = true

        % Gamma value used by LinearDetect::BIC() when HBIC_step2 is true.
        % Note: LinearDetect defaults gamma.val = 1 inside BIC().
        gamma_step2 (1, 1) double {mustBePositive} = 1

        % Minimum distance between change points for clustering.
        % Default: block_size.
        min_dist double = []

        % Tolerance for convergence in coordinate descent.
        % Default 1e-2 (matching R).
        tol (1, 1) double {mustBePositive} = 1e-2

        % Maximum iterations for coordinate descent.
        % Default 100 (matching R).
        maxiter (1, 1) double {mustBePositive, mustBeInteger} = 100

        % Linear solver used in Step I (per-block regression updates).
        % - "inv": explicit inverse (closest to LinearDetect/Armadillo path)
        % - "solve": linear solve (typically more stable and faster)
        linear_solver (1, 1) string {mustBeMember(linear_solver, ["inv", "solve"])} = "inv"

        % Display frequency.
        % Default inf (no display).
        disp_freq (1, 1) double = inf

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% Results and intermediate variables.                         %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Estimated regression coefficients for each node.
        % Beta{j} is a (p-1, num_blocks) matrix of coefficients for node j.
        Beta cell

        % Differences between consecutive coefficient matrices.
        % Delta{j} is a (p-1, num_blocks-1) matrix.
        Delta cell

        % L2 norms of jumps at each block boundary.
        % In LinearDetect, jumps are computed from the increment parameter:
        % jumps_l2(i) = ||phi_i||_F^2 for i=2..n_blocks.
        jumps_l2 double

        % Candidate change points from Step II (before local refinement).
        candidates double

        % Candidate clusters from Step II (R: pts.list).
        cp_first cell = {}

        % Final detected change points after Step III local refinement.
        breaks double

        % Step I increment estimates (R: phi.hat.full).
        phi_hat_full double = []

        % Step I block-wise coefficients (R: beta.full).
        beta_full cell = {}

        % Step III segment-wise coefficients (R: beta.hat.list).
        beta_hat_list cell = {}

        % HBIC values for each block size in the grid.
        hbic_vals double

        % Selected block size (from HBIC or user-specified).
        selected_block_size (1, 1) double = 1

        % Selected lambda1 and lambda2.
        selected_lambda1 (1, 1) double = 0
        selected_lambda2 (1, 1) double = 0

        % Block boundaries (1-indexed, including T+1).
        blocks double

        % Convergence flag.
        converged (1, 1) logical = false
    end

    methods
        function obj = TBFL(opts)
            %% TBFL: Threshold Block Fused Lasso for GGM change-point detection.
            % Implements the algorithm from Bai & Safikhani (2022).
            % Reference: arXiv:2207.09007v1
            %
            % The algorithm detects changes in the precision matrix (inverse
            % covariance) of a Gaussian Graphical Model over time using
            % neighborhood selection regression with fused lasso penalties.
            %
            % For each node j, we solve:
            %   min sum_b ||X_j^(b) - X_{-j}^(b) beta^(j,b)||^2
            %       + lambda_1 * ||beta^(j)||_1
            %       + lambda_2 * sum_b ||beta^(j,b+1) - beta^(j,b)||_1
            %
            % The algorithm has three steps:
            %   Step I:   Block fused lasso via neighborhood selection
            %   Step II:  Hard thresholding to identify candidate change points
            %   Step III: Clustering and local exhaustive search for refinement
            %
            % Output convention:
            %   Detected change points are time indices in 1..T-1. A change
            %   point tau indicates a change between rows tau and tau+1.
            %
            % - Usage:
            %   Est = TBFL(X=X);
            %   Est = TBFL(X=X, block_size=10);
            %   Est = TBFL(X=X, gamma_val=1.5);
            %
            % - Input:
            %   @X:              Data matrix, size (T, p). Row t is x_t'.
            %   @lambda1_cv:     Grid of lambda1 values. Default: auto.
            %   @lambda2_cv:     Grid of lambda2 values. Default: auto.
            %   @block_size:     Block size. Default: HBIC selection.
            %   @block_size_grid: Grid for HBIC search. Default: auto.
            %   @gamma_val:      HBIC penalty strength. Default 1.5.
            %   @min_dist:       Min distance for clustering.
            %                    Default: block_size.
            %   @tol:            Convergence tolerance. Default 1e-2.
            %   @maxiter:        Maximum iterations. Default 100.
            %   @disp_freq:      Display frequency. Default inf.

            arguments (Input)
                opts.X double
                opts.lambda1_cv double = []
                opts.lambda2_cv double = []
                opts.block_size double = []
                opts.block_size_grid double = []
                opts.gamma_val (1, 1) double {mustBePositive} = 1.5
                opts.HBIC_step2 (1, 1) logical = true
                opts.gamma_step2 (1, 1) double {mustBePositive} = 1
                opts.min_dist double = []
                opts.tol (1, 1) double {mustBePositive} = 1e-2
                opts.maxiter (1, 1) double {mustBePositive, mustBeInteger} = 100
                opts.linear_solver (1, 1) string {mustBeMember(opts.linear_solver, ["inv", "solve"])} = "inv"
                opts.disp_freq (1, 1) double = inf
            end

            % Store data.
            obj.X = opts.X;

            % Get dimensions.
            [obj.T, obj.p] = size(obj.X);

            % Store hyperparameters.
            obj.lambda1_cv = opts.lambda1_cv;
            obj.lambda2_cv = opts.lambda2_cv;
            obj.block_size = opts.block_size;
            obj.block_size_grid = opts.block_size_grid;
            obj.gamma_val = opts.gamma_val;
            obj.HBIC_step2 = opts.HBIC_step2;
            obj.gamma_step2 = opts.gamma_step2;
            obj.min_dist = opts.min_dist;
            obj.tol = opts.tol;
            obj.maxiter = opts.maxiter;
            obj.linear_solver = opts.linear_solver;
            obj.disp_freq = opts.disp_freq;

            % Set default lambda2_cv if not provided (matching R).
            % lambda.2.cv = c(10, 1, 0.1) * sqrt((log(p) + log(p))/T)
            if isempty(obj.lambda2_cv)
                base_lambda2 = sqrt(2 * log(obj.p) / obj.T);
                obj.lambda2_cv = [10, 1, 0.1] * base_lambda2;
            end

            % Set default block size grid if not provided (matching R).
            if isempty(obj.block_size_grid)
                obj.block_size_grid = obj.ComputeBlockSizeGrid();
            end

            % Initialize results.
            obj.Beta = cell(obj.p, 1);
            obj.Delta = cell(obj.p, 1);
        end

        % Run the full TBFL algorithm.
        run(obj)

        % Step I: Solve block fused lasso via neighborhood selection.
        [phi_hat_full, beta_full, flag_full] = SolveNeighborhoodSelection( ...
            obj, blocks, lambda1, lambda2, cv_index, initial_phi_full)

        % Solve fused lasso for a single node.
        [phi_hat, beta_blocks, flag] = SolveFusedLassoNode( ...
            obj, j, blocks, lambda1, lambda2, cv_index, initial_phi)

        % Select lambda1/lambda2 via LinearDetect-style CV.
        [best_lambda1, best_lambda2, phi_hat_best, beta_full_best, cv_all] = ...
            SelectLambdasForBlocks(obj, blocks, cv_index, lambda1_grid, lambda2_grid)

        % Step II: Hard thresholding to identify candidate change points.
        HardThreshold(obj)

        % Step III: Local exhaustive search for refinement.
        LocalRefinement(obj)

        % Estimate segment-wise precision matrices Omega (GGM).
        [Omega_hat, info] = EstimateOmega(obj, opts)

        % Compute HBIC for model selection.
        hbic = ComputeHBIC(obj, block_size)

        % Select block size and lambdas via HBIC.
        SelectBlockSize(obj)

        % Compute goodness-of-fit for a segment.
        gof = ComputeGOF(obj, j, I, beta_j)

        % Compute block size grid (matching R LinearDetect).
        grid = ComputeBlockSizeGrid(obj)

        % Compute lambda1 grid via warm start (matching R).
        lambda1_grid = ComputeLambda1Grid(obj, block_size)

        % Build block structure (matching R LinearDetect).
        blocks = BuildBlocks(obj, block_size)
    end

    methods (Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% Static helper functions.                                    %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function z = SoftThreshold(x, lambda)
            %% Soft-thresholding operator.
            % S_lambda(x) = sign(x) * max(|x| - lambda, 0)
            %
            % Match LinearDetect's C++ `soft_full()` implementation exactly:
            %   if (x >  lambda) x <- x - lambda
            %   else if (x < -lambda) x <- x + lambda
            %   else x <- 0
            %
            % This is algebraically equivalent to sign(x)*max(|x|-lambda,0),
            % but the branch form improves bit-level agreement near the
            % threshold, which can affect CV selection in sensitive cases.
            z = zeros(size(x), "like", x);
            pos = x > lambda;
            neg = x < -lambda;
            z(pos) = x(pos) - lambda;
            z(neg) = x(neg) + lambda;
        end

        function Xvec = vec(X)
            %% Vectorize a matrix or tensor.
            Xvec = X(:);
        end
    end
end
