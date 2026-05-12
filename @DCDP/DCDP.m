classdef DCDP < handle
    properties
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% The data-related properties.                                %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The sample size.
        n double {mustBePositive, mustBeInteger}

        % The dimension of the data.
        p double {mustBePositive, mustBeInteger}

        % The data matrix of size (n, p).
        % For mean model: X_i = mu_i* + eps_i
        % For GGM: X_i ~ N(0, Sigma_i*)
        X double {mustBeNumeric}

        % For linear regression model: response vector of size (n, 1).
        y double {mustBeNumeric}

        % Model type: 'mean', 'linear', or 'ggm'.
        model char {mustBeMember(model, {'mean', 'linear', 'ggm'})} = 'mean'

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% The hyperparameters and settings.                           %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Penalty parameter for divide step (controls number of change points).
        % Default 10.
        gamma (1, 1) double {mustBeNonnegative} = 10

        % Penalty parameter for conquer step (local refinement).
        % Default 1.
        zeta (1, 1) double {mustBeNonnegative} = 1

        % Grid size for divide step.
        % Default: computed as 4 * n / Delta_min * log^2(n).
        Q (1, 1) double {mustBeNonnegative, mustBeInteger}

        % Lasso penalty parameter for sparse estimation.
        % Default: C_lambda * sqrt(log(p or n)).
        lambda (1, 1) double {mustBeNonnegative}

        % Minimum interval length threshold.
        % Intervals shorter than this are assigned F = 0.
        % Default: C_F * s * log(p or n) for mean/linear, C_F * p * log(p or n) for GGM.
        min_interval_length (1, 1) double {mustBeNonnegative, mustBeInteger}

        % Constants for tuning.
        C_lambda (1, 1) double {mustBePositive} = 1
        C_F (1, 1) double {mustBePositive} = 1

        % Estimated sparsity level (for mean/linear models).
        s (1, 1) double {mustBePositive, mustBeInteger} = 5

        % Buffer for local refinement (minimum distance from window boundaries).
        % Default: same as min_interval_length.
        buffer_refine (1, 1) double {mustBeNonnegative, mustBeInteger}

        % Step size for local refinement search.
        % Default: 1 (search every position).
        step_refine (1, 1) double {mustBePositive, mustBeInteger} = 1

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% Results and intermediate variables.                         %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Grid points for divide step: s_i = floor(i * n / (Q + 1)), i = 1..Q.
        grid double

        % Preliminary change points from divide step.
        prelim_cps double

        % Final refined change points from conquer step.
        final_cps double

        % DP cost array for divide step.
        B double

        % DP backpointer array for divide step.
        p_back double

        % Estimated parameters for each segment.
        theta_segments cell
    end

    methods
        function obj = DCDP(opts)
            %% DCDP: Divide and Conquer Dynamic Programming for change point detection.
            % Implements the algorithm from Li, Wang, Rinaldo (2023).
            %
            % - Usage:
            %   Est = DCDP(X=X, model='mean', gamma=gamma, zeta=zeta, Q=Q);
            %   Est = DCDP(X=X, y=y, model='linear', gamma=gamma, zeta=zeta);
            %   Est = DCDP(X=X, model='ggm', gamma=gamma, zeta=zeta);
            %
            % - Note:
            %   1. Use explicit argument names like "X=X", not positional arguments.
            %   2. The order of input arguments does not matter.
            %   3. Skip arguments to use default values.
            %
            % - Input:
            %   @X:         Data matrix, size (n, p).
            %   @y:         Response vector for linear model, size (n, 1).
            %   @model:     Model type: 'mean', 'linear', or 'ggm'. Default 'mean'.
            %   @gamma:     Penalty for divide step, positive scalar. Default 10.
            %   @zeta:      Penalty for conquer step, positive scalar. Default 1.
            %   @Q:         Grid size. Default: auto-computed.
            %   @lambda:    Lasso penalty. Default: auto-computed.
            %   @s:         Sparsity level for mean/linear models. Default 5.
            %   @C_lambda:  Constant for lambda. Default 1.
            %   @C_F:       Constant for min interval length. Default 1.
            %   @buffer_refine: Buffer for local refinement. Default: auto.
            %   @step_refine: Step size for refinement. Default 1.

            arguments (Input)
                opts.X double
                opts.y double = []
                opts.model char {mustBeMember(opts.model, {'mean', 'linear', 'ggm'})} = 'mean'
                opts.gamma (1, 1) double {mustBeNonnegative} = 10
                opts.zeta (1, 1) double {mustBeNonnegative} = 1
                opts.Q (1, 1) double {mustBeNonnegative, mustBeInteger} = 0
                opts.lambda (1, 1) double {mustBeNonnegative} = 0
                opts.s (1, 1) double {mustBePositive, mustBeInteger} = 5
                opts.C_lambda (1, 1) double {mustBePositive} = 1
                opts.C_F (1, 1) double {mustBePositive} = 1
                opts.buffer_refine (1, 1) double {mustBeNonnegative, mustBeInteger} = 0
                opts.step_refine (1, 1) double {mustBePositive, mustBeInteger} = 1
            end

            % Store data.
            obj.X = opts.X;
            obj.y = opts.y;
            obj.model = opts.model;

            % Validate that y is provided for linear model.
            if strcmp(opts.model, 'linear') && isempty(opts.y)
                error('DCDP:MissingY', 'Response vector y must be provided for linear model.');
            end

            % Get dimensions.
            [obj.n, obj.p] = size(obj.X);

            % Store hyperparameters.
            obj.gamma = opts.gamma;
            obj.zeta = opts.zeta;
            obj.s = opts.s;
            obj.C_lambda = opts.C_lambda;
            obj.C_F = opts.C_F;

            % Set default lambda if not provided.
            if opts.lambda == 0
                obj.lambda = obj.C_lambda * sqrt(log(max(obj.p, obj.n)));
            else
                obj.lambda = opts.lambda;
            end

            % Set minimum interval length based on model.
            if strcmp(obj.model, 'ggm')
                obj.min_interval_length = ceil(obj.C_F * obj.p * log(max(obj.p, obj.n)));
            else
                obj.min_interval_length = ceil(obj.C_F * obj.s * log(max(obj.p, obj.n)));
            end

            % Set default Q if not provided.
            if opts.Q == 0
                % Default: Q = 4 * n / Delta_min * log^2(n)
                % Assume Delta_min ~ n/10 as a heuristic.
                obj.Q = ceil(40 * log(obj.n)^2);
            else
                obj.Q = opts.Q;
            end

            % Cap Q to avoid duplicate grid points when Q >= n.
            obj.Q = min(obj.Q, max(obj.n - 1, 0));

            % Set buffer_refine (default: same as min_interval_length).
            if opts.buffer_refine == 0
                obj.buffer_refine = obj.min_interval_length;
            else
                obj.buffer_refine = opts.buffer_refine;
            end

            % Set step_refine.
            obj.step_refine = opts.step_refine;

            % Build grid.
            % Grid points: s_i = floor(i * n / (Q + 1)) for i = 1..Q
            % Include endpoints 0 and n.
            obj.grid = [0; floor((1:obj.Q)' * obj.n / (obj.Q + 1)); obj.n];
        end

        % Run the full DCDP algorithm.
        run(obj)

        % Divide step: Divided Dynamic Programming.
        DDP(obj)

        % Conquer step: Penalized Local Refinement.
        PLR(obj)

        % Compute F(theta_hat, I) for an interval I.
        [F_val, theta_hat] = ComputeF(obj, I)

        % Compute penalty R for PLR step.
        R_val = ComputeR(obj, theta1, theta2, eta, s_k, e_k)

        % Estimate parameter on interval I.
        theta_hat = EstimateTheta(obj, I)

        % Estimate parameters for each segment after change points are obtained.
        EstimateSegmentParameters(obj)
    end

    methods (Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% Static helper functions.                                    %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function theta_hat = SoftThreshold(x, lambda)
            %% Soft-thresholding operator for lasso.
            %
            % - Input:
            %   @x:         Input vector.
            %   @lambda:    Threshold parameter.
            %
            % - Output:
            %   @theta_hat: Soft-thresholded result.

            theta_hat = sign(x) .* max(abs(x) - lambda, 0);
        end
    end
end
