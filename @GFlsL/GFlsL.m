classdef GFlsL < handle
    properties
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% The data-related stuff.                                     %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The sample size.
        T double {mustBePositive, mustBeInteger}
        
        % The dimension of the vector in each time slot.
        p double {mustBePositive, mustBeInteger}
        
        % The time series data of size (d, T).
        X double {mustBeNumeric}
        
        % The covariance matrices obtained by X. The size of XXt is (d, d, T).
        XXt double {mustBeNumeric}

        % Constant loss target in the Theta-update right-hand side.
        XXt_over_T double {mustBeNumeric}

        % The inverse of matrix square root of sum of XXt.
        inv_sq_XXt double {mustBeNumeric}

        % The inverse of sum of XXt.
        inv_XXt double {mustBeNumeric}
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% The hyperparameters and settings.                           %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The hyperparameter lamb1 for the lasso regularizer.
        % Default 0.1.
        lamb1 double {mustBeNonnegative} = 0.1
        
        % The hyperparameter lamb2 for the fused lasso regularizer.
        % Default 100.
        lamb2 double {mustBePositive} = 100
        
        % The hyperparameter epsilon used as a small ridge term to stabilize
        % the least-squares covariance fitting; should be nonnegative.
        % Default 0.01
        epsilon (1, 1) double {mustBeNonnegative} = 0.01
        
        % The augmented Lagrangian hyperparameter beta_ sets the overall
        % penalty scale. The three constraint blocks may be preconditioned by
        % fixed positive multipliers in penalty_scale.
        % Default 1.
        beta_ (1, 1) double {mustBePositive} = 1

        % Relative penalties for the V, Upsilon, and D constraints. A fixed
        % positive block-diagonal metric is a standard preconditioned ADMM;
        % [1, 1, 1] recovers the original iteration.
        penalty_scale (1, 3) double {mustBePositive} = [1, 1, 1]

        % Relaxation factor in the dual ascent. The convergence proof used by
        % this implementation permits 0 < relaxation < (1 + sqrt(5)) / 2.
        relaxation (1, 1) double {mustBePositive} = 1.61

        % Penalties after applying penalty_scale to beta_.
        beta_v (1, 1) double {mustBePositive} = 1
        beta_u (1, 1) double {mustBePositive} = 1
        beta_d (1, 1) double {mustBePositive} = 1
        
        % The tolerance to stop the algorithm.
        % Default 1e-2.
        tol (1, 1) double {mustBePositive, mustBeLessThan(tol, 1)} = 1e-2
        
        % Maximal iterations.
        % Default 5000.
        maxiter (1, 1) double {mustBeNumeric, mustBePositive} = 5000
        
        % Display frequency.
        % Default 1.
        disp_freq (1, 1) double {mustBePositive} = 1

        % Merge nearby estimated breaks after estimation.
        % Default false.
        merge_breaks (1, 1) logical = false

        % Compute stopping diagnostics every termination_check_freq
        % iterations. Default 1 matches the original algorithm.
        termination_check_freq (1, 1) double {mustBePositive, mustBeInteger} = 1

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% The followings are declarations of primal / dual variables. %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Primal variables.
        % Theta
        Theta_k double
        % V
        V_k double
        % Upsilon
        Upsilon_k double
        % D_k
        D_k double
        
        % Dual variables.
        % Y
        Y_k double
        % Z
        Z_k double
        % A
        A_k double
        % Auxiliary variable delta_.
        delta_ double

        % Norm of first-order difference of Theta.
        norm_diff double

        % Helper variable indicating indices of diagonal elements.
        idx

        % The two sparse diagonal matrices used in Theta_up, which are
        % fixed after initialization.
        A double % The one for off-diagonal elements.
        Adiag double % The one for diagonal elements.
        Ainv double % Dense inverse for the off-diagonal Theta update, small T only.
        Adiaginv double % Dense inverse for the diagonal Theta update, small T only.
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% The followings are lossfunc / primval / dualval.            %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The loss function value.
        lossval (1, 1) double = inf
        
        % The primal objective value.
        primval (1, 1) double = inf
        
        % The dual objective value.
        dualval (1, 1) double = inf

        % The AIC value.
        AIC (1, 1) double = inf

        % The BIC value.
        BIC (1, 1) double = inf

        % The HBIC value.
        HBIC (1, 1) double = inf
        HBICG (1, 1) double = inf

        % The Gaussian negative log-likelihood fit term used by HBICG.
        gauss_loss (1, 1) double = inf

        % Total model dimension K: active parameters summed over segments.
        model_dim (1, 1) double = 0

        % breaks
        breaks double
    end
    methods
        function obj = GFlsL(opts)
            %% The estimator based on least-squares covariance fitting.
            % - Usage:
            %   Est = GFlsL(X=X, lamb1=lamb1, lamb2=lamb2, beta_=beta_, 
            %              epsilon=epsilon, tol=tol, maxiter=maxiter,
            %              disp_freq=disp_freq, merge_breaks=merge_breaks);
            % - Note:
            %   1. Please declare arguments in an explicit way like "X=X", 
            %      do not use positional arguments.
            %   2. The order of input arguments does not matter, i.e., 
            %      the following two ways are equivalent:
            %      >> Est = GFlsL(X=X, lamb1=lamb1);
            %      >> Est = GFlsL(lamb1=lamb1, X=X);
            %   3. Just skip the arguments that you want it to use the default
            %      value. For example, skipping "lamb2" will let "lamb2" be
            %      1e-1 by default.
            %   4. Normally, you only need to provide
            %   X, lamb1, lamb2, epsilon, tol, beta_, maxiter, disp_freq.
            %
            % - Input:
            %   @X:         Data, a Txp matrix.
            %   @lamb1:     lamb1, nonnegative scalar or array of size
            %   pxpxT, default 0.1.
            %   @lamb2:     lamb2, positive scalar or array of size T-1, default 100.
            %   @beta_:     Overall augmented-Lagrangian penalty scale,
            %               positive scalar, default 1.
            %   @penalty_scale: Relative penalties for the V, Upsilon, and D
            %               constraints, positive 1x3 vector, default [1 1 1].
            %               Fixed unequal values use a block-diagonal
            %               preconditioned ADMM without changing the estimator.
            %   @relaxation: Dual-ascent relaxation in (0, (1+sqrt(5))/2),
            %               default 1.61.
            %   @epsilon:   Parameter for PSD constraint, positive scalar, 
            %               default 0.01.
            %   @tol:       Tolerance to stop the algorithm, positive scalar < 1, 
            %               default 1e-2.
            %   @maxiter:   Maximal iteration, positive scalar, default 10000.
            %   @disp_freq: Frequency for information display, default 1, 
            %               set to "inf" if you don't want any display.
            %   @merge_breaks: Merge nearby estimated breaks, default false.

            arguments (Input)
                opts.X double
                opts.lamb1 {mustBeNumeric, mustBeNonnegative} = 0.1
                opts.lamb2 {mustBeNumeric, mustBeNonnegative} = 100
                opts.beta_ (1, 1) {mustBeNumeric, mustBePositive} = 1
                opts.penalty_scale (1, 3) double {mustBePositive} = [1, 1, 1]
                opts.relaxation (1, 1) {mustBeNumeric, mustBePositive} = 1.61
                opts.epsilon (1, 1) {mustBeNumeric, mustBeNonnegative} = 0.01
                opts.tol (1, 1) {mustBeNumeric, mustBePositive} = 1e-2
                opts.maxiter (1, 1) {mustBeNumeric, mustBePositive} = 5000
                opts.disp_freq (1, 1) {mustBeNumeric} = 1
                opts.merge_breaks (1, 1) logical = false
                opts.termination_check_freq (1, 1) {mustBeNumeric, mustBePositive, mustBeInteger} = 1
                opts.init_state = []
            end

            % Data.
            obj.X = opts.X;
            % Obtain dimension and sample size.
            [obj.T, obj.p] = size(obj.X);

            % Helper variable indicating indices of diagonal elements.
            obj.idx = (1 : obj.p+1 : obj.p^2).' + obj.p^2 .* (0:obj.T-1);
            
            % For lamb1, check the size.
            % If the input is a scalar, then we use an all-same array of
            % size pxpxT, which means we do not use adaptive weight of the
            % off-diagonal 1-norm regularizer.
            if isscalar(opts.lamb1)
                obj.lamb1 = opts.lamb1 * ones(obj.p, obj.p, obj.T);
            % If the input is an array of size pxpxT.
            elseif isequal(size(opts.lamb1), [obj.p, obj.p, obj.T])
                % Check if it is symmetric.
                % Compute the norms of differences of lamb1 and its
                % transport.
                diff_ = vecnorm(reshape(opts.lamb1 - permute(opts.lamb1, ...
                                                [2, 1, 3]), [], obj.T));
                % If the maximum norm of difference is relatively small,
                % then just symmetrize lamb1 and use it.
                if max(diff_) <= 1e-5
                    % Symmetrization.
                    obj.lamb1 = (opts.lamb1 + permute(opts.lamb1, [2, 1, 3])) / 2;

                    % Let the diagonal elements be 0.
                    obj.lamb1(obj.idx) = 0;
                % Otherwise, raise an error.
                else
                    error(['lamb1 must be a scalar or an array of size ',...
                    '%dx%dx%d where each matrix should be symmetric.'], ...
                                                     obj.p, obj.p, obj.T);
                end
            % Otherwise, raise an error.
            else            
                error(['lamb1 must be a scalar or an array of size ', ...
                    '%dx%dx%d where each matrix should be symmetric.'], ...
                                                    obj.p, obj.p, obj.T);
            end

            % For lamb2, check the size.
            % If the input is a scalar, then we use an all-same array of
            % size 1x(T-1), which means we do not use adaptive weight of
            % the group fused lasso regualrizer.
            if isscalar(opts.lamb2)
                obj.lamb2 = opts.lamb2 * ones(1, obj.T - 1);
            % If the input is an array of size 1x(T-1), then just use it.
            elseif isequal(size(opts.lamb2), [1, obj.T - 1])
                 obj.lamb2 = opts.lamb2;
            % If the input is an array of size (T-1)x1, then transport it.
            elseif isequal(size(opts.lamb2), [obj.T - 1, 1])
                 obj.lamb2 = opts.lamb2'; 
            % Otherwise, raise an error.
            else
                error("lamb2 musst be a scalar or a matrix of size 1x(%d - 1).", obj.T);
            end

            % Other parameters and options.
            obj.epsilon = opts.epsilon;
            obj.beta_ = opts.beta_;
            obj.penalty_scale = opts.penalty_scale;
            obj.beta_v = obj.beta_ * obj.penalty_scale(1);
            obj.beta_u = obj.beta_ * obj.penalty_scale(2);
            obj.beta_d = obj.beta_ * obj.penalty_scale(3);
            obj.relaxation = opts.relaxation;
            if obj.relaxation >= (1 + sqrt(5)) / 2
                error('GFlsL:relaxation', ...
                    'relaxation must be below (1 + sqrt(5)) / 2.');
            end
            obj.tol = opts.tol;
            obj.maxiter = opts.maxiter;
            obj.disp_freq = opts.disp_freq;
            obj.merge_breaks = opts.merge_breaks;
            obj.termination_check_freq = opts.termination_check_freq;

            % Compute XXt.
            X_reshaped = reshape(obj.X, [1, obj.T, obj.p]);
            obj.XXt = bsxfun(@times, permute(X_reshaped, [3, 1, 2]), ...
                permute(X_reshaped, [1, 3, 2]));
            obj.XXt = (obj.XXt + permute(obj.XXt, [2, 1, 3])) / 2;
            obj.XXt_over_T = obj.XXt / obj.T;

            % Compute sum of XtXt^T.
            sumXXt = sum(obj.XXt, 3);
            % Symmetrization.
            sumXXt = (sumXXt + sumXXt') / 2;
            % Compute the inverse of its matrix square root.
            [V, S] = eig(sumXXt);
            Sinv = 1 ./ diag(S);
            obj.inv_sq_XXt = V * diag(sqrt(Sinv)) * V';
            obj.inv_XXt = V * diag(Sinv) * V';

            % Construct the two sparse tridiagonal matrices used in
            % Theta_Up. The three splittings use a fixed block-diagonal
            % penalty metric: V acts on every entry, Upsilon only on
            % off-diagonal entries, and D on each temporal neighbour.
            degree_d = [1; repmat(2, obj.T - 2, 1); 1];
            diagonal_d = 1 / obj.T + degree_d * obj.beta_d + obj.beta_v;
            diagonal_off = diagonal_d + obj.beta_u;
            off_diagonal = repmat(-obj.beta_d, obj.T, 1);
            obj.A = spdiags([off_diagonal, diagonal_off, off_diagonal], ...
                            [-1, 0, 1], obj.T, obj.T);
            obj.Adiag = spdiags([off_diagonal, diagonal_d, off_diagonal], ...
                                [-1, 0, 1], obj.T, obj.T);
            % Applying a dense inverse costs O(T^2 p^2) per iteration but
            % runs in multithreaded BLAS, which pays off only for short
            % panels. Past the crossover Theta_Up solves the banded systems
            % directly, and the inverses are left empty.
            if obj.T <= 400
                obj.Ainv = full(obj.A \ eye(obj.T));
                obj.Adiaginv = full(obj.Adiag \ eye(obj.T));
            else
                obj.Ainv = [];
                obj.Adiaginv = [];
            end

            % Initialization. When init_state provides the full primal/dual
            % state (a converged solution of a neighboring lambda), skip the
            % data-driven initialization and start from it instead.
            if isempty(opts.init_state)
                obj.Initialization();
            else
                st = opts.init_state;
                if ~isfield(st, 'penalty_scale')
                    if any(obj.penalty_scale ~= [1, 1, 1])
                        error('GFlsL:initStatePenalty', ...
                            ['init_state has no penalty_scale, so it can only ' ...
                            'warm-start the default [1 1 1] metric.']);
                    end
                elseif ~isequal(reshape(st.penalty_scale, 1, 3), ...
                        obj.penalty_scale)
                    state_scale = reshape(st.penalty_scale, 1, 3);
                    error('GFlsL:initStatePenalty', ...
                        ['init_state was produced with penalty_scale [%g %g %g], ' ...
                        'but the new fit uses [%g %g %g].'], ...
                        state_scale(1), state_scale(2), state_scale(3), ...
                        obj.penalty_scale(1), obj.penalty_scale(2), ...
                        obj.penalty_scale(3));
                end
                if ~isfield(st, 'relaxation')
                    if obj.relaxation ~= 1.61
                        error('GFlsL:initStateRelaxation', ...
                            ['init_state has no relaxation, so it can only ' ...
                            'warm-start the default 1.61 setting.']);
                    end
                elseif st.relaxation ~= obj.relaxation
                    error('GFlsL:initStateRelaxation', ...
                        'init_state and new fit use different relaxation.');
                end
                obj.Theta_k = st.Theta_k;
                obj.V_k = st.V_k;
                obj.Upsilon_k = st.Upsilon_k;
                obj.D_k = st.D_k;
                obj.A_k = st.A_k;
                obj.Y_k = st.Y_k;
                obj.Z_k = st.Z_k;
            end

            % Compute initilized objective value.
            obj.PrimObjVal();
            obj.GetDelta_();
            obj.DualObjVal();
        end

        % The function to call the algorithm to solve the problem.
        run(obj)
        
        % The function to initialize the primal / dual variables.
        Initialization(obj)

        function st = solver_state(obj)
            %% Return the full primal/dual state for warm-starting a
            %% neighboring-lambda fit via the init_state constructor option.
            st = struct('Theta_k', obj.Theta_k, 'V_k', obj.V_k, ...
                'Upsilon_k', obj.Upsilon_k, 'D_k', obj.D_k, ...
                'A_k', obj.A_k, 'Y_k', obj.Y_k, 'Z_k', obj.Z_k, ...
                'penalty_scale', obj.penalty_scale, ...
                'relaxation', obj.relaxation);
        end

        % The function to compute the loss function.
        val = lossfunc(obj, X)

        % The function to compute the primal objective value.
        PrimObjVal(obj)

        % The function to compute the dual objective value.
        DualObjVal(obj)

        % The function to update Theta.
        Theta_Up(obj)

        % The function to update V.
        V_Up(obj)

        % The function to update Upsilon.
        Upsilon_Up(obj)

        % The function to update D.
        D_Up(obj)
        
        % The function to obtain auxiliary variable delta_.
        GetDelta_(obj)

        % The function to compute dual infeasibility.
        dfeas = DualFeas(obj)

        function GetICs(obj)
            %% Compute information criteria.
            % - Usage:
            %   Est.GetICs;
            %
            % Normally, this function needn't be called by the user as it
            % will be called after the function `run' is finished and the
            % problem is feasible.
            % If the function `run' is interupted by the user, then
            % `GetICs' should be called to compute the information
            % criteria.

            % Use one common model dimension for every information
            % criterion. Each segment contributes p diagonal parameters and
            % one parameter per active upper-triangular off-diagonal entry.
            threshold_ = 1e-6;
            obj.EstBreaks;
            segment_starts = [1, obj.breaks(:)' + 1];
            model_dimension = 0;
            for seg_ind = 1:length(segment_starts)
                Theta_seg = obj.Theta_k(:, :, segment_starts(seg_ind));
                model_dimension = model_dimension + sum( ...
                    abs(triu(Theta_seg)) >= threshold_, 'all');
            end
            % Compute AIC and BIC.
            obj.model_dim = model_dimension;
            obj.AIC = 2 * model_dimension + 2 * obj.lossval;
            obj.BIC = model_dimension * obj.p * log(obj.T) + obj.lossval;

            % X_tX_t' is rank one, so
            %   tr(X_tX_t' inv(Theta_t)) = x_t' inv(Theta_t) x_t.
            % Apply the eigenvectors to one vector per page instead of
            % forming the inverse matrices.
            [VTheta, DTheta] = pageeig(obj.Theta_k);
            eigvals = reshape(DTheta(obj.idx), obj.p, obj.T);
            proj = pagemtimes(permute(VTheta, [2, 1, 3]), ...
                reshape(obj.X', obj.p, 1, obj.T));
            quad = sum(reshape(proj, obj.p, obj.T) .^ 2 ./ eigvals, 1);
            gau_loss = (sum(log(eigvals), 'all') + sum(quad)) / obj.T;
            C_T = log(obj.T) * log(obj.p) / obj.T;
            alpha_T = log(obj.T) * obj.p / obj.T;

            obj.gauss_loss = gau_loss;
            obj.HBIC = obj.lossval + C_T * model_dimension + ...
                alpha_T * length(obj.breaks);
            obj.HBICG = gau_loss + C_T * model_dimension + ...
                alpha_T * length(obj.breaks);
        end

        function breaks = EstBreaks(obj)
            %% Estimate change points from the fitted covariance path.
            % A split variable can satisfy its proximal active-set condition
            % before the equality residual is small enough to identify the
            % fitted path. Read change points from the actual jumps of Theta
            % and scale the threshold by the solver accuracy and path norm.

            jumps = vecnorm(reshape(diff(obj.Theta_k, 1, 3), [], ...
                obj.T - 1));
            path_scale = max(vecnorm(reshape(obj.Theta_k, [], obj.T)));
            breaks = find(jumps > obj.tol * path_scale);
            if obj.merge_breaks
                breaks = mergeBreaks(breaks);
                breaks = breaks(:)';
            end

            obj.breaks = breaks;
        end
    end

    methods (Static)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% The followings are declarations of helper functions as      %%%%
        %%%% static functions.                                           %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function Xvec = vec(X)
            %% Given a high dimensional array X, return the vectorized column vector.
            % 
            % - Input:
            %   @X:     A high dimensional array.
            %
            % - Output:
            %   @Xvec:  A vectorized column vector.

            Xvec = X(:);
        end
    end
end
