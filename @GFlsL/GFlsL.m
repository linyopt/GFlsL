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
        
        % The augmented Lagrangian hyperparameter beta_ to balance the primal
        % and dual. It should be positive.
        % Default 1.
        beta_ (1, 1) double {mustBePositive} = 1
        
        % The tolerance to stop the algorithm.
        % Default 1e-2.
        tol (1, 1) double {mustBePositive, mustBeLessThan(tol, 1)} = 1e-2
        
        % Maximal iterations.
        % Default 5000.
        maxiter (1, 1) double {mustBeNumeric, mustBePositive} = 5000
        
        % Display frequency.
        % Default 1.
        disp_freq (1, 1) double {mustBePositive} = 1

        % Show figure or not.
        % Default false.
        showfig (1, 1) = false

        % Merge nearby estimated breaks after estimation.
        % Default false.
        merge_breaks (1, 1) logical = false

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

        % breaks
        breaks double
    end
    methods
        function obj = GFlsL(opts)
            %% The estimator based on least-squares covariance fitting.
            % - Usage:
            %   Est = GFlsL(X=X, lamb1=lamb1, lamb2=lamb2, beta_=beta_, 
            %              epsilon=epsilon, tol=tol, maxiter=maxiter,
            %              disp_freq=disp_freq, showfig=showfig,
            %              merge_breaks=merge_breaks);
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
            %   @beta_:     Argumented Lagrangian parameter, positive scalar,
            %               default 1.
            %   @epsilon:   Parameter for PSD constraint, positive scalar, 
            %               default 0.01.
            %   @tol:       Tolerance to stop the algorithm, positive scalar < 1, 
            %               default 1e-2.
            %   @maxiter:   Maximal iteration, positive scalar, default 10000.
            %   @disp_freq: Frequency for information display, default 1, 
            %               set to "inf" if you don't want any display.
            %   @showfig:   Showing or not showing figures of difference of 
            %               Theta per disp_freq, default false.
            %   @merge_breaks: Merge nearby estimated breaks, default false.

            arguments (Input)
                opts.X double
                opts.lamb1 {mustBeNumeric, mustBeNonnegative} = 0.1
                opts.lamb2 {mustBeNumeric, mustBeNonnegative} = 100
                opts.beta_ (1, 1) {mustBeNumeric, mustBePositive} = 1
                opts.epsilon (1, 1) {mustBeNumeric, mustBeNonnegative} = 0.01
                opts.tol (1, 1) {mustBeNumeric, mustBePositive} = 1e-2
                opts.maxiter (1, 1) {mustBeNumeric, mustBePositive} = 5000
                opts.disp_freq (1, 1) {mustBeNumeric} = 1
                opts.showfig (1, 1) = false
                opts.merge_breaks (1, 1) logical = false
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
            obj.tol = opts.tol;
            obj.maxiter = opts.maxiter;
            obj.disp_freq = opts.disp_freq;
            obj.showfig = opts.showfig;
            obj.merge_breaks = opts.merge_breaks;

            % Compute XXt.
            X_reshaped = reshape(obj.X, [1, obj.T, obj.p]);
            obj.XXt = bsxfun(@times, permute(X_reshaped, [3, 1, 2]), ...
                permute(X_reshaped, [1, 3, 2]));
            obj.XXt = (obj.XXt + permute(obj.XXt, [2, 1, 3])) / 2;

            % Compute sum of XtXt^T.
            sumXXt = sum(obj.XXt, 3);
            % Symmetrization.
            sumXXt = (sumXXt + sumXXt') / 2;
            % Compute the inverse of its matrix square root.
            [V, S] = eig(sumXXt);
            Sinv = 1 ./ diag(S);
            obj.inv_sq_XXt = V * diag(sqrt(Sinv)) * V';
            obj.inv_XXt = V * diag(Sinv) * V';

            % Construct the two sparse diagonal matrices used in Theta_up.
            diagonal = [3*obj.beta_ + 1 / (obj.T); ...
                        repmat(4*obj.beta_ + 1 / (obj.T), obj.T-2, 1); ...
                        3*obj.beta_ + 1 / (obj.T)];
            off_diagonal = repmat(-obj.beta_, obj.T, 1);
            obj.A = spdiags([off_diagonal, diagonal, off_diagonal], ...
                            [-1, 0, 1], obj.T, obj.T);
            obj.Adiag = spdiags([off_diagonal, diagonal - obj.beta_, ...
                                 off_diagonal], [-1, 0, 1], ...
                                 obj.T, obj.T);

            % Initialization.
            obj.Initialization();

            % Compute initilized objective value.
            obj.PrimObjVal();
            obj.GetDelta_();
            obj.DualObjVal();
        end

        % The function to call the algorithm to solve the problem.
        run(obj)
        
        % The function to initialize the primal / dual variables.
        Initialization(obj)

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

        % Plot.
        plot(obj, figID, time_)

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

            % The first-order difference of Theta.
            diff_Theta = diff(obj.Theta_k, 1, 3);
            % The tolerance.
            threshold_ = 1e-6;

            % Some helper indices.
            idx_diff = ~repmat(eye(obj.p), [1, 1, obj.T - 1]);
            idx_1 = 1:obj.p^2;
            % Compute freedom.
            K = sum(abs(diff_Theta(idx_diff)) >= threshold_) ...
                + sum(abs(obj.Theta_k(idx_1(~eye(obj.p)))) >= threshold_, 'all');
            % Compute AIC and BIC.
            obj.AIC = 2 * K + 2 * obj.lossval;
            obj.BIC = K * obj.p * log(obj.T) + obj.lossval;

            % Compute the Gaussian loss.
            % 1. Compute the eigendecomposition of Theta.
            [VTheta, DTheta] = pageeig(obj.Theta_k);
            % 2. Compute the inverse of Theta.
            invDTheta = DTheta;
            invDTheta(obj.idx) = 1 ./ invDTheta(obj.idx);
            invTheta = pagemtimes(pagemtimes(VTheta, invDTheta), ...
                                  permute(VTheta, [2, 1 ,3]));
            invTheta = (invTheta + permute(invTheta, [2, 1, 3])) / 2;
            % 3. Compute the Gaussian loss value.
            gau_loss = (sum(log(DTheta(obj.idx)), 'all') + ...
                sum(obj.XXt .* invTheta, 'all')) / obj.T;
            obj.EstBreaks;
            C_T = log(obj.T) * log(obj.p) / obj.T;
            alpha_T = log(obj.T) * obj.p / obj.T;

            % Breaks are stored as segment-end indices, so each later
            % segment starts at breaks(j) + 1.
            segment_starts = [1, obj.breaks(:)' + 1];
            edge_total = 0;
            for seg_ind = 1:length(segment_starts)
                E_j = sum(abs(triu(obj.Theta_k(:, :, segment_starts(seg_ind)))) >= threshold_, 'all');
                edge_total = edge_total + E_j;
            end

            obj.HBIC = obj.lossval + C_T * edge_total + alpha_T * length(obj.breaks);
            obj.HBICG = gau_loss + C_T * edge_total + alpha_T * length(obj.breaks);
        end

        function breaks = EstBreaks(obj)
            %% Use adpative threshold to estimate the breaks.
            % The breaks are obtained by picking those time slots whose D_k
            % are nonzero.
            %
            % - Usage:
            %   breaks = Est.EstBreaks;
            %   breaks % OR 
            %   Est.breaks

            tmp_norm = vecnorm(reshape(obj.D_k(:, :, 2:obj.T), [], obj.T-1));
            breaks = find(tmp_norm >= 1e-6)';
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
