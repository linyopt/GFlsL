function [Omega_hat, info] = EstimateOmega(obj, opts)
    %% Estimate segment-wise precision matrices Omega (GGM).
    %
    % TBFL is formulated in terms of nodewise regressions (neighborhood
    % selection). For a zero-mean Gaussian vector with precision matrix
    % Omega, the conditional mean is:
    %   E[X_j | X_{-j}] = -Omega(j,-j) / Omega(j,j) * X_{-j}.
    %
    % TBFL estimates, for each segment, a coefficient matrix Beta where
    % row j contains the regression coefficients of node j on nodes -j.
    % With an estimate of the conditional variance sigma_j^2, we can map
    % these quantities back to an estimated precision matrix:
    %   Omega(j,j)   = 1 / sigma_j^2
    %   Omega(j,-j)  = -Omega(j,j) * Beta(j,:)
    %
    % Since neighborhood selection is performed separately for each node,
    % the resulting Omega can be asymmetric. This interface optionally
    % symmetrizes Omega by averaging with its transpose.
    %
    % Prerequisites:
    % - Run `obj.run()` so that `obj.breaks` and `obj.beta_hat_list` exist.
    %
    % - Usage:
    %   [Omega_hat, info] = Est.EstimateOmega();
    %   Omega_1 = Omega_hat(:, :, 1);
    %
    % - Input:
    %   @symmetrize: How to symmetrize Omega. One of:
    %               "avg"  - (Omega + Omega')/2 (default).
    %               "none" - return the raw, possibly asymmetric Omega.
    %   @min_var:    Lower bound for residual variance to avoid division by
    %               zero. Default 1e-8.
    %   @center:     If true, center X within each segment before computing
    %               residual variances. Default false (match TBFL usage).
    %
    % - Output:
    %   @Omega_hat: p-by-p-by-m array of estimated precision matrices,
    %              where m = length(obj.breaks) + 1.
    %   @info:      Struct with diagnostic fields:
    %              - breaks: change points (sorted).
    %              - bounds: segment boundaries [1; breaks; T+1].
    %              - sigma2: p-by-m residual variances.
    %              - symmetrize: symmetrization mode used.

    arguments
        obj
        opts.symmetrize (1, 1) string ...
            {mustBeMember(opts.symmetrize, ["avg", "none"])} = "avg"
        opts.min_var (1, 1) double {mustBePositive} = 1e-8
        opts.center (1, 1) logical = false
    end

    if isempty(obj.beta_hat_list)
        error("TBFL:EstimateOmega:missingBetaHatList", ...
            "beta_hat_list is empty. Run Est.run() first.");
    end

    breaks = obj.breaks(:);
    breaks = breaks(~isnan(breaks));
    breaks = sort(unique(breaks));

    T = obj.T;
    p = obj.p;

    bounds = [1; breaks; T + 1];
    m_hat = numel(bounds) - 1;

    if numel(obj.beta_hat_list) ~= m_hat
        error("TBFL:EstimateOmega:badBetaHatListSize", ...
            "beta_hat_list must have %d segments; got %d.", ...
            m_hat, numel(obj.beta_hat_list));
    end

    Omega_hat = zeros(p, p, m_hat);
    sigma2 = zeros(p, m_hat);

    for seg = 1:m_hat
        lb = bounds(seg);
        ub = bounds(seg + 1) - 1;
        X_seg = obj.X(lb:ub, :);

        if opts.center
            X_seg = X_seg - mean(X_seg, 1);
        end

        beta_seg = obj.beta_hat_list{seg};
        if ~isequal(size(beta_seg), [p, p - 1])
            error("TBFL:EstimateOmega:badBetaSegmentSize", ...
                "beta_hat_list{%d} must be %d-by-%d.", seg, p, p - 1);
        end

        Beta_full = zeros(p, p);
        for j = 1:p
            idx_minus_j = [1:(j - 1), (j + 1):p];
            Beta_full(j, idx_minus_j) = beta_seg(j, :);
        end

        Y_hat = X_seg * Beta_full.';
        residual = X_seg - Y_hat;

        sigma2_seg = mean(residual .^ 2, 1).';
        sigma2_seg = max(sigma2_seg, opts.min_var);
        w_diag = 1 ./ sigma2_seg;

        Omega_raw = -diag(w_diag) * Beta_full;
        Omega_raw(1:(p + 1):end) = w_diag;

        if opts.symmetrize == "avg"
            Omega_seg = (Omega_raw + Omega_raw.') / 2;
            Omega_seg(1:(p + 1):end) = w_diag;
        else
            Omega_seg = Omega_raw;
        end

        Omega_hat(:, :, seg) = Omega_seg;
        sigma2(:, seg) = sigma2_seg;
    end

    info = struct();
    info.breaks = breaks;
    info.bounds = bounds;
    info.sigma2 = sigma2;
    info.symmetrize = opts.symmetrize;
end

