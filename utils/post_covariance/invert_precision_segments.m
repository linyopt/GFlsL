function Sigma_t = invert_precision_segments(Omega_segments, T, breaks, opts)
%% Convert regime-wise precision matrices into a covariance path.
%
% - Usage:
%   Sigma_t = invert_precision_segments(Omega_segments, T, breaks, opts);
%
% - Output:
%   @Sigma_t: Covariance path of size p-by-p-by-T.

    arguments
        Omega_segments double
        T (1, 1) double {mustBePositive, mustBeInteger}
        breaks double = []
        opts.min_eig (1, 1) double {mustBePositive} = 1e-4
    end

    % Check that the supplied precision tensor and the break vector imply
    % the same number of segments.
    [p, ~, num_segments] = size(Omega_segments);
    bounds = build_segment_bounds(breaks, T);

    if size(bounds, 1) ~= num_segments
        error("post_covariance:invert_precision_segments:badSegmentCount", ...
            "Omega_segments has %d segments but breaks imply %d segments.", ...
            num_segments, size(bounds, 1));
    end

    % Preallocate the expanded time-indexed covariance path.
    Sigma_t = zeros(p, p, T);

    % Invert one precision matrix per segment. We project to a positive
    % definite matrix first, then prefer Cholesky-based linear solves over
    % inv() for numerical stability.
    for kk = 1:num_segments
        Omega_k = project_min_eig(Omega_segments(:, :, kk), opts.min_eig);
        Omega_k = (Omega_k + Omega_k') / 2;

        [R, flag] = chol(Omega_k);
        if flag == 0
            Sigma_k = R \ (R' \ eye(p));
        else
            % This fallback should be rare because project_min_eig has
            % already enforced a positive eigenvalue floor.
            Sigma_k = inv(Omega_k);
        end
        Sigma_k = (Sigma_k + Sigma_k') / 2;

        lb = bounds(kk, 1);
        ub = bounds(kk, 2);
        Sigma_t(:, :, lb:ub) = repmat(Sigma_k, [1, 1, ub - lb + 1]);
    end
end
