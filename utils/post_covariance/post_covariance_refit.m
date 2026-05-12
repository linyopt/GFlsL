function Sigma_t = post_covariance_refit(X, breaks, method, opts)
%% Refit a covariance path using one competitor per segment.
%
% Usage:
%   Sigma_t = post_covariance_refit(X, breaks, "ec2");
%   Sigma_t = post_covariance_refit(X, breaks, "ec2", opts);
%
% Inputs:
%   X       T-by-p data matrix.
%   breaks  Break positions (possibly empty).
%   method  One of: "adaptive_threshold", "adaptive_threshold_fspd",
%           "xue_ma_zou", "ec2", "convex_banding", "sample_covariance".
%   opts    Optional struct. Missing fields default to:
%             center=true, min_eig=0.01, lambda=NaN, lambda_scale=1,
%             delta=2, rho=2.5, mu=2.5, maxiter=500, tol=1e-5.
%
% Output:
%   Sigma_t  p-by-p-by-T covariance path.

    if nargin < 4
        opts = struct();
    end
    opts = with_defaults(opts);

    [T, p] = size(X);
    breaks = sort(unique(round(breaks(:))));
    breaks = breaks(breaks >= 1 & breaks < T);
    bounds = build_segment_bounds(breaks, T);

    Sigma_t = zeros(p, p, T);
    for kk = 1:size(bounds, 1)
        lb = bounds(kk, 1);
        ub = bounds(kk, 2);
        X_seg = X(lb:ub, :);

        switch string(method)
            case "adaptive_threshold"
                Sigma_k = adaptive_threshold(X_seg, opts);
            case "adaptive_threshold_fspd"
                Sigma_k = adaptive_threshold_fspd(X_seg, opts);
            case "xue_ma_zou"
                Sigma_k = xue_ma_zou(X_seg, opts);
            case "ec2"
                Sigma_k = ec2(X_seg, opts);
            case "convex_banding"
                Sigma_k = convex_banding(X_seg, opts);
            case {"sample", "sample_covariance"}
                Sigma_k = sample_covariance(X_seg, opts);
            otherwise
                error("post_covariance_refit:unknown_method", ...
                    "Unknown method: %s", string(method));
        end

        Sigma_t(:, :, lb:ub) = repmat(Sigma_k, [1, 1, ub - lb + 1]);
    end
end

function opts = with_defaults(opts)
    defaults = struct( ...
        'center', true, 'min_eig', 0.01, ...
        'lambda', NaN, 'lambda_scale', 1, ...
        'delta', 2, 'rho', 2.5, 'mu', 2.5, ...
        'maxiter', 500, 'tol', 1e-5);
    f = fieldnames(defaults);
    for kk = 1:numel(f)
        if ~isfield(opts, f{kk})
            opts.(f{kk}) = defaults.(f{kk});
        end
    end
end
