function step_norms = realdata_step_norms(covariance_paths, T)
%% Frobenius norms of consecutive differences along each covariance path.
%
% The break diagnostics are the only consumer of the estimated covariance
% paths, and they plot ||Theta_t - Theta_{t-1}||_F. Reducing each
% p-by-p-by-T path to its T-1 step norms keeps the plot inputs at a few
% hundred kilobytes instead of several gigabytes.
%
% Usage:
%   step_norms = realdata_step_norms(method_thetas, T);
%
% Input:
%   @covariance_paths: Cell array of p-by-p-by-T covariance paths.
%   @T: Number of time observations.
%
% Output:
%   @step_norms: (T-1)-by-numel(covariance_paths) matrix whose (i, k)-th
%       entry is ||Theta^{(k)}_{i+1} - Theta^{(k)}_i||_F.

    covariance_paths = covariance_paths(:);
    num_methods = numel(covariance_paths);
    step_norms = nan(T - 1, num_methods);

    for kk = 1:num_methods
        Theta_k = covariance_paths{kk};
        if size(Theta_k, 3) ~= T
            error('realdata_step_norms:badPathLength', ...
                'Covariance path %d has T=%d but T=%d was expected.', ...
                kk, size(Theta_k, 3), T);
        end
        % Symmetrize before differencing; floating-point accumulation can
        % leave the estimated precision paths slightly asymmetric.
        for tt = 2:T
            Theta_curr = (Theta_k(:, :, tt) + Theta_k(:, :, tt)') / 2;
            Theta_prev = (Theta_k(:, :, tt - 1) + Theta_k(:, :, tt - 1)') / 2;
            step_norms(tt - 1, kk) = norm(Theta_curr - Theta_prev, 'fro');
        end
    end
end
