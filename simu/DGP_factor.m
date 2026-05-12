function [Y, Y_test, Sigma_tt, breaks, num_factors_blocks] = DGP_factor(T, p, m, prob, Nsim, regimes)
%% Generate data with sparse factor model covariance structure.
%
% This DGP generates multivariate Gaussian data where the covariance matrix
% in each regime follows a sparse factor model structure. The factor model
% assumes:
%   Sigma = Lambda * Lambda' + Psi
% where Lambda is the factor loading matrix and Psi is a diagonal matrix.
%
% Usage:
%   [Y, Y_test, Sigma_tt, breaks, num_factors_blocks] = DGP_factor(T, p, m, prob, Nsim, regimes)
%
% Input:
%   @T:       Sample size (number of time points).
%   @p:       Dimension (number of variables).
%   @m:       Number of break points. The data will have m+1 regimes.
%   @prob:    Sparsity parameter for the factor loading matrix. Can be a
%             scalar (same for all regimes) or a vector of length m+1.
%             Higher values lead to denser factor loadings.
%   @Nsim:    Number of simulated validation datasets. Default is 10.
%   @regimes: 'constant' or 'non-constant' to control regime lengths.
%             - 'constant': All regimes have approximately equal length.
%             - 'non-constant': Regime lengths are randomly determined
%               with a minimum length constraint.
%
% Output:
%   @Y:        Training data of size (T x p).
%   @Y_test:   Test/validation data of size (T x p x Nsim).
%   @Sigma_tt: True covariance matrices over time, size (p x p x T).
%   @breaks:   Vector of true break point locations (indices in 1..T-1).
%   @num_factors_blocks: True factor counts for the m+1 regimes.

arguments (Input)
    T (1, 1) double = 100
    p (1, 1) double = 10
    m (1, 1) double = 2
    prob double = 0.8
    Nsim (1, 1) int32 = 10
    regimes string = "non-constant"
end

assert(any(strcmp(regimes, {'constant', 'non-constant'})), ...
    "The 'regimes' must be 'constant' or 'non-constant'.");

% Number of distinct regimes (one more than the number of breaks).
Nblocks = m + 1;

% Minimum fraction of T that each regime must contain.
% This prevents degenerate cases with very short regimes.
eps = 1 / (m + 8);

% Handle the sparsity parameter: if scalar, replicate for all regimes.
len_prob = length(prob);
if len_prob == 1
    prob = prob * ones(Nblocks, 1);
else
    % If prob vector is shorter than Nblocks, pad with default value 0.8.
    if len_prob < Nblocks
        prob = [reshape(prob, [], 1); 0.8 * ones(Nblocks - len_prob, 1)];
    end
end

% Generate break point locations based on regime type.
switch regimes
    case 'non-constant'
        % Randomly generate break points ensuring minimum regime length.
        % Uses rejection sampling: keep generating until all regimes
        % satisfy the minimum length constraint (eps * T).
        cond = true;
        iter = 0;
        while cond
            iter = iter + 1;
            breaks = sort(randi([round(eps * T) + 1, T - round(eps * T)], m, 1));
            diff_breaks = breaks(2:end) - breaks(1:end-1);
            cond = (min(diff_breaks) < eps * T);
            % Safety valve to prevent infinite loops.
            if iter > 1e5
                break
            end
        end
    case 'constant'
        % Evenly spaced break points for equal-length regimes.
        len = round(T / (m + 1));
        breaks = ((1:m) * len)';
end

% Generate sparse factor model covariance for each regime.
% m_f: number of factors (randomly chosen between 2 and 6).
% sparsity: number of non-zero entries in the factor loading matrix.
Sigma = zeros(p, p, m + 1);
num_factors_blocks = zeros(m + 1, 1);
for k = 1:m + 1
    m_f = randi([2, 6]);
    num_factors_blocks(k) = m_f;
    sparsity = round(p * m_f * prob(k));
    Sigma(:, :, k) = simulate_sparse_factor(p, m_f, sparsity, 0.8);
end

% Generate training and test data from multivariate Gaussian distribution.
if m > 0
    % Case with at least one break point.
    Y = zeros(T, p);
    ind_ = [0; breaks; T];

    % Generate training data for each regime.
    for i = 1:m + 1
        regime_start = ind_(i) + 1;
        regime_end = ind_(i + 1);
        regime_length = regime_end - regime_start + 1;
        Y(regime_start:regime_end, :) = mvnrnd(zeros(p, 1), ...
            Sigma(:, :, i), regime_length);
    end

    % Generate validation/test datasets.
    Y_test = zeros(T, p, Nsim);
    for tt = 1:Nsim
        for i = 1:m + 1
            regime_start = ind_(i) + 1;
            regime_end = ind_(i + 1);
            regime_length = regime_end - regime_start + 1;
            Y_test(regime_start:regime_end, :, tt) = mvnrnd(zeros(p, 1), ...
                Sigma(:, :, i), regime_length);
        end
    end

    % Construct time-varying covariance matrix Sigma_tt.
    % Each time point t gets the covariance of its corresponding regime.
    Sigma_tt = zeros(p, p, T);
    Sigma_tt(:, :, 1:breaks(1)) = repmat(Sigma(:, :, 1), [1, 1, breaks(1)]);
    for t = 2:m
        regime_start = breaks(t - 1) + 1;
        regime_end = breaks(t);
        regime_length = regime_end - regime_start + 1;
        Sigma_tt(:, :, regime_start:regime_end) = ...
            repmat(Sigma(:, :, t), [1, 1, regime_length]);
    end
    Sigma_tt(:, :, breaks(m) + 1:end) = ...
        repmat(Sigma(:, :, m + 1), [1, 1, T - breaks(m)]);
else
    % Case with no breaks (single regime).
    Y = mvnrnd(zeros(p, 1), Sigma(:, :, 1), T);

    Y_test = zeros(T, p, Nsim);
    for tt = 1:Nsim
        Y_test(:, :, tt) = mvnrnd(zeros(p, 1), Sigma(:, :, 1), T);
    end

    % Single covariance matrix replicated across all time points.
    Sigma_tt = zeros(p, p, T);
    Sigma_tt(:, :, :) = repmat(Sigma(:, :, 1), [1, 1, T]);
end

end
