% Covariance CUSUM matrix S_t^{s,e} from Definition 1 (Wang, Yu, Rinaldo, 2018).
function S = CUSUM_cov(X, s, e, t)
%% CUSUM_cov
% Covariance CUSUM statistic from Wang, Yu, Rinaldo (2018), Definition 1.
%
% Data layout and indexing (paper convention).
% X is p x n with columns X_1,...,X_n.
% Endpoints satisfy 0 <= s < t < e <= n.
% The interval (s,e) contains samples {X_i}_{i=s+1,...,e}.
% The split point t partitions into {s+1,...,t} and {t+1,...,e}.
%
% Matrix formula (paper notation).
% S_t^{s,e} =
%   sqrt((e-t)/((e-s)(t-s))) * sum_{i=s+1}^t X_i X_i'
% - sqrt((t-s)/((e-s)(e-t))) * sum_{i=t+1}^e X_i X_i'

    % Validate endpoints and split point under the paper indexing convention.
    if s < 0 || e > size(X, 2) || ~(s < t && t < e)
        error('CUSUM_cov:invalidIndices', 'Require 0 <= s < t < e <= n.');
    end

    % Segment sizes.
    denom = double(e - s);
    leftN = double(t - s);
    rightN = double(e - t);

    % Second-moment sums on the left and right segments.
    X_left = X(:, (s + 1):t);
    X_right = X(:, (t + 1):e);

    leftSum = X_left * X_left.';
    rightSum = X_right * X_right.';

    % Combine with the paper’s CUSUM weights to produce S_t^{s,e}.
    % Weight for the left second-moment sum in Definition 1.
    wLeft = sqrt(rightN / (denom * leftN));
    % Weight for the right second-moment sum in Definition 1.
    wRight = sqrt(leftN / (denom * rightN));
    % Weighted difference of second-moment sums.
    S = wLeft * leftSum - wRight * rightSum;
end
