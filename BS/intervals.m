% Random interval generator for WBS/WBSIP.
function [Alpha, Beta] = intervals(M, lower, upper, minWidth)
%% intervals
% Generate random intervals for WBS/WBSIP.
%
% Sampling rule.
% Endpoints are sampled i.i.d. uniformly from {lower,...,upper}, then
% ordered so that Alpha(j) < Beta(j).
% Optionally enforce Beta(j)-Alpha(j) > minWidth (strict).
% 
% Usage:
%   [Alpha, Beta] = intervals(M, lower, upper)
%   [Alpha, Beta] = intervals(M, lower, upper, minWidth)
%
% Inputs:
% M: number of random intervals to return.
%   @M:     The number random intervals.
% lower: smallest admissible endpoint value.
%   @lower: The lower bound of random intervals.
% upper: largest admissible endpoint value.
%   @upper: The upper bound of random intervals.
% minWidth: strict lower bound on interval length (Beta-Alpha).
%   @minWidth: (optional) enforce Beta-Alpha > minWidth.
%
% Outputs:
% Alpha: vector of left endpoints (always strictly smaller than Beta).
%   @Alpha: The vector storing the starting indices of the random intervals.
% Beta: vector of right endpoints (always strictly larger than Alpha).
%   @Beta:  The vector storing the ending indices of the random intervals.

% Argument validation (defaults and types).
arguments (Input)
    M int32 = 100
    lower int32 = 1
    upper int32 = 10
    minWidth double = 0
end

% Validate that the endpoint range is non-empty.
if (lower >= upper)
    error("Integer 'lower' should be strictly smaller than integer 'upper'.");
end

% Validate that the requested minimum width is feasible given [lower, upper].
if minWidth >= double(upper - lower)
    error("minWidth must be smaller than (upper-lower).");
end

% Sample each interval independently until the strict width constraint is
% satisfied.
Alpha = zeros(1, double(M));
Beta = zeros(1, double(M));
lo = double(lower);
hi = double(upper);
for j = 1:double(M)
    while true
        a = randi([lo, hi], 1, 1);
        b = randi([lo, hi], 1, 1);
        % Reject degenerate intervals.
        if a == b
            continue;
        end
        % Order the endpoints so that Alpha(j) < Beta(j).
        aa = min(a, b);
        bb = max(a, b);
        % Enforce the strict minimum-width requirement.
        if (double(bb - aa) > minWidth)
            Alpha(j) = aa;
            Beta(j) = bb;
            break;
        end
    end
end

% Return endpoints as double to match downstream code expectations.
Alpha = double(Alpha);
Beta = double(Beta);

end
