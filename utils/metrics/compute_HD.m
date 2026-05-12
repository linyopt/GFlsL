function HD = compute_HD(trueBreaks, estimatedBreaks)
%% Compute Hausdorff distance between true and estimated breaks.
%
% - Usage:
%   HD = compute_HD(trueBreaks, estimatedBreaks)
%
% - Input:
%   @trueBreaks:      The true breaks (can be empty).
%   @estimatedBreaks: The estimated breaks (can be empty).
%
% - Output:
%   @HD: The Hausdorff distance.

% Convert to column vectors to ensure consistent dimensions
trueBreaks = trueBreaks(:);
estimatedBreaks = estimatedBreaks(:);

% Handle empty cases
if isempty(trueBreaks)
    if isempty(estimatedBreaks)
        HD = 0;
    else
        HD = max(estimatedBreaks);
    end
else
    if isempty(estimatedBreaks)
        HD = max(trueBreaks);
    else
        HD = HausdorffDist(trueBreaks, estimatedBreaks);
    end
end
end
