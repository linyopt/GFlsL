function mergedBreaks = mergeBreaks(breaks)
%% Merge close elements in the array of breaks.
% As the algorithm will sometimes produce some very close breaks (e.g.,
% the difference between neighboring breaks is less than 5), this function
% merges these breaks into one.
% This function will be used to postprocess the breaks, and to improve the
% performance of the algorithm.
%
% - Usage:
%   mergedBreaks = mergeBreaks(breaks);
%
% - Input:
%   @breaks: The breaks.
%
% - Output:
%   @mergedBreaks:  The merged breaks.

    % Handle edge cases:
    % Check if the sequence is empty or has only one element.
    if isempty(breaks) || isscalar(breaks)
        mergedBreaks = breaks;
        return
    end

    % Calculate differences between consecutive elements.
    differences = diff(breaks);

    % Identify the start of new groups (where difference is 5 or more).
    % Include the first element as a new group start.
    groupStarts = find(differences >= 5) + 1;
    groupStarts = groupStarts(:);
    groupStarts = [1; groupStarts];

    % Assign group labels to each element.
    groupLabels = cumsum(ismember(1:length(breaks), groupStarts));

    % Calculate the median of each group and round it to the nearest integer.
    mergedBreaks = accumarray(groupLabels', breaks', [], @(x) round(median(x)));
end
