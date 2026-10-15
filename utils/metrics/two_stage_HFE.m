function resHFE = two_stage_HFE(HDs, F1s, errors, accuracy, breaks)
%% Find minimum HD -> maximum F1 -> minimum error.

    % The result corresponding to this criterion, it will contain
    %   HD, F1, error, accuracy, number of breaks.
    resHFE = zeros(1, 5);

    % Ignore unevaluated candidates: metric arrays are Inf-initialized and
    % stage-1 stop lamb2 columns carry Inf/NaN, which would otherwise win
    % the min(HD) step and produce null-metric rows.
    valid = isfinite(HDs(:)) & isfinite(F1s(:)) & isfinite(errors(:)) & isfinite(accuracy(:));
    HDs = HDs(valid);
    F1s = F1s(valid);
    errors = errors(valid);
    accuracy = accuracy(valid);
    breaks = breaks(valid);
    if isempty(HDs)
        resHFE(:) = NaN;
        return;
    end

    % Step 1: Find the minimum value in HDs and its indices.
    resHFE(1) = min(HDs(:));
    indHD = find(HDs == resHFE(1));
    
    % Step 2: From the indHD, find the maximum value in F1s.
    [resHFE(2), indF1InHD] = max(F1s(indHD));
    indF1 = indHD(indF1InHD);
    
    % Step 3: From the indF1, find the minimum value in errors.
    [resHFE(3), indEInF1] = min(errors(indF1));
    indE1 = indF1(indEInF1);
    
    % Return the first one to avoid there are multiple results.
    % This will be the obtained index.
    idx = indE1(1);
    % Accuracy.
    resHFE(4) = accuracy(idx);
    % Number of breaks.
    resHFE(5) = length(breaks{idx});
end