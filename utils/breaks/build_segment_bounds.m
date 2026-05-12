function bounds = build_segment_bounds(breaks, T)
%% Build segment bounds from estimated break locations.
%
% Breaks are interpreted as the last index of a segment. For example,
% breaks = [40; 90] on a series of length T = 120 yields segments
% [1, 40], [41, 90], and [91, 120].

    % The no-break case is a single segment covering the full series.
    if isempty(breaks)
        bounds = [1, T];
        return
    end

    % Normalize the break vector so every downstream caller sees the same
    % sorted, unique, interior break convention.
    breaks = sort(unique(round(breaks(:))));
    breaks = breaks(breaks >= 1 & breaks < T);

    % Convert break endpoints into inclusive segment bounds.
    edges = [0; breaks; T];
    bounds = [edges(1:end - 1) + 1, edges(2:end)];
end
