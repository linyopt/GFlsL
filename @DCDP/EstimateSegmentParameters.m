function EstimateSegmentParameters(obj)
    %% Estimate parameters for each segment after change points are obtained.
    % Uses the final change points to divide data into segments and estimates
    % parameters for each segment.
    %
    % - Usage:
    %   obj.EstimateSegmentParameters();
    %
    % - Output:
    %   Results are stored in obj.theta_segments (cell array of parameters).

    % Define segment boundaries based on final change points.
    % If final_cps = [cp1, cp2, ..., cpK], segments are:
    % [1, cp1], [cp1+1, cp2], ..., [cpK+1, n]

    if isempty(obj.final_cps)
        % No change points detected - entire data is one segment.
        num_segments = 1;
        segment_starts = 1;
        segment_ends = obj.n;
    else
        num_segments = length(obj.final_cps) + 1;
        segment_starts = [1; obj.final_cps + 1];
        segment_ends = [obj.final_cps; obj.n];
    end

    % Initialize cell array to store parameters for each segment.
    obj.theta_segments = cell(num_segments, 1);

    % Estimate parameters for each segment.
    for k = 1:num_segments
        I = segment_starts(k):segment_ends(k);
        obj.theta_segments{k} = obj.EstimateTheta(I);
    end
end
