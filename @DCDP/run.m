function run(obj)
    %% Run the full DCDP algorithm.
    % Executes divide step (DDP) followed by conquer step (PLR).
    %
    % - Usage:
    %   Est = DCDP(X=X, model='mean', gamma=10, zeta=1);
    %   Est.run();
    %   change_points = Est.final_cps;
    %   parameters = Est.theta_segments;
    %
    % - Output:
    %   Results are stored in obj.prelim_cps (preliminary change points),
    %   obj.final_cps (refined change points), and obj.theta_segments
    %   (estimated parameters for each segment).

    % Step 1: Divide - run DDP on grid.
    obj.DDP();

    % Step 2: Conquer - refine each preliminary change point.
    obj.PLR();

    % Step 3: Estimate parameters for each segment.
    obj.EstimateSegmentParameters();
end
