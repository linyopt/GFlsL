function plot(obj, opts)
    %% Plot the difference.
    arguments (Input)
        obj GFlsL

        % The figure ID, to control where to plot, default 998 to aovid
        % overwrite current figures.
        opts.figID {mustBePositive, mustBeInteger} = 998

        % The time slots of the time-series data, used as x axis.
        % For real data of econimic, it can be useful for verification.
        opts.time_ = 1:obj.T-1;
    end

    % Declare figure.
    figure(opts.figID);
    % Call built-in plot.
    plot(opts.time_, obj.norm_diff, '.-');
end