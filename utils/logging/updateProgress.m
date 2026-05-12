function updateProgress(total_, reset)
    % Update a single-line spinner + percentage progress bar in the command
    % window. Designed for DataQueue callbacks inside parfor loops, so it
    % aggressively rewrites the same line using backspaces to avoid flooding the
    % console.
    %
    % @total_: total number of tasks to track.
    % @reset:  optional flag (default=false). When true, resets counters without
    %          printing progress. Use before starting a new experiment.
    persistent spinner_idx last_len progress_count

    if nargin < 2
        reset = false;
    end

    % Initialize spinner position and last printed length once.
    if isempty(spinner_idx)
        spinner_idx = 1;
    end
    if isempty(last_len)
        last_len = 0;
    end
    if isempty(progress_count)
        progress_count = 0;
    end

    if reset
        progress_count = 0;
        spinner_idx = 1;
        last_len = 0;
        return;
    end

    % Bump completed counter and compute bar state.
    progress_count = progress_count + 1;
    pct = progress_count / total_ * 100;
    w = 60; % bar width characters
    filled = round(w * progress_count / total_);
    spinner = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏';
    bar = [repmat('█', 1, filled), repmat('░', 1, w - filled)];

    % Compose the line to print: bar + percent + spinner + counts.
    msg = sprintf('│%s│ %5.1f%% %s %4d/%d', bar, pct, spinner(spinner_idx), progress_count, total_);
    spinner_idx = mod(spinner_idx, numel(spinner)) + 1;

    % Erase previous characters then print the new message in-place.
    if last_len > 0
        fprintf(1, repmat('\b', 1, last_len));
    end
    fprintf(1, '%s', msg);
    last_len = length(msg);

    % Reset trackers on completion (keep bar on the same line).
    if progress_count == total_
        spinner_idx = 1;
        last_len = 0;
        progress_count = 0;
    end

    drawnow limitrate;
end
