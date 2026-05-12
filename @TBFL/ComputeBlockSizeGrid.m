function grid = ComputeBlockSizeGrid(obj)
    %% Compute block size grid (matching R LinearDetect).
    %
    % LinearDetect can automatically select the block size by evaluating
    % HBIC over a small candidate set.
    %
    % For GGM:
    %   b_n.max = ceiling(min(sqrt(T), T/20)) if sqrt(T) > p
    %           = ceiling(min(sqrt(T) * log(p), T/20)) otherwise
    %   b_n.min = floor(min(log(T) * log(p), T/20))
    %   grid = round(seq(b_n.min, b_n.max, length.out=5))
    %
    % - Output:
    %   @grid: Vector of block sizes to try.

    TT = obj.T;
    p_x = obj.p;

    if sqrt(TT) > p_x
        b_n_max = ceil(min(sqrt(TT), TT / 20));
    else
        b_n_max = ceil(min(sqrt(TT) * log(p_x), TT / 20));
    end

    b_n_min = floor(min(log(TT) * log(p_x), TT / 20));

    % Ensure b_n_min >= 2.
    b_n_min = max(2, b_n_min);
    b_n_max = max(b_n_min, b_n_max);

    % Generate 5 equally spaced values.
    grid = round(linspace(b_n_min, b_n_max, 5));
    grid = unique(grid);
    grid = grid(grid > 1);

    if isempty(grid)
        grid = max(2, floor(sqrt(TT)));
    end
end
