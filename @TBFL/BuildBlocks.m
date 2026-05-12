function blocks = BuildBlocks(obj, block_size)
    %% Build block boundaries (LinearDetect-compatible).
    %
    % TBFL operates on a blocked version of the time axis. LinearDetect uses
    % a non-uniform blocking scheme with "guard" blocks near the boundary.
    % This reduces edge effects when fitting blocks and when selecting
    % candidate jump locations.
    %
    % This function mirrors the exact construction in LinearDetect:
    %
    %   b_n_bound = 2 * block.size
    %   blocks <- c(
    %     seq(1, b_n_bound * 2 + 1, b_n_bound),
    %     seq(b_n_bound * 2 + block.size + 1, TT + 1 - 2 * b_n_bound,
    %         block.size),
    %     seq(TT + 1 - b_n_bound, TT + 1, b_n_bound)
    %   )
    %
    % - Input:
    %   @block_size: Positive integer. Interior block length (b_n).
    %
    % - Output:
    %   @blocks: Row vector of block boundaries (1-indexed, includes T+1).
    %            Block b covers indices blocks(b):(blocks(b+1)-1).

    TT = obj.T;
    b_n_bound = 2 * block_size;

    % Guard blocks at the left boundary.
    part1 = 1:b_n_bound:(b_n_bound * 2 + 1);

    % Interior blocks with length `block_size`.
    start_mid = b_n_bound * 2 + block_size + 1;
    end_mid = TT + 1 - 2 * b_n_bound;
    if start_mid <= end_mid
        part2 = start_mid:block_size:end_mid;
    else
        part2 = [];
    end

    % Guard blocks at the right boundary.
    start_last = TT + 1 - b_n_bound;
    part3 = start_last:b_n_bound:(TT + 1);

    % Combine and deduplicate. unique() returns sorted boundaries, which is
    % consistent with the monotone construction in the R code.
    blocks = unique([part1, part2, part3]);

    % Defensive check. For typical settings, part3 always includes TT+1.
    if blocks(end) < TT + 1
        blocks = [blocks(1:end-1), TT + 1];
    end

    blocks = blocks(:)';
end
