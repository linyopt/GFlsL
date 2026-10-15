function chains = build_warm_chains(missing_idx, idxcomb, n_chains)
%% Partition candidate indices into warm-start chains.
%
% Candidates sharing the same lamb1 index form a base chain, ordered by
% the second lambda grid position descending (penalty from strong to
% weak), so each fit starts from the sparser neighboring solution. Each
% base chain is then split into contiguous segments so that the total
% number of chains reaches roughly n_chains for parallelism.
%
% - Input:
%   @missing_idx: Indices of the candidates still to be fitted.
%   @idxcomb:     Grid index pairs, one row per candidate.
%   @n_chains:    Desired number of chains.
%
% - Output:
%   @chains:      Cell array of index vectors, each fitted sequentially.

    lamb1_idx = idxcomb(missing_idx, 1);
    lamb2_idx = idxcomb(missing_idx, 2);
    u1 = unique(lamb1_idx);
    base = cell(numel(u1), 1);
    for uu = 1:numel(u1)
        sel = missing_idx(lamb1_idx == u1(uu));
        [~, ord] = sort(lamb2_idx(lamb1_idx == u1(uu)), 'descend');
        base{uu} = sel(ord);
    end
    total_len = sum(cellfun(@numel, base));
    target = ceil(total_len / max(n_chains, 1));
    chains = {};
    for uu = 1:numel(base)
        b = base{uu};
        pos = 1;
        while pos <= numel(b)
            seg = min(target, numel(b) - pos + 1);
            chains{end + 1} = b(pos:pos + seg - 1); %#ok<AGROW>
            pos = pos + seg;
        end
    end
end
