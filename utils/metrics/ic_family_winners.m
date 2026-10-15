function winners = ic_family_winners(grids, ndim)
%% Winner subscripts of every criterion in the information-criterion family.
%
% Unevaluated grid cells carry Inf/NaN and never win a criterion.
%
% - Input:
%   @grids: 1-by-N cell of criterion arrays over the same candidate grid.
%   @ndim:  Number of subscripts to return for each criterion.
%
% - Output:
%   @winners: N-by-ndim matrix with one row of subscripts per criterion.

    winners = zeros(numel(grids), ndim);
    subs = cell(1, ndim);
    for qq = 1:numel(grids)
        g = grids{qq};
        g(~isfinite(g)) = inf;
        [subs{:}] = argmin_finite(g);
        winners(qq, :) = [subs{:}];
    end
end
