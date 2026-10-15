function [names, grids] = ic_family_criteria(lossval, gauss, dim, nbreak, p, T)
%% Evaluate the full information-criterion family on a candidate grid.
%
% Every criterion is a scalar combination of the recorded per-candidate
% squared-Frobenius loss, Gaussian negative log-likelihood, model dimension
% and break count, so a stored metric grid can be rescored under any
% criterion without refitting the candidate. The family mirrors the
% criteria reported by the simulation pipeline: AIC, BIC, HBIC, HBICG, the
% fixed-penalty HBIC/HBICG sensitivity variants, the break-only variant
% HBICG(0,2) and the plug-in calibrated HBIC_c1/HBIC_c2.
%
% With C_T = log(T) log(p) / T and alpha_T = log(T) p / T the criteria are
%
%   BIC          = loss + p log(T) K
%   HBIC         = loss + C_T K + alpha_T nbreak
%   HBICG        = gauss + C_T K + alpha_T nbreak
%   AIC          = 2 K + 2 loss
%   HBIC_g*_b*   = loss + g*C_T K + b*alpha_T nbreak
%   HBICG_g*_b*  = gauss + g*C_T K + b*alpha_T nbreak
%   HBICG(0,2)   = gauss + 2 alpha_T nbreak
%   HBIC_c1/c2   = loss + C_T K + c*alpha_T nbreak, c = c_est or 2 c_est
%
% Unevaluated cells (Inf/NaN inputs) are set to Inf so they never win a
% selection.
%
% - Input:
%   @lossval: Squared-Frobenius losses over candidates.
%   @gauss:   Gaussian negative log-likelihood fits.
%   @dim:     Model dimension K per candidate.
%   @nbreak:  Number of estimated breaks per candidate.
%   @p, @T:   Panel dimension and sample size.
%
% - Output:
%   @names: 1-by-N cell of criterion names, matching @grids.
%   @grids: 1-by-N cell of criterion arrays, each the size of @lossval.

    C_T = log(T) * log(p) / T;
    alpha_T = log(T) * p / T;

    names = {'BIC', 'HBIC', 'HBICG', ...
        'AIC', 'HBIC_g0_b8', 'HBIC_g2_b8', 'HBIC_g4_b1', ...
        'HBICG_g0_b2', 'HBICG_g025_b1', 'HBICG_g05_b1', ...
        'HBICG(0,2)', 'HBIC_c1', 'HBIC_c2'};
    grids = cell(size(names));

    grids{1} = dim * p * log(T) + lossval;
    grids{2} = lossval + C_T * dim + alpha_T * nbreak;
    grids{3} = gauss + C_T * dim + alpha_T * nbreak;
    grids{4} = 2 * dim + 2 * lossval;

    fixed_hbic_gamma = [0, 2, 4];
    fixed_hbic_beta = [8, 8, 1];
    for vv = 1:numel(fixed_hbic_gamma)
        grids{4 + vv} = lossval + fixed_hbic_gamma(vv) * C_T * dim + ...
            fixed_hbic_beta(vv) * alpha_T * nbreak;
    end

    fixed_hbicg_gamma = [0, 0.25, 0.5];
    fixed_hbicg_beta = [2, 1, 1];
    for vv = 1:numel(fixed_hbicg_gamma)
        grids{7 + vv} = gauss + fixed_hbicg_gamma(vv) * C_T * dim + ...
            fixed_hbicg_beta(vv) * alpha_T * nbreak;
    end

    grids{11} = gauss + 2 * alpha_T * nbreak;

    finite = isfinite(lossval) & isfinite(gauss) & isfinite(dim) & ...
        isfinite(nbreak);
    c_est = hbic_c_est(lossval, dim, nbreak, alpha_T, C_T, finite);
    if isnan(c_est)
        % Degenerate landscape: fall back to the plain HBIC penalty.
        grids{12} = grids{2};
        grids{13} = grids{2};
    else
        grids{12} = lossval + C_T * dim + c_est * alpha_T * nbreak;
        grids{13} = lossval + C_T * dim + 2 * c_est * alpha_T * nbreak;
    end

    for vv = 1:numel(grids)
        g = grids{vv};
        g(~finite) = inf;
        grids{vv} = g;
    end
end
