function c_est = hbic_c_est(lossval, edge, nbreak, alpha_T, C_T, finite)
%% Plug-in calibration of the HBIC break penalty.
%
% The marginal fit improvement of an additional break, net of the
% within-segment (edge) penalty and measured in units of alpha_T, is taken
% between the candidate with the fewest breaks and the candidate with the
% smallest loss:
%
%   c_est = (loss_min - loss_fit - C_T (edge_fit - edge_min))
%           / (alpha_T (nbreak_fit - nbreak_min))
%
% When the fewest-break and best-fit candidates share a break count, c_est
% is undefined and NaN is returned.
%
% - Input:
%   @lossval: Squared-Frobenius losses over candidates.
%   @edge:    Edge penalty per candidate, equal to the model dimension when
%             it is available directly.
%   @nbreak:  Number of estimated breaks per candidate.
%   @alpha_T: Break-count penalty scale log(T)*p/T.
%   @C_T:     Within-segment penalty scale log(T)*log(p)/T.
%   @finite:  Optional logical mask of evaluated candidates. Default
%             isfinite(lossval) & isfinite(edge) & isfinite(nbreak).
%
% - Output:
%   @c_est: Scalar plug-in estimate, or NaN when it is undefined.

    if nargin < 6 || isempty(finite)
        finite = isfinite(lossval) & isfinite(edge) & isfinite(nbreak);
    end
    if C_T <= 0
        error('hbic_c_est:scale', 'C_T must be positive.');
    end

    c_est = NaN;
    idx = find(finite);
    if isempty(idx)
        return
    end
    loss_f = lossval(idx);
    edge_f = edge(idx);
    nb_f = nbreak(idx);
    [~, imn] = min(nb_f);
    [~, imx] = min(loss_f);
    if nb_f(imx) <= nb_f(imn)
        return
    end
    c_est = (loss_f(imn) - loss_f(imx) - C_T * (edge_f(imx) - edge_f(imn))) / ...
        (alpha_T * (nb_f(imx) - nb_f(imn)));
end
