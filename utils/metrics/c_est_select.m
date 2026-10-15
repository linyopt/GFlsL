function [res_c1_hbic, res_c2_hbic] = ...
    c_est_select(hbic_vals, lossvals, est_breaks, ...
                 hds, f1s, errors, accs, alpha_T, C_T)
%% Plug-in penalty calibration for HBIC.
%
% Recovers the fitted loss landscape (lossval, edge_total, nbreak) from the
% HBIC values, calibrates c_est through @hbic_c_est, and returns selector
% results for the c_est and 2*c_est rescalings. The calibration itself is not
% repeated here: hbic_c_est is the single implementation, shared with the
% criteria family of the real-data path.
%
% Rescaled selectors:
%   HBIC_c = HBIC + (c - 1) * alpha_T * nbreak
%
% - Input:
%   @hbic_vals:   HBIC values over candidates.
%   @lossvals:    Loss values over candidates.
%   @est_breaks:  Cell array of estimated break vectors.
%   @hds, @f1s, @errors, @accs: Truth-comparison metrics per candidate.
%   @alpha_T:     Break penalty scale log(T)*p/T.
%   @C_T:         Edge penalty scale log(T)*log(p)/T.
%
% - Output:
%   @res_c1_hbic:  1x5 row [HD, F1, error, acc, nbreak] under c_est.
%   @res_c2_hbic:  1x5 row under 2*c_est.

% Flatten all inputs to 1-D vectors.
hbic_f   = hbic_vals(:);
loss_f   = lossvals(:);
hd_f     = hds(:);
f1_f     = f1s(:);
err_f    = errors(:);
acc_f    = accs(:);
eb_cell  = est_breaks(:);

% Restrict to evaluated candidates: metric arrays are Inf-initialized,
% so unevaluated grid points carry Inf/NaN and must not win any argmin.
valid = isfinite(hbic_f) & isfinite(loss_f);
hbic_f   = hbic_f(valid);
loss_f   = loss_f(valid);
hd_f     = hd_f(valid);
f1_f     = f1_f(valid);
err_f    = err_f(valid);
acc_f    = acc_f(valid);
eb_cell  = eb_cell(valid);

n = numel(hbic_f);
nbs = zeros(n, 1);
for i = 1:n
    nbs(i) = length(eb_cell{i});
end

% Recover edge_total from HBIC = lossval + C_T * edge_total + alpha_T * nb
edge_total = (hbic_f - loss_f - alpha_T * nbs) / C_T;

% The calibration is shared with the criteria family of the real-data path.
c_est = hbic_c_est(loss_f, edge_total, nbs, alpha_T, C_T, ...
    true(numel(loss_f), 1));

if isnan(c_est)
    % c_est is undefined when every candidate carries the same number of
    % breaks. Fall back to the original HBIC selection.
    [~, i] = min(hbic_f);
    res_c1_hbic  = [hd_f(i), f1_f(i), err_f(i), acc_f(i), nbs(i)];
    res_c2_hbic  = res_c1_hbic;
    return;
end

% Rescaled penalty values.
% HBIC_c = lossval + C_T*edge_total + c*alpha_T*nb
%        = HBIC + (c - 1)*alpha_T*nb
hbic_c1  = hbic_f  + (c_est       - 1) * alpha_T * nbs;
hbic_c2  = hbic_f  + (2 * c_est    - 1) * alpha_T * nbs;

% Select and package results.
[~, i] = min(hbic_c1);
res_c1_hbic  = [hd_f(i), f1_f(i), err_f(i), acc_f(i), nbs(i)];
[~, i] = min(hbic_c2);
res_c2_hbic  = [hd_f(i), f1_f(i), err_f(i), acc_f(i), nbs(i)];
end
