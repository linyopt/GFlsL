function result = hbic_penalty_select(hbic_vals, lossvals, est_breaks, ...
    hds, f1s, errors, accs, alpha_T, C_T, gamma, beta_ic)
%% Fixed-penalty HBIC variant selector.
%
% Selects by lossval + gamma*C_T*edge_total + beta_ic*alpha_T*nbreak,
% where edge_total is recovered from the stored HBIC values.
%
% - Input:
%   @hbic_vals:   HBIC values over candidates.
%   @lossvals:    Loss values over candidates.
%   @est_breaks:  Cell array of estimated break vectors.
%   @hds, @f1s, @errors, @accs: Truth-comparison metrics per candidate.
%   @alpha_T:     Break penalty scale log(T)*p/T.
%   @C_T:         Edge penalty scale log(T)*log(p)/T.
%   @gamma:       Multiplier on the edge penalty.
%   @beta_ic:     Multiplier on the break penalty.
%
% - Output:
%   @result:  1x5 row [HD, F1, error, acc, nbreak] of the winner.

hbic_f = hbic_vals(:);
loss_f = lossvals(:);
hd_f = hds(:);
f1_f = f1s(:);
err_f = errors(:);
acc_f = accs(:);
eb_cell = est_breaks(:);

valid = isfinite(hbic_f) & isfinite(loss_f) & isfinite(f1_f);
hbic_f = hbic_f(valid);
loss_f = loss_f(valid);
hd_f = hd_f(valid);
f1_f = f1_f(valid);
err_f = err_f(valid);
acc_f = acc_f(valid);
eb_cell = eb_cell(valid);
if isempty(hbic_f)
    error('hbic_penalty_select:noValidCandidate', ...
        'No finite candidate is available for HBIC selection.');
end

nbs = cellfun(@length, eb_cell);
edge_total = (hbic_f - loss_f - alpha_T * nbs) / C_T;
criterion = loss_f + gamma * C_T * edge_total + ...
    beta_ic * alpha_T * nbs;
[~, ii] = min(criterion);
result = [hd_f(ii), f1_f(ii), err_f(ii), acc_f(ii), nbs(ii)];
end
