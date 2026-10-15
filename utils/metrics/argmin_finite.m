function varargout = argmin_finite(x)
%% Argmin over finite entries only.
%
% Metric arrays are Inf-initialized, so unevaluated grid points must not
% win the selection.
%
% - Usage:
%   jj = argmin_finite(x);            % linear index
%   [i1, i2, i3, i4] = argmin_finite(x);   % subscripts
%
% - Input:
%   @x:  Numeric array, possibly holding Inf/NaN at unevaluated cells.
%
% - Output:
%   @jj or [i1, ...]:  The linear index (single output) or subscripts
%                      (multiple outputs) of the first finite minimum.

mask = isfinite(x);
if ~any(mask(:))
    error('argmin_finite:noValidCandidate', ...
        'No finite candidate is available for criterion selection.');
end

xvalid = x;
xvalid(~mask) = Inf;
[~, lin] = min(xvalid(:));
nout = max(nargout, 1);
if nout <= 1
    varargout = {lin};
    return
end

% Slicing a metric array along its last dimension drops trailing
% singleton dimensions, so size(x) can be shorter than the number of
% subscripts the caller asks for. Pad it to nout so ind2sub fills every
% requested output.
sz = size(x);
if numel(sz) < nout
    sz = [sz, ones(1, nout - numel(sz))];
end
subs = cell(1, nout);
[subs{:}] = ind2sub(sz, lin);
varargout = subs;
end
