function key = lambda_key(val)
% Filesystem-safe cache key of a tuning value.
%
% The candidate caches of @get_two_stage_est and @runcode_realdata name their
% files after the tuning values a candidate was fitted at, so the key has to
% be stable across sessions and machines: it is the value printed with 15
% significant digits, with every character a file name cannot carry replaced
% by an underscore.
%
% Usage:
%   key = lambda_key(0.0625);
%
% - Input:
%   @val: Tuning value. Numeric values are formatted; anything else is used
%       as it is.
%
% - Output:
%   @key: String cache key.

    if isnumeric(val)
        key = string(sprintf('%.15g', val));
    else
        key = string(val);
    end
    key = regexprep(key, '[^0-9A-Za-z_.-]', '_');
    key = strrep(key, '.', 'p');
    key = strrep(key, '-', 'm');
end
