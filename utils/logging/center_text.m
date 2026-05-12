function out = center_text(str, width, pad_char)
%CENTER_TEXT Center a string within a fixed width using a padding character.
%   out = CENTER_TEXT(str, width, pad_char) returns str centered in a field
%   of length width. pad_char (default: space) fills the remaining space.
    if nargin < 3
        pad_char = ' ';
    end
    str = char(str);
    str_len = numel(str);
    pad_total = max(width - str_len, 0);
    pad_left = floor(pad_total / 2);
    pad_right = pad_total - pad_left;
    out = [repmat(pad_char, 1, pad_left), str, repmat(pad_char, 1, pad_right)];
end
