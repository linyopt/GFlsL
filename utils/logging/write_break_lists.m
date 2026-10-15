function write_break_lists(fid, title_text, labels, break_lists, dates)
%% Write break indices and their mapped dates in one log block.
%
% Usage:
%   write_break_lists(fid, 'Break locations', labels, break_lists, dates);
%
% Input:
%   @fid: File handle opened for writing.
%   @title_text: Section title.
%   @labels: Cell array or string array of method labels.
%   @break_lists: Cell array whose entries are break-index vectors.
%   @dates: Date vector aligned with the series.

    labels = cellstr(string(labels(:)));
    label_width = max(cellfun(@numel, labels));

    title_width = numel(title_text);
    pad_line = repmat('─', 1, title_width + 2);
    fprintf(fid, '\n┌%s┐\n', pad_line);
    fprintf(fid, '│ %s │\n', title_text);
    fprintf(fid, '└%s┘\n', pad_line);

    fprintf(fid, 'Break locations (indices):\n');
    for ii = 1:numel(labels)
        write_one_break_line(fid, labels{ii}, break_lists{ii}, label_width);
    end

    % Only map breaks to calendar dates when the series carries a real
    % datetime axis; numeric indices would merely repeat the indices block.
    if isdatetime(dates)
        fprintf(fid, '\nBreak locations (dates):\n');
        for ii = 1:numel(labels)
            write_one_break_line(fid, labels{ii}, ...
                map_breaks_to_dates(break_lists{ii}, dates), label_width);
        end
    end
end

function values = map_breaks_to_dates(breaks, dates)
    if isempty(breaks)
        values = [];
    else
        values = dates(breaks);
    end
end

function write_one_break_line(fid, label, values, label_width)
    if isempty(values)
        fprintf(fid, '  %-*s : []\n', label_width, label);
        return
    end

    if isdatetime(values)
        values_text = join(string(values), ', ');
    else
        values_text = join(string(values(:)'), ', ');
    end
    fprintf(fid, '  %-*s : %s\n', label_width, label, values_text);
end
