function display_results_table(fid, rows)
%% Display performance metrics as a formatted table with Unicode box-drawing.
%
% This function creates a unified table showing all method comparison results
% in a single view for easy comparison. The table is organized into sections
% with headers and data rows.
%
% - Usage:
%   display_results_table(fid, rows)
%
% - Input:
%   @fid:       File identifier (1 for stdout, or use fopen to write to file)
%   @rows:      Cell array where each row is either:
%               - Section row: {section_name, NaN, NaN, NaN, NaN, NaN}
%               - Data row: {method_name, HD, F1, error, accuracy, nbreaks}
%
% Table Structure:
%   ┌─────────────────────────────┐  <- Top border
%   │      Section Title          │  <- Section header (centered)
%   ├──────┬──────┬──────┬────────┤  <- Column separator
%   │Method│  HD  │  F1  │  Error │  <- Column headers
%   ├──────┼──────┼──────┼────────┤  <- Header separator
%   │HFE   │0.0000│1.0000│ 0.0872 │  <- Data rows
%   └─────────────────────────────┘  <- Bottom border

    headers = {'method', 'HD', 'F1', 'error', 'acc', 'nbreak'};
    num_cols = numel(headers);
    fmt_num = @(x) ternary(isnan(x), '', sprintf('%.2f', x));

    % Compute column widths based on header and data.
    col_widths = zeros(1, num_cols);
    is_section = false(size(rows, 1), 1);
    for rr = 1:size(rows, 1)
        is_section(rr) = is_section_row(rows(rr, :));
    end
    data_rows = rows(~is_section, :);
    if isempty(data_rows)
        data_rows = {headers{1}, nan, nan, nan, nan, nan};
    end
    col_widths(1) = max(numel(headers{1}), max(cellfun(@(x) numel(char(x)), data_rows(:, 1))));
    for cc = 2:num_cols
        col_widths(cc) = numel(headers{cc});
        for rr = 1:size(data_rows, 1)
            col_widths(cc) = max(col_widths(cc), numel(fmt_num(data_rows{rr, cc})));
        end
    end

    % Build formatted lines.
    header_cells = cell(1, num_cols);
    header_cells{1} = sprintf('%-*s', col_widths(1), headers{1});
    for cc = 2:num_cols
        header_cells{cc} = sprintf('%*s', col_widths(cc), headers{cc});
    end
    header_line = strjoin(header_cells, ' │ ');

    formatted_rows = cell(size(rows, 1), 1);
    for rr = 1:size(rows, 1)
        cells = cell(1, num_cols);
        cells{1} = sprintf('%-*s', col_widths(1), rows{rr, 1});
        for cc = 2:num_cols
            cells{cc} = sprintf('%*s', col_widths(cc), fmt_num(rows{rr, cc}));
        end
        formatted_rows{rr} = strjoin(cells, ' │ ');
    end

    inner_width = sum(col_widths) + 3 * (num_cols - 1);
    pad_line = repmat('─', 1, inner_width);

    % Create separators with proper junction characters.
    header_sep_parts = cell(1, num_cols);
    for cc = 1:num_cols
        header_sep_parts{cc} = repmat('─', 1, col_widths(cc));
    end
    header_sep_top = strjoin(header_sep_parts, '─┬─');
    header_sep_middle = strjoin(header_sep_parts, '─┼─');
    bottom_sep = strjoin(header_sep_parts, '─┴─');

    fprintf(fid, '┌%s┐\n', pad_line);
    first_section = true;
    for rr = 1:size(rows, 1)
        if is_section_row(rows(rr, :))
            if ~first_section
                fprintf(fid, '├%s┤\n', bottom_sep);
            end
            title_line = center_text(rows{rr, 1}, inner_width, ' ');
            fprintf(fid, '│%s│\n', title_line);
            fprintf(fid, '├%s┤\n', header_sep_top);
            fprintf(fid, '│%s│\n', header_line);
            fprintf(fid, '├%s┤\n', header_sep_middle);
            first_section = false;
        else
            fprintf(fid, '│%s│\n', formatted_rows{rr});
        end
    end
    fprintf(fid, '└%s┘\n', bottom_sep);
end

function out = ternary(cond, a, b)
%% Ternary operator: return a if cond is true, else return b.
    if cond
        out = a;
    else
        out = b;
    end
end

function tf = is_section_row(row)
%% Check if a row is a section header (all metrics NaN).
    tf = ischar(row{1}) && all(cellfun(@(x) isnan(x), row(2:end)));
end
