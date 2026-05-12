function write_settings_block(fid, s)
    % Pretty-print key tuning/scaling settings for traceability.
    % Writes a formatted box with all hyperparameters to the given file handle.
    %
    % Parameters:
    %   fid - File handle opened for writing
    %   s   - Settings struct. Common fields include portfolio, scale,
    %         timeframe, lambda grids, detector tuning parameters, and
    %         optimization controls. Optional fields are printed only when
    %         they are present.
    rows = cell(0, 2);
    rows(end + 1, :) = {'Portfolio', sprintf('%s', char(s.portfolio))};
    rows(end + 1, :) = {'Scale', sprintf('%.4g', s.scale)};
    rows(end + 1, :) = {'Timeframe', sprintf('[%d ... %d] (n=%d)', ...
        min(s.timeframe), max(s.timeframe), numel(s.timeframe))};
    rows = append_row_if_present(rows, s, 'center_data', @(x) char(string(logical(x))));
    rows = append_row_if_present(rows, s, 'diagnostic_window', @(x) sprintf('%.4g', x));

    rows = append_row_if_present(rows, s, 'lamb1s', @(x) summarize_grid(x));
    rows = append_row_if_present(rows, s, 'lamb2s', @(x) summarize_grid(x));
    rows = append_row_if_present(rows, s, 'lamb2s2', @(x) summarize_grid(x));
    rows = append_row_if_present(rows, s, 'tau_BSOP', @(x) sprintf('%.4g', x));
    rows = append_row_if_present(rows, s, 'delta_WBSIP', @(x) sprintf('%.4g', x));
    rows = append_row_if_present(rows, s, 'M_WBSIP', @(x) sprintf('%.4g', x));
    rows = append_row_if_present(rows, s, 'tau_WBSIP', @(x) sprintf('%.4g', x));
    rows = append_row_if_present(rows, s, 'gamma_DCDP', @(x) summarize_grid(x));
    rows = append_row_if_present(rows, s, 'zeta_DCDP', @(x) sprintf('%.4g', x));
    rows = append_row_if_present(rows, s, 'Q_DCDP', @(x) sprintf('%.4g', x));
    rows = append_row_if_present(rows, s, 'mu1', @(x) sprintf('%.4g', x));
    rows = append_row_if_present(rows, s, 'mu2', @(x) sprintf('%.4g', x));
    rows = append_row_if_present(rows, s, 'kappa_', @(x) sprintf('%.4g', x));
    rows = append_row_if_present(rows, s, 'beta_', @(x) sprintf('%.4g', x));
    rows = append_row_if_present(rows, s, 'tol', @(x) sprintf('%.4g', x));

    key_width = max(cellfun(@(x) numel(x), rows(:, 1)));
    lines = cell(size(rows, 1), 1);
    for ii = 1:size(rows, 1)
        lines{ii} = sprintf('%-*s : %s', key_width, rows{ii, 1}, rows{ii, 2});
    end

    title_text = 'Run Settings';
    inner_width = max([cellfun(@numel, lines); numel(title_text)]);
    pad_line = repmat('─', 1, inner_width + 2);

    fprintf(fid, '┌%s┐\n', pad_line);
    fprintf(fid, '│ %s │\n', center_text(title_text, inner_width, ' '));
    fprintf(fid, '├%s┤\n', pad_line);
    for ii = 1:numel(lines)
        fprintf(fid, '│ %-*s │\n', inner_width, lines{ii});
    end
    fprintf(fid, '└%s┘\n', pad_line);
end

function rows = append_row_if_present(rows, s, field_name, formatter)
    % Add one formatted settings row only when the field exists.
    if ~isfield(s, field_name)
        return;
    end
    rows(end + 1, :) = {field_name, formatter(s.(field_name))};
end

function txt = summarize_grid(vec)
    % Summarize a numeric grid by its endpoints and length.
    if isempty(vec)
        txt = '[]';
        return;
    end
    txt = sprintf('[%.3g … %.3g] n=%d', min(vec(:)), max(vec(:)), numel(vec));
end
