function [Y, dates, timeframe] = load_realdata_panel(data_dir, portfolio, scale, timeframe)
%% Load the SP100 price panel and convert it to log returns.
%
% The panel is stored as a spreadsheet whose first column is the trading
% calendar, whose last column is the S&P 100 index level, and whose remaining
% columns are the closing prices of the p constituents. The index level is
% not an asset and is dropped.
%
% Usage:
%   [Y, dates, timeframe] = load_realdata_panel(data_dir, "SP100", 100);
%   [Y, dates, timeframe] = load_realdata_panel(data_dir, "SP100", 100, 1:2499);
%
% Input:
%   @data_dir: Directory containing the raw price spreadsheet.
%   @portfolio: "SP100".
%   @scale: Multiplicative scale applied to log returns.
%   @timeframe: Optional index range after return construction.
%
% Output:
%   @Y: T-by-p log-return matrix.
%   @dates: T-by-1 datetime vector aligned with Y.
%   @timeframe: The index range actually used.

    portfolio_key = upper(string(portfolio));
    if portfolio_key ~= "SP100"
        error('load_realdata_panel:portfolio', ...
            'Only the SP100 panel is available, got "%s".', char(portfolio));
    end

    % Read with the built-in reader rather than the Excel automation server
    % that Windows with Excel would use, so the calendar and the prices are
    % parsed identically on every platform.
    tbl = readtable(fullfile(data_dir, 'SP100.xls'), ...
        'VariableNamingRule', 'preserve', 'UseExcel', false);
    names = string(tbl.Properties.VariableNames);
    if names(end) ~= "S&P100"
        error('load_realdata_panel:indexColumn', ...
            'Expected the last column to be the S&P100 index level, got "%s".', ...
            char(names(end)));
    end
    if ~isdatetime(tbl{:, 1})
        error('load_realdata_panel:calendar', ...
            'Expected the first column to be a date column, got %s.', ...
            class(tbl{:, 1}));
    end
    prices = tbl{:, 2:width(tbl) - 1};

    if nargin < 4 || isempty(timeframe)
        % Defaults are defined on the return panel, not the raw price panel.
        % SP100 has n price rows and therefore n-1 log-return observations.
        timeframe = 1:(size(prices, 1) - 1);
    end

    % Return t is realized between the closing prices of trading days t and
    % t+1, so it carries the later date; the dates are shifted by one
    % observation to align with prices(2:end, :).
    dates = tbl{:, 1}(2:end);
    Y = scale * (log(prices(2:end, :)) - log(prices(1:end - 1, :)));
    Y = Y(timeframe, :);
    dates = dates(timeframe, :);
end
