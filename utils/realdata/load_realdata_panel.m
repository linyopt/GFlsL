function [Y, dates, timeframe] = load_realdata_panel(data_dir, portfolio, scale, timeframe)
%% Load one real-data price panel and convert it to log returns.
%
% Usage:
%   [Y, dates, timeframe] = load_realdata_panel(data_dir, "SP500", 100);
%   [Y, dates, timeframe] = load_realdata_panel(data_dir, "SP500", 100, 1:1902);
%
% Input:
%   @data_dir: Directory containing the raw Excel files.
%   @portfolio: "SP500" or "MSCI".
%   @scale: Multiplicative scale applied to log returns.
%   @timeframe: Optional index range after return construction.
%
% Output:
%   @Y: T-by-p log-return matrix.
%   @dates: T-by-1 date vector aligned with Y.
%   @timeframe: The index range actually used.

    if nargin < 4 || isempty(timeframe)
        switch string(portfolio)
            case "MSCI"
                timeframe = 2100:3900;
            case "SP500"
                timeframe = 1:1902;
            otherwise
                error('Unsupported portfolio "%s".', char(portfolio));
        end
    end

    switch string(portfolio)
        case "MSCI"
            filename = 'MSCI.xls';
        case "SP500"
            filename = 'SP500_large.xlsx';
        otherwise
            error('Unsupported portfolio "%s".', char(portfolio));
    end

    table_in = readtable(fullfile(data_dir, filename), ...
        'VariableNamingRule', 'preserve');
    prices = table_in{:, 2:end};
    dates = table_in{2:end, 1};
    Y = scale * (log(prices(2:end, :)) - log(prices(1:end - 1, :)));

    Y = Y(timeframe, :);
    dates = dates(timeframe, :);
end
