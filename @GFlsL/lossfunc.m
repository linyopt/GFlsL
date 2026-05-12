function val = lossfunc(obj, Y)
%% Compute the mean of loss function values of given Y's, which are out of training set.
% The input Y can be a series of data, then the output will be the mean of
% the loss function values.
%
% - Usage:
%   lossfunc = Est.lossfunc(Y);
%
% - Input:
%   @Y:         The data.
%
% - Output:
%   @val:       The loss function value.

% The number of datasets in Y.
num_Y = size(Y, 3);

% The loss function values.
vals(num_Y) = 0;

% For each dataset in Y.
for tt = 1 : num_Y
    % Compute the XXt.
    Y_reshaped = reshape(Y(:, :, tt), [1, obj.T, obj.p]);
    XXt = bsxfun(@times, permute(Y_reshaped, [3, 1, 2]), ...
                    permute(Y_reshaped, [1, 3, 2]));

    % Compute loss function values.
    vals(tt) = norm(obj.Theta_k - XXt, 'fro') ^ 2 / 2;
end
% Compute mean and return.
val = mean(vals);
end