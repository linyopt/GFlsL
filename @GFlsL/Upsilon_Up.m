function Upsilon_Up(obj)
    %% Update Upsilon via l1 proximal.

    % Compute the l1 proximal.
    obj.Upsilon_k = prox_l1(obj.Theta_k - obj.Y_k / obj.beta_, ...
                    obj.lamb1 ./ obj.beta_);
    % Let the diagonal elements be 0.
    obj.Upsilon_k(obj.idx) = 0;
end

function x = prox_l1(v, lambda)
% PROX_L1    The proximal operator of the l1 norm.
%
%   prox_l1(v,lambda) is the proximal operator of the l1 norm
%   with parameter lambda.

    x = max(0, v - lambda) - max(0, -v - lambda);
end