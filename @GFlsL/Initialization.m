function Initialization(obj)
    %% Initialization of the algorithm.
    % This initialization method initializes the algorithm.

    % The initialization of the algorithm.
    obj.Theta_k = obj.XXt;
    obj.V_k = obj.Theta_k;
    obj.Upsilon_k = obj.Theta_k;
    % Let the diagonal elements be 0.
    obj.Upsilon_k(obj.idx) = 0;
    obj.D_k = zeros(obj.p, obj.p, obj.T);
    obj.D_k(:, :, 2:obj.T) = diff(obj.XXt, 1, 3);
    obj.A_k = ones(obj.p, obj.p, obj.T);
    obj.Y_k = zeros(obj.p, obj.p, obj.T);
    obj.Z_k = zeros(obj.p, obj.p, obj.T);
end