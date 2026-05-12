function GetDelta_(obj)
    %% Compute the auxiliary variable delta_.

    % Declare delta_.
    delta_ = zeros(obj.p, obj.p, obj.T);
    % For the first one.
    delta_(:, :, 1) = obj.Z_k(:, :, 2) + (obj.Theta_k(:, :, 1) ...
                            - obj.XXt(:, :, 1)) / obj.T - obj.Y_k(:, :, 1);
    % For the last one.
    delta_(:, :, obj.T) = -obj.Z_k(:, :, obj.T) + (obj.Theta_k(:, :, obj.T) ...
            - obj.XXt(:, :, obj.T)) / obj.T - obj.Y_k(:, :, obj.T);
    % For the middle ones.
    delta_(:, :, 2:obj.T-1) = obj.Z_k(:, :, 3:obj.T) - obj.Z_k(:, :, 2:obj.T-1) ...
            + (obj.Theta_k(:, :, 2:obj.T-1) - obj.XXt(:, :, 2:obj.T-1)) / obj.T ...
            - obj.Y_k(:, :, 2:obj.T-1);

    % Symmetrization.
    obj.delta_ = (delta_ + permute(delta_, [2, 1, 3])) / 2;
end