function Theta_Up(obj)
    %% Theta update.

    % Compute Psi_k.
    % Compute Z_diff = Z_{t+1} - Z_t, where the first one and the last one
    % are different.
    Z_diff = zeros(obj.p, obj.p, obj.T);
    % The first one.
    Z_diff(:, :, 1) = obj.Z_k(:, :, 2);
    % The middle ones.
    Z_diff(:, :, 2:obj.T-1) = diff(obj.Z_k(:, :, 2:obj.T), 1, 3);
    % The last one.
    Z_diff(:, :, obj.T) = - obj.Z_k(:, :, obj.T);

    % Compute D_diff = D_{t+1} - D_t.
    D_diff = zeros(obj.p, obj.p, obj.T);
    % The first one.
    D_diff(:, :, 1) = obj.D_k(:, :, 2);
    % The middle ones.
    D_diff(:, :, 2:obj.T-1) = diff(obj.D_k(:, :, 2:obj.T), 1, 3);
    % The last one.
    D_diff(:, :, obj.T) = - obj.D_k(:, :, obj.T);

    % Compute Psi.
    Psi_k = obj.XXt / (obj.T) + obj.A_k + obj.Y_k - Z_diff ...
        + obj.beta_ * (obj.V_k + obj.Upsilon_k - D_diff);

    % Use \ to solve the linear system.
    Theta_kp1 = obj.A \ reshape(Psi_k, obj.p * obj.p, obj.T)';
    Theta_kp1 = reshape(Theta_kp1', obj.p, obj.p, obj.T);
    Theta_kp1(obj.idx) = (obj.Adiag \ reshape(Psi_k(obj.idx), obj.p, obj.T)')';

    % Symmetrization.
    obj.Theta_k = (Theta_kp1 + permute(Theta_kp1, [2, 1, 3])) / 2;
end
