function Theta_Up(obj)
    %% Theta update.

    % Assemble the right-hand side with the fixed block-diagonal metric.
    % V acts on every entry, Upsilon only on off-diagonal entries, and D
    % on each temporal neighbour.
    Psi_k = obj.XXt_over_T + obj.Y_k + obj.beta_u * obj.Upsilon_k;
    Psi_k = Psi_k + obj.A_k + obj.beta_v * obj.V_k;

    % Subtract (Z_{t+1} - Z_t) + beta_d * (D_{t+1} - D_t), using the
    % boundary convention Z_1 = Z_{T+1} = D_1 = D_{T+1} = 0.
    G_shift = obj.Z_k(:, :, 2:obj.T) + obj.beta_d * obj.D_k(:, :, 2:obj.T);
    Psi_k(:, :, 1:obj.T-1) = Psi_k(:, :, 1:obj.T-1) - G_shift;
    Psi_k(:, :, 2:obj.T) = Psi_k(:, :, 2:obj.T) + G_shift;

    % A and Adiag are tridiagonal and fixed. Below the crossover the dense
    % inverses win because the product runs in multithreaded BLAS; above it
    % the O(T^2 p^2) product dominates and the banded solve is cheaper.
    Psi_flat = reshape(Psi_k, obj.p * obj.p, obj.T)';
    Psi_diag = reshape(Psi_k(obj.idx), obj.p, obj.T)';
    if isempty(obj.Ainv)
        Theta_kp1 = obj.A \ Psi_flat;
        Theta_diag = obj.Adiag \ Psi_diag;
    else
        Theta_kp1 = obj.Ainv * Psi_flat;
        Theta_diag = obj.Adiaginv * Psi_diag;
    end
    Theta_kp1 = reshape(Theta_kp1', obj.p, obj.p, obj.T);
    Theta_kp1(obj.idx) = Theta_diag';

    % Psi_k is assembled from symmetric blocks and both systems act along
    % the time axis, so Theta_kp1 is already symmetric page by page.
    obj.Theta_k = Theta_kp1;
end
