function D_Up(obj)
    %% Update D_k.

    % Compute the input of the proximal operator.
    Xi = diff(obj.Theta_k, 1, 3) - obj.Z_k(:, :, 2:obj.T) * ( 1 /  obj.beta_);

    % Compute D_kp1 by computing the proximal operator of the Frobenius
    % norm of Xi with the regularization parameter being lamb2 / beta.
    D_kp1 = prox_FrobeniusNorm(Xi, obj.lamb2 ./ obj.beta_);

    % Symmetrization.
    obj.D_k(:, :, 2:obj.T) = (D_kp1 + permute(D_kp1, [2, 1, 3])) / 2;
end

function X_prox = prox_FrobeniusNorm(X, lambda)
    % Computes the proximal operator of the Frobenius norm.
    % X is the input array of size (d, d, T-1).
    % lambda is the regularization parameter.
    T = size(X, 3);
    X_prox = reshape(max(1 - lambda ./ vecnorm(reshape(X, [], T)), 0), [1, 1, T]) .* X;
end