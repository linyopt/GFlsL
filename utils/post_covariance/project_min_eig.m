function A_pd = project_min_eig(A_in, min_eig)
%% Project a symmetric matrix onto the cone {A : lambda_min(A) >= min_eig}.

    arguments
        A_in double
        min_eig (1, 1) double {mustBePositive} = 1e-4
    end

    % Symmetrize first so the eigendecomposition represents the intended
    % matrix even when the input carries small numerical asymmetry.
    A = (A_in + A_in') / 2;
    [V, D] = eig(A);
    d = diag(D);

    % Clamp the spectrum from below and reconstruct the matrix.
    d = max(d, min_eig);
    A_pd = V * diag(d) * V';
    A_pd = (A_pd + A_pd') / 2;
end
