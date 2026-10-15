function C = simulate_sparse_varcov(p,sparsity,threshold,a1,a2,b1,b2,nzlb)

% Input:
%       - p: dimension
%       - sparsity: sparsity degree, precentage of p*(p-1)/2 set as zero
%       coefficients
%       - threshold: controls for how many zero coefficients are generated
%       in the true vector of coefficients; the higher, the more zero
%       coefficients are generated
%       - a1 and a2: outer bounds of the true non-zero off-diag
%       coefficients; each is drawn from [a1, -nzlb] or [nzlb, a2], so its
%       absolute value is bounded below by nzlb
%       - b1 and b2: lower and upper bounds of the interval for the uniform
%       distribution when generating the true non-zero diagonal coefficients
%       - nzlb: non-zero lower bound on the absolute value of the true
%       non-zero off-diag coefficients (optional, default 0.2)

if nargin < 8 || isempty(nzlb)
    nzlb = 0.2;
end

dim = p*(p-1)/2;
cond = true;
while cond
    temp = round(rand(dim,1)); temp(temp==0)=-1;
    % Draw from [a1, -nzlb] (negative side) or [nzlb, a2] (positive side),
    % selected by temp, so |coefficient| >= nzlb.
    val_neg = a1 + (-nzlb - a1) * rand(dim,1);
    val_pos = nzlb + (a2 - nzlb) * rand(dim,1);
    val = (temp == 1) .* val_pos + (temp == -1) .* val_neg;
    C = vech_off((rand(dim,1)>threshold).*val,p);
    beta_true = vech_on(C,p);
    count = 0;
    for ii = 1:dim
        if (beta_true(ii)==0)
            count = count+1;
        else
            count = count+0;
        end
    end
    % Check that the two following conditions are satisifed: number of zero
    % coefficients (remains unchanged for all batches);
    % positive-definiteness of the true correlation (or
    % variance-covariance) matrix of the Gaussian copula
    cond = (count>sparsity)||(count<sparsity);
end
C = C + diag((b1+(b2-b1)*rand(p,1)));
if min(eig(C))<0.01
    zeta = 0;
    while (min(eig(C))<0.01)
        C = C + (zeta + abs(min(eig(C))))*eye(p);
        zeta = zeta + 0.005;
    end
end