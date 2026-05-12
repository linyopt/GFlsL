function Sigma = simulate_sparse_factor(p,m,sparsity,threshold)

cond = true; parameter = zeros(p,m);

while cond
    
    for kk=1:p
        for jj=1:m
            cond_dist=true;
            while cond_dist
                param = -2+(2-(-2))*rand(1);
                cond_dist=(param>-0.5 & param<0.5);
            end
            parameter(kk,jj)=param;
        end
    end
    Lambda = (rand(p,m)>threshold).*parameter;
    L = vec(Lambda);
    count = 0;
    for ii = 1:p*m
        if (L(ii)==0)
            count = count+1;
        else
            count = count+0;
        end
    end
    cond = (count>sparsity)||(count<sparsity);
end
Psi = diag(diag((0.5+0.5*rand(p)))); Sigma = Lambda*Lambda'+Psi;
