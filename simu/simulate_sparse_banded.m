function C = simulate_sparse_banded(p,a1,a2,ar)

% Input:
%       - p: dimension
%       - a1 and a2: lower and upper bounds of the interval for the uniform
%       distribution when generating the true non-zero off-diag coefficients
%       - ar: coefficient for banded pattern

D = diag(a1+(a2-a1)*rand(p,1));
Omega = zeros(p,p);
for k = 1:p
    for l = 1:p
        Omega(k,l) = ar^(abs(l-k));
    end
end
Sigma = (D.^(1/2))*inv(Omega)*(D.^(1/2)); %Sigma(abs(Sigma))=0;
C = Sigma; C(abs(C)<0.05)=0;
C_temp = binornd(1,0.5,p,p); C_temp(C_temp==0)=-1;
C_temp = triu(C_temp,1); 
C = (C_temp + C_temp' + eye(p)).*C;

if min(eig(C))<0.01
   zeta = 0;
   while (min(eig(C))<0.01)
          C = C + (zeta + abs(min(eig(C))))*eye(p);
          zeta = zeta + 0.005;
    end
end

