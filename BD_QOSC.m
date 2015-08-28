function [ Z, funVal, iteration ] = BD_QOSC( X,k,lambda_1, lambda_2, gamma_1,p, maxIterations, diagconstraint,pos)
%% Problem
%
% min L(Z) = 1/2 ||X - XZ||^2_F - lambda_1 * ||ZTZ||_1 + lambda_2 * ||ZR||_2/1
% 
% where ||B||_2/1 = ||b_1||_2 + ||b_2||_2 + ... + ||b_n||_2
%
%% Solution 
% We solve this problem via the ADMM (Alternating direction method of multipliers) variant
% of Augmented Lagrangian method as follows:
% 
% Let U = ZR then we have
% 
% T(Z,S,U) = 1/2||X - XS||^2_F + lambda_1 * ||ZTZ||_1 + lambda_2 * ||U||_2/1
%<F,U - ZR>+ gamma_2/2 * ||U - ZR||^2_F

if (~exist('diagconstraint','var'))
    diagconstraint = 0;
end

if (~exist('pos','var'))
    pos = 0;
end

funVal = zeros(maxIterations,1);

[~, xn, ~] = size(X);
F = zeros(xn, xn-1); 
Z = zeros(xn, xn);
R = (triu(ones(xn,xn-1),1) - triu(ones(xn, xn-1))) + (triu(ones(xn, xn-1),-1)-triu(ones(xn, xn-1)));
R = sparse(R);
U = Z*R;
for iteration=1:maxIterations

    %% Step 1
    %% Step 1
    A = X'*X;
    B = 2*lambda_1*ones(size(Z))+gamma_1*(R*R');%+(1*10^-5)*ones(xn,xn)
    C = -(X'*X + gamma_1*U*R' + F*R');
    Z = lyap(A, B, C);
    
    if(pos)
    Z(Z<0)=0;
    end
    
    if (diagconstraint)
        Z(logical(eye(size(Z)))) = 0;
    end
    
    if iteration>maxIterations/2
    Z = projKappa(Z,Z,k); 
    end
   %% Step 2
    V = Z*R - (1/gamma_1)*F;

    U = mysolve_l1l2(V, lambda_2/gamma_1);

    %% Step 3

    F = F + gamma_1 * (U - Z*R);

    %% Step 4
    gamma_1 = p * gamma_1;

    %% Calculate function values
    funVal(iteration) = .5 * norm(X - X*Z,'fro')^2 + lambda_1*norm(Z'*Z,1) +lambda_2*l2l1norm(Z*R);

    if iteration > 1
        if funVal(iteration) < 1*10^-3
            break
        end
    end

    if iteration > 100
        if funVal(iteration) < 1*10^-3 || funVal(iteration-1) == funVal(iteration) ...
                || funVal(iteration-1) - funVal(iteration) < 1*10^-3
            break
        end
    end


end

end