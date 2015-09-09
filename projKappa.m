function Z = projKappa(Z0,S,k) 
    Z = zeros(size(Z0));
    temp = Z0 + S;
    W = 0.5*(abs(temp) + abs(temp'));
    NcutDiscrete = ncutW(W,k);

for i = 1:k
     index = find(NcutDiscrete(:,i));
     Z(index,index) = Z0(index,index);
end
end