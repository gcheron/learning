function [kernel,gamma] = rbf_kernel(fset1,fset2,gamma)

c_norm = zeros(size(fset1,1),size(fset2,1)) ;

% compute  ||u-v||^2
parfor i=1:size(fset1,1)
    u=fset1(i,:);
    mi=(bsxfun(@minus,fset2,u)).^2;
    c_norm(i,:) = sum(mi,2) ;
end
if nargin < 3
gamma=1/mean(c_norm(:));
end

kernel = exp(-gamma*c_norm);

assert(sum(isnan(kernel(:)))==0 && sum(isinf(kernel(:)))==0)
