function [kernel,gamma] = conc_chi2_exp_kernel(fset1,fset2,gamma)

assert(size(fset1,1)==size(fset2,1));
nb_set1 = size(fset1,2);
nb_set2 = size(fset2,2) ;

kernel = zeros(nb_set1,nb_set2) ;

% compute Dc
parfor i=1:nb_set1
    Hi=fset1(:,i);
    pl=bsxfun(@plus,fset2,Hi);
    mi=(bsxfun(@minus,fset2,Hi)).^2;
    quot=mi./pl;
    quot(isnan(quot))=0;
    kernel(i,:)=0.5*sum(quot);
end

if nargin < 3
    mn=mean(kernel(:));
    gamma=1/mn;
    fprintf('Mean kernel: %f\n',mn);
    fprintf('gamma= %f\n',gamma);    
end

kernel = exp(-gamma*kernel);

assert(sum(isinf(kernel(:)))==0 && sum(isnan(kernel(:)))==0)