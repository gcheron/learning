function conf=test_svm_kernel(kernel,svm)
if svm.Label(1) == -1
    error('Please check this case')
%     b = -model.rho;
%     w = -w;
%     b = -b;
end
%[~, ~,conf]  = svmpredict(ones(size(kernel,1),1),[(1:size(kernel,1))' kernel],svm,'-q') ;

conf=kernel(:,svm.sv_indices) * svm.sv_coef - svm.rho ;

end