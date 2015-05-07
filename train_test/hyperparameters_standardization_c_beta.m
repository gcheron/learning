function output_info=hyperparameters_standardization_c_beta(output_infolist,savefig,split)
% Given a split and the cv_metrics according to each class
% ---> Select the hyperparameters that maximize the metric over all classes
if nargin < 2
    savefig = 0 ;
end
nbclasses=length(output_infolist);
cv_metrics=0;
for i=1:nbclasses
    cv_metrics=cv_metrics+output_infolist{i}.cv_metrics ;
end
cv_metrics=cv_metrics/nbclasses;

[max_metric] = max(cv_metrics(:));

beta_range=output_infolist{1}.beta_range ;
cost_range=output_infolist{1}.cost_range ;

assert(size(cv_metrics,1)==length(beta_range));
assert(size(cv_metrics,2)==length(cost_range));

[b,c]=find(cv_metrics==max_metric) ;

if length(c) > 1
    if c(1) == 1
        c=c(2) ;
    else
        c=c(1) ;
    end
end
if length(b) > 1
    if b(1) == 1
        b=b(2) ;
    else
        b=b(1) ;
    end
end

beta=beta_range(b);
cost=cost_range(c);

output_info.cv_metrics=cv_metrics;
output_info.cost = cost ;
output_info.cost_position = sprintf('COST: %d out of %d [from %.3f to %.3f]',c(1),size(cost_range,2),cost_range(1),cost_range(end));
output_info.cost_range=cost_range;
output_info.max_metric = max_metric ;
output_info.beta = beta ;
output_info.beta_position = sprintf('BETA: %d out of %d [from %.3f to %.3f]',b(1),size(beta_range,2),beta_range(1),beta_range(end));
output_info.beta_range=beta_range ;


if savefig
    imagesc(cv_metrics) ;
    
    xlabel('C index');
    ylabel('beta index');
    title(sprintf('Split %d: mAP (max(mAP)=%.3f)\nC=%.3f, beta=%.3f (rho=%.3f)\nC index=%d/%d, beta index=%d/%d',split,max_metric,cost,beta,1/(1+beta),c,length(cost_range),b,length(beta_range)));
    colorbar
    print('-djpeg',sprintf('mAp_split%d',split));
    
    imagesc(std_cv_metrics) ;
    xlabel('C index');
    ylabel('beta index');
    title(sprintf('Split %d: std of mAP\nfor selected C=%.3f and beta=%.3f\nstd(mAP)=%.3f ',split,cost,beta,std_cv_metrics(b,c)));
    colorbar
    print('-djpeg',sprintf('stdmAp_split%d',split));
end
end
