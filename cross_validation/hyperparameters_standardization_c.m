function output_info=hyperparameters_standardization_c(output_infolist)
% Given a split and the cv_metrics according to each class
% ---> Select the hyperparameters that maximize the metric over all classes

nbclasses=length(output_infolist);
cv_metrics=0;
for i=1:nbclasses
    cv_metrics=cv_metrics+output_infolist{i}.cv_metrics ;
end
cv_metrics=cv_metrics/nbclasses;

[max_metric] = max(cv_metrics(:));

cost_range=output_infolist{1}.cost_range ;

assert(length(cv_metrics)==length(cost_range));

c=find(cv_metrics==max_metric) ;

if length(c) > 1
    if c(1) == 1
        c=c(2) ;
    else
        c=c(1) ;
    end
end

cost=cost_range(c);

output_info.cv_metrics=cv_metrics;
output_info.cost = cost ;
output_info.cost_position = sprintf('COST: %d out of %d [from %.3f to %.3f]',c(1),size(cost_range,2),cost_range(1),cost_range(end));
output_info.cost_range=cost_range;
output_info.max_metric = max_metric ;
