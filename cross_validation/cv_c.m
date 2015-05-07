function [output_info,conf_pos,conf_neg]=cv_c(kernel,labels,fun,cross_parameters)

nbpos = sum(labels==1);
nbneg = sum(labels~=1);

cost_range=cross_parameters.cost_range;

display_evolution=cross_parameters.display_evolution ;
display_APs=cross_parameters.display_metrics ;

cv_metrics=zeros(length(cost_range),1);


rand_cross=cross_parameters.rand_cross;

if ~rand_cross
    cross_parameters.folds = split_data_idx(labels,cross_parameters.K) ;
end

for c=1:size(cost_range,2);
        cost   = cost_range(c);
        
        if display_evolution
            fprintf('C = %.3f  [%i out of %i]\n',cost,c,size(cost_range,2));
        end
        hyperparams.cost = cost ;
        cv_metrics(c)=cv_valfun(kernel,labels,fun,hyperparams,cross_parameters);

        if display_APs
            disp(cv_metrics)
        end        
        
end

max_metric=max(cv_metrics);

c=find(cv_metrics==max_metric) ;
cost=cost_range(c(1));


if nargout > 1 % retrain with the selected parameters to get confidences on validation sets
    [~,conf_pos,conf_neg]=cv_valfun(kernel,labels,fun,hyperparams,cross_parameters);
end
output_info.cv_metrics=cv_metrics;
output_info.cost = cost ;
output_info.cost_position = sprintf('COST: %d out of %d [from %.3f to %.3f]',c(1),size(cost_range,2),cost_range(1),cost_range(end));
output_info.cost_range=cost_range;
output_info.max_metric = max_metric ;
end