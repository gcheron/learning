function output_info=cv_overfit_c_beta(kernel,labels,ktest,ytest,fun,cross_parameters)

nbpos = sum(labels==1);
nbneg = sum(labels~=1);

beta_range=cross_parameters.beta_range;
cost_range=cross_parameters.cost_range;

display_evolution=cross_parameters.display_evolution ;
display_APs=cross_parameters.display_metrics ;

cv_metrics=zeros(length(beta_range),length(cost_range));

for b=1:size(beta_range,2);
    beta = beta_range(b) ; 
    
    rho = 1 ./ (1 + beta);
    wp = 2 .* rho .* (nbpos+nbneg) / nbpos;
    wn = 2 .* (1 - rho) .* (nbpos+nbneg) / nbneg;
    
    
    for c=1:size(cost_range,2);
        cost   = cost_range(c);
        
        if display_evolution
            fprintf('Beta = %.3f  [%i out of %i]  | C = %.3f  [%i out of %i]\n',beta,b,size(beta_range,2),cost,c,size(cost_range,2));
        end
        hyperparams.cost = cost ;
        hyperparams.wp = wp ;
        hyperparams.wn = wn ;
        cv_metrics(b,c)=cv_overtit_valfun(kernel,labels,ktest,ytest,fun,hyperparams,cross_parameters);

        if display_APs
            disp(cv_metrics)
        end        
        
    end
end

max_metric=max(max(cv_metrics));

[b,c]=find(cv_metrics==max_metric) ;

beta=beta_range(b(1));
rho = 1 / (1 + beta);
wp = 2 * rho * (nbpos+nbneg) / nbpos;
wn = 2 * (1 - rho) * (nbpos+nbneg) / nbneg;

cost=cost_range(c(1));

output_info.cv_metrics=cv_metrics;
output_info.cost = cost ;
output_info.cost_position = sprintf('COST: %d out of %d [from %.3f to %.3f]',c(1),size(cost_range,2),cost_range(1),cost_range(end));
output_info.cost_range=cost_range;
output_info.wp = wp ;
output_info.wn = wn ;
output_info.max_metric = max_metric ;
output_info.beta = beta ;
output_info.beta_position = sprintf('BETA: %d out of %d [from %.3f to %.3f]',b(1),size(beta_range,2),beta_range(1),beta_range(end));
output_info.beta_range=beta_range;
end