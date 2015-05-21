function [model,output_info]=cv_main_c(kernel,labels,fun,varargin)

% ---- Default params ------------------------------------------
cross_parameters.cost_range =logspace(-6,3,10);
cross_parameters.K = 5; % K-fold number
cross_parameters.rand_cross=0 ; % regular or random splits?
cross_parameters.display_evolution=1;
cross_parameters.display_metrics=0;
cross_parameters.display_results=1;
cross_parameters.cross_metric='AP';

cross_parameters = vl_argparse(cross_parameters, varargin) ;
% --------------------------------------------------------------

nbpos = sum(labels==1);
nbneg = sum(labels~=1);

labels(labels~=1) = -1 ;

if nbneg < 2 || nbpos < 2
    error(['Only ' num2str(nbneg) ' negative examples and ' num2str(nbpos) ' positive examples']) ;
end

%[output_info,conf_pos,conf_neg]=cv_c(kernel,labels,cross_parameters);
% output_info.conf_pos = conf_pos ;
% output_info.conf_neg = conf_neg ;

output_info=cv_c(kernel,labels,fun,cross_parameters);

cost= output_info.cost ;
max_metric= output_info.max_metric ;



% train on the whole dataset with selected parameters
hyperparams.cost = cost ;
model = fun.train(kernel,labels,hyperparams);

% train error
conf = fun.test(model,kernel);
[~, ~, info] = vl_pr(labels, conf); % AP
acc = fun.accuracy(conf,labels); % accuracy

output_info.train_ap = info.ap ;
output_info.train_acc = acc ;

if cross_parameters.display_results
    disp(['COST=' num2str(cost)]) ;
    disp(['MAX_metric=' num2str(max_metric)]) ;
    disp(output_info.cost_position);
    fprintf('AP on train: %.2f\n',info.ap) ;
    fprintf('Accuracy on train: %.2f\n',acc) ;
end