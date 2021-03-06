function [model,output_info]=cv_main_overfittest_c_beta(kernel,labels,ktest,ytest,fun,varargin)

% ---- Default params ------------------------------------------
cross_parameters.cost_range =logspace(-6,3,10);
cross_parameters.beta_range = 2.^(-5:1:5);
cross_parameters.display_evolution=1;
cross_parameters.display_metrics=0;
cross_parameters.display_results=1;
cross_parameters.cross_metric='AP';

cross_parameters = vl_argparse(cross_parameters, varargin) ;
% --------------------------------------------------------------

labels(labels~=1) = -1 ;
ytest(ytest~=1) = -1 ;
output_info=cv_overfit_c_beta(kernel,labels,ktest,ytest,fun,cross_parameters);

cost= output_info.cost ;
wp= output_info.wp ;
wn= output_info.wn ;
beta= output_info.beta;
max_metric= output_info.max_metric ;

% train on the whole dataset with selected parameters
hyperparams.cost = cost ;
hyperparams.wp = wp ;
hyperparams.wn = wn ;
model = fun.train(kernel,labels,hyperparams);

% train error
conf = fun.test(model,ktest);
[~, ~, info] = vl_pr(labels, conf); % AP
acc = fun.accuracy(conf,labels); % accuracy

output_info.train_ap = info.ap ;
output_info.train_meanrec = meanrec ;
output_info.train_acc = acc ;

if cross_parameters.display_results
    disp(['COST=' num2str(cost)]) ;
    disp(['wp=' num2str(wp)]) ;
    disp(['wn=' num2str(wn)]) ;
    disp(['BETA=' num2str(beta)]) ;
    disp(['MAX_metric=' num2str(max_metric)]) ;
    disp(output_info.cost_position);
    disp(output_info.beta_position);
    fprintf('AP on test: %.2f\n',info.ap) ;
    fprintf('Accuracy on test: %.2f\n',acc) ;
end