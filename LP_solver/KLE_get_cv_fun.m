function fun=KLE_get_cv_fun()
% 
% fun.train = @(kernel,labels,hyperparams)(get_trainfun(kernel,labels,hyperparams)) ;
% fun.test = @(model,kernel)(get_testfun(model,kernel));
% 
% function model = get_trainfun(kernel,labels,hyperparams)
% if isfield(hyperparams,'wp')
%     parameter_string = sprintf('-s 0 -t 4 -c %.9f -w1 %.9f -w-1 %.9f -q',hyperparams.cost,hyperparams.wp,hyperparams.wn);
% else
%     parameter_string = sprintf('-s 0 -t 4 -c %.9f -q',hyperparams.cost);
% end
% model = svmtrain(labels,[(1:size(kernel,1))' kernel], parameter_string);
% 
% function conf = get_testfun(model,kernel)
% labels=zeros(size(kernel,1),1);
% [~, ~, conf]  = svmpredict(labels,[(1:size(kernel,1))' kernel],model,'-q');