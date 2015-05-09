function model=KLE_train(kernel,labels,hyperparams,param)

ep = 1e-4 ; % tolerance
max_it = 100 ; % maximum number of iterations
if nargin > 3
    if isfield(param,'ep')
        ep = param.ep;
    end
    if isfield(param,'max_it')
        max_it = param.max_it ;
    end
end
pb=LP_dual_logistic_reg_problem(kernel,labels);
param.ep = ep ;
param.max_it = max_it ;
model.alpha = LP_newton_raphson(pb,1/hyperparams.cost,zeros(size(kernel,1),1),param);