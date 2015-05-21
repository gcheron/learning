function fun=KLE_get_cv_fun(param)

ep = 1e-4 ; % tolerance
max_it = 100 ; % maximum number of iterations
if nargin > 0
    if isfield(param,'ep')
        ep = param.ep;
    end
    if isfield(param,'max_it')
        max_it = param.max_it ;
    end
end

flogistic=@(x)(1./(1+exp(-x)));
KLE_problem=@(kernel,labels)(LP_dual_logistic_reg_problem(kernel,labels));

fun.train = @(kernel,labels,hyperparams)(get_trainfun(kernel,labels,hyperparams,KLE_problem,ep,max_it)) ;
fun.test = @(model,kernel)(get_testfun(model,kernel,flogistic));
fun.accuracy = @(conf,labels) (get_accuracyfun(conf,labels));

function model = get_trainfun(kernel,labels,hyperparams,KLE_problem,ep,max_it)
pb=KLE_problem(kernel,labels);
param.ep = ep ;
param.max_it = max_it ;
model.alpha = LP_newton_raphson(pb,1/hyperparams.cost,zeros(size(kernel,1),1),param);

function conf = get_testfun(model,kernel,flogistic)
conf=flogistic(kernel * model.alpha);

function acc = get_accuracyfun(conf,labels)
acc = (sum(conf(labels==1)>=0.5) + sum(conf(labels~=1)<0.5))/(length(labels)) ;