function metric_value=cv_overtit_valfun(kernel,labels,ktest,ytest,fun,hyperparams,cross_parameters)

labels(labels~=1) = -1 ;
ytest(ytest~=1) = -1 ;

switch cross_parameters.cross_metric
    case 'AP'
        cross_metric=1 ;
    case 'MeanRecalls'
        cross_metric=2 ;
    case 'Accuracy'
        cross_metric=3 ;
    otherwise
        error('Unknown cross-validation metric');
end

model = fun.train(kernel,labels,hyperparams);
conf = fun.test(model,ktest);

% prevent from saturation (scores are almost all the same):
% put negative samples first
posidx=ytest==1;
negidx=ytest~=1;
ytest=[ytest(negidx);  ytest(posidx)];
conf=[conf(negidx);  conf(posidx)];

if sum(isinf(conf)) + sum(isnan(conf)) > 0
    metric_value = -Inf ;
    return ;
end

% get metric value
validposnum = sum(ytest==1) ;
validnegnum = sum(ytest~=1) ;
if cross_metric == 1
    [~, ~, info] = vl_pr(ytest, conf);
    metric_value = info.auc;
elseif cross_metric == 2
    TP = sum(conf(ytest==1)>=0) ;
    TN = sum(conf(ytest~=1)<0) ;
    metric_value = 0.5*(TP/validposnum + TN/validnegnum);  % recall for positive and negative classes
elseif cross_metric == 3 % Accuracy
    metric_value = (sum(conf(ytest==1)>=0) + sum(conf(ytest~=1)<0))/(validposnum+validnegnum) ;
end

