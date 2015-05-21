function [metric_mean,conf_pos,conf_neg]=cv_valfun(kernel,labels,fun,hyperparams,cross_parameters)

labels(labels~=1) = -1 ;
K=cross_parameters.K;
metric_value_tab  = zeros(1,K);

if nargout > 1
    conf_pos = [] ;
    conf_neg = [] ;
    save_conf=1 ;
else
    save_conf=0;
end

rand_cross=cross_parameters.rand_cross;

if rand_cross
    trainp=cross_parameters.trainp;
    min_wanted = cross_parameters.randfold_min_wanted ;
else
    folds = cross_parameters.folds ;
end


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

for k=1:K
    
    % use pre-computed kernel
    if rand_cross
        [ind_train,ind_val,trset_labels,vlset_labels] = split_data_idx_rand(labels,trainp,min_wanted) ;
    else
        ind_train=folds.ind_train{k};
        ind_val=folds.ind_val{k};
        trset_labels=folds.labels_train{k};
        vlset_labels=folds.labels_val{k};
    end
    
    trainposnum = sum(trset_labels==1) ;
    trainnegnum = sum(trset_labels~=1) ;
    validposnum = sum(vlset_labels==1) ;
    validnegnum = sum(vlset_labels~=1) ;
    assert(trainposnum >=1 && trainnegnum >=1 && validposnum >=1 && validnegnum >=1)
    
    train_kernel = kernel(ind_train,ind_train);
    model = fun.train(train_kernel,trset_labels,hyperparams);
    
    val_kernel = kernel(ind_val,ind_train);
    conf = fun.test(model,val_kernel);
    
    if save_conf
        conf_pos = [conf_pos ; conf(vlset_labels==1) ] ;
        conf_neg = [conf_neg ; conf(vlset_labels~=1) ] ;
    end
    
    % prevent from saturation (scores are almost all the same):
    % put negative samples first
    posidx=vlset_labels==1;
    negidx=vlset_labels~=1;
    vlset_labels=[vlset_labels(negidx);  vlset_labels(posidx)];
    conf=[conf(negidx);  conf(posidx)];
    
    if sum(isinf(conf)) + sum(isnan(conf)) > 0
        metric_value_tab(k) = -Inf ;
        break ;
    end
    
    % get metric value
    if cross_metric == 1
        [~, ~, info] = vl_pr(vlset_labels, conf);
        metric_value = info.auc; 
    elseif cross_metric == 2
        TP = sum(conf(vlset_labels==1)>=0) ;
        TN = sum(conf(vlset_labels~=1)<0) ;
        metric_value = 0.5*(TP/validposnum + TN/validnegnum);  % recall for positive and negative classes
    elseif cross_metric == 3 % Accuracy
        metric_value = fun.accuracy(conf,vlset_labels);
    end
    metric_value_tab(k)=metric_value;
end

metric_mean=mean(metric_value_tab);