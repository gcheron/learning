% example on JHMDB split 1 with HLPF (L2)

nbclasses=21;
featdirraw='/sequoia/data1/gcheron/code/learning/train_test/data_example/HLPF_JHMDB_train1';

% get sample list names and mutliclass labels
[samplelist_train,yTrain]=get_sample_list('data_example/JHMDB_train1.txt',featdirraw,nbclasses);
[samplelist_test,yTest]=get_sample_list('data_example/JHMDB_test1.txt',featdirraw,nbclasses);

% load features
x = load(samplelist_train{1});
x = x.histogram(:) ;
xTrain =    zeros(length(samplelist_train),length(x));
xTest =     zeros(length(samplelist_test),length(x));
parfor i=1:length(samplelist_train)
    x = load(samplelist_train{i}); x = x.histogram(:) ;
    xTrain(i,:) = x(:)';
end
parfor i=1:length(samplelist_test)
    x = load(samplelist_test{i}); x = x.histogram(:) ;
    xTest(i,:) = x(:)';
end

% get kernels
try
    load('data_example/Ktrain','Ktrain');
    load('data_example/Ktest','Ktest');
catch
    [Ktrain,gamma] = conc_chi2_exp_kernel(xTrain',xTrain');
    [Ktest,~]   = conc_chi2_exp_kernel(xTest',xTrain',gamma);
    save('data_example/Ktrain','Ktrain');
    save('data_example/Ktest','Ktest');
end


%% SVM

% one-versus-all (with CV)
fun = svm_get_cv_fun();
svm=cell(nbclasses,1);
output_infolist=cell(nbclasses,1);
scores = zeros(size(yTest));
parfor c=1:nbclasses
    [svm{c},output_infolist{c}]=cv_main_c_beta(Ktrain,yTrain(:,c),fun,'beta_range', 2.^(-8:1:8), 'cost_range',logspace(-3,2.5,10),'cross_metric','AP');
    scores(:,c)=test_svm_kernel(Ktest,svm{c});
end

% standardize parameters
output_info_std=hyperparameters_standardization_c_beta(output_infolist);
cost = output_info_std.cost;
beta = output_info_std.beta;
rho = 1 ./ (1 + beta);

% one-versus-all standardized
svm_std=cell(nbclasses,1);
scores_std = zeros(size(yTest));
for c=1:nbclasses
    
    y = 2*(yTrain(:,c)==1)-1;
    nbpos = sum(y==1);
    nbneg = sum(y~=1);
    wp = 2 .* rho .* (nbpos+nbneg) / nbpos;
    wn = 2 .* (1 - rho) .* (nbpos+nbneg) / nbneg;
    
    svm_std{c} = svmtrain(y,[(1:size(Ktrain,1))' Ktrain], sprintf('-s 0 -t 4 -c %.9f -w1 %.9f -w-1 %.9f -q',cost,wp,wn));
    scores_std(:,c)=test_svm_kernel(Ktest,svm_std{c});
end


%% Kernel Logistic Regression
% one-versus-all (with CV)
fun = KLE_get_cv_fun();
KLE=cell(nbclasses,1);
output_infolist=cell(nbclasses,1);
scores = zeros(size(yTest));
parfor c=1:nbclasses
    [KLE{c},output_infolist{c}]=cv_main_c(Ktrain,yTrain(:,c),fun,'cost_range',logspace(-3,2.5,10),'cross_metric','AP');
    scores(:,c)=test_svm_kernel(Ktest,KLE{c});
end


%% Results
[~,per_sample_accuracy_svm,~]=multiclass_accuracy(scores,yTest)
[~,per_sample_accuracy_svmstd,~]=multiclass_accuracy(scores_std,yTest)


