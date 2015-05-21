% example on JHMDB split 1 with HLPF (L2)
nbclasses=21;
featdirraw=[learningtoolboxroot '/train_test/data_example/HLPF_JHMDB_train1'];

% get sample list names and mutliclass labels
[samplelist_train,yTrain]=get_sample_list([learningtoolboxroot '/train_test/data_example/JHMDB_train1.txt'],featdirraw,nbclasses);
[samplelist_test,yTest]=get_sample_list([learningtoolboxroot '/train_test/data_example/JHMDB_test1.txt'],featdirraw,nbclasses);

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
    load([learningtoolboxroot '/train_test/data_example/Ktrain'],'Ktrain');
    load([learningtoolboxroot '/train_test/data_example/Ktest'],'Ktest');
catch
    xTrain = [xTrain ones(size(xTrain,1),1)];
    xTest =   [xTest  ones(size(xTest,1),1)];
    [Ktrain,gamma] = conc_chi2_exp_kernel(xTrain',xTrain');
    [Ktest,~]   = conc_chi2_exp_kernel(xTest',xTrain',gamma);
    save([learningtoolboxroot '/train_test/data_example/Ktrain'],'Ktrain');
    save([learningtoolboxroot '/train_test/data_example/Ktest'],'Ktest');
end


%% SVM

% one-versus-all (with CV)
fun = svm_get_cv_fun();
svm=cell(nbclasses,1);
output_infolist=cell(nbclasses,1);
scoressvm = zeros(size(yTest));
parfor c=1:nbclasses
    [svm{c},output_infolist{c}]=cv_main_c_beta(Ktrain,yTrain(:,c),fun,'beta_range', 2.^(-8:1:8), 'cost_range',logspace(-3,2.5,10),'cross_metric','AP');
    scoressvm(:,c)=test_svm_kernel(Ktest,svm{c});
end

% standardize parameters
output_info_std=hyperparameters_standardization_c_beta(output_infolist);
cost = output_info_std.cost;
beta = output_info_std.beta;
rho = 1 ./ (1 + beta);

% one-versus-all standardized
svm_std=cell(nbclasses,1);
scoressvm_std = zeros(size(yTest));
for c=1:nbclasses
    
    y = 2*(yTrain(:,c)==1)-1;
    nbpos = sum(y==1);
    nbneg = sum(y~=1);
    wp = 2 .* rho .* (nbpos+nbneg) / nbpos;
    wn = 2 .* (1 - rho) .* (nbpos+nbneg) / nbneg;
    
    svm_std{c} = svmtrain_libsvm(y,[(1:size(Ktrain,1))' Ktrain], sprintf('-s 0 -t 4 -c %.9f -w1 %.9f -w-1 %.9f -q',cost,wp,wn));
    scoressvm_std(:,c)=test_svm_kernel(Ktest,svm_std{c});
end


%% Kernel Logistic Regression
% one-versus-all (with CV)
fun = KLE_get_cv_fun();
KLE=cell(nbclasses,1);
output_infolist=cell(nbclasses,1);
scores = zeros(size(yTest));
parfor c=1:nbclasses
    [KLE{c},output_infolist{c}]=cv_main_c(Ktrain,yTrain(:,c),fun,'cost_range',logspace(-1,10,25),'cross_metric','AP');
    scores(:,c)=KLE_test(KLE{c}.alpha,Ktest);
end

% standardize parameters
output_info_std=hyperparameters_standardization_c(output_infolist);
cost = output_info_std.cost;
hyperparams.cost = cost ;

% one-versus-all standardized
KLE_std=cell(nbclasses,1);
scores_std = zeros(size(yTest));
for c=1:nbclasses  
    y = 2*(yTrain(:,c)==1)-1;
    KLE_std{c} = KLE_train(Ktrain,y,hyperparams);
    scores_std(:,c)=KLE_test(KLE_std{c}.alpha,Ktest);
end



%% Results
[~,per_sample_accuracy_svm,~]=multiclass_accuracy(scoressvm,yTest)
[~,per_sample_accuracy_svmstd,~]=multiclass_accuracy(scoressvm_std,yTest)
[~,per_sample_accuracy_KLE,~]=multiclass_accuracy(scores,yTest)
[~,per_sample_accuracy_KLEstd,~]=multiclass_accuracy(scores_std,yTest)



%% Overfit hyperparameters on test to see if problem is solvable
% one-versus-all (with CV)
fun = svm_get_cv_fun();
svm=cell(nbclasses,1);
output_infolist=cell(nbclasses,1);
scoressvm = zeros(size(yTest));
parfor c=1:nbclasses
    [svm{c},output_infolist{c}]=cv_main_overfittest_c_beta(Ktrain,yTrain(:,c),Ktest,yTest(:,c),fun,'beta_range', 2.^(-8:1:8), 'cost_range',logspace(-3,2.5,10),'cross_metric','AP','display_evolution',0);
    scoressvm(:,c)=test_svm_kernel(Ktest,svm{c});
end

% standardize parameters
output_info_std=hyperparameters_standardization_c_beta(output_infolist);
cost = output_info_std.cost;
beta = output_info_std.beta;
rho = 1 ./ (1 + beta);

% one-versus-all standardized
svm_std=cell(nbclasses,1);
scoressvm_std = zeros(size(yTest));
for c=1:nbclasses
    
    y = 2*(yTrain(:,c)==1)-1;
    nbpos = sum(y==1);
    nbneg = sum(y~=1);
    wp = 2 .* rho .* (nbpos+nbneg) / nbpos;
    wn = 2 .* (1 - rho) .* (nbpos+nbneg) / nbneg;
    
    svm_std{c} = svmtrain_libsvm(y,[(1:size(Ktrain,1))' Ktrain], sprintf('-s 0 -t 4 -c %.9f -w1 %.9f -w-1 %.9f -q',cost,wp,wn));
    scoressvm_std(:,c)=test_svm_kernel(Ktest,svm_std{c});
end

fprintf('Overfitted scores:\n');
[~,per_sample_accuracy_svm,~]=multiclass_accuracy(scoressvm,yTest)
[~,per_sample_accuracy_svmstd,~]=multiclass_accuracy(scoressvm_std,yTest)


