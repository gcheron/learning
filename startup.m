if ~isdeployed
    currdir = pwd ;
    addpath(sprintf('%s/vlfeat-0.9.18/:%s/svm_using_libsvm/libsvm-3.18/matlab:%s/utils/:%s/LP_solver/:%s/svm_using_libsvm/:%s/train_test/:%s/cross_validation/',currdir,currdir,currdir,currdir,currdir,currdir,currdir), ...
    genpath(sprintf('%s/logistic_regression/',currdir)));
    
    run('vlfeat-0.9.18/toolbox/vl_setup.m');
end
