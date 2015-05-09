if ~isdeployed
    learningtoolboxroot = pwd ;
    addpath(sprintf('%s/vlfeat-0.9.18/:%s/svm_using_libsvm/libsvm-3.18/matlab:%s/utils/:%s/LP_solver/:%s/svm_using_libsvm/:%s/train_test/:%s/cross_validation/',learningtoolboxroot,learningtoolboxroot,learningtoolboxroot,learningtoolboxroot,learningtoolboxroot,learningtoolboxroot,learningtoolboxroot), ...
    genpath(sprintf('%s/logistic_regression/',learningtoolboxroot)));
    
    run('vlfeat-0.9.18/toolbox/vl_setup.m');
end
