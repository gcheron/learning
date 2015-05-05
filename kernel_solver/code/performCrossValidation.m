function [ bestLambda ] = performCrossValidation(xTrain,yTrain,crossError,scoreC,jacobianC,hessianC,w )


% generate candidate regularization coefficients, lambda
Nlambdas                = 30;
lambda_range            = logsample(0.0001, 10000,Nlambdas);
K = 5; % perform K-fold cross validation

Ntrain = size(xTrain,1);
%% 
% We implement cross-validation to determine the best regularization
% coefficient $\lambda$. 
for i=1:Nlambdas
    lambda = lambda_range(1,i);    
    
    for validation_run=1:K       
        % split data into training set (trset) and validation set (vlset)
        [trset_features,trset_labels,vlset_features,vlset_labels] =  ...
            split_data(xTrain',yTrain',Ntrain,K,validation_run);
                
        
        emptyFunc = @(x,y,w) 0;
        [w,score] = newtonRaphson(trset_features',trset_labels',emptyFunc,scoreC,jacobianC,hessianC,lambda,w);
        
        nerrors(1,validation_run) = crossError(vlset_features', vlset_labels',trset_features',w);
    end
    
    cv_error(i)=mean(nerrors,2); % The cross-validation error is the mean of the error
end

%%
% Cross validation error as a function of the regularization coefficient
% $\lambda$ :
%figure,plot(cv_error),title('Cross validation error');  

%%
% We pick $\lambda_0^*$ that minimizes the cross-validation error:
bestLambda = lambda_range(find(cv_error==min(cv_error),1));



end

