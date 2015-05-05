%% Kernel logistic regression - Homework 4
%% Cheron - Coquelin

Ntest = 1000;

Ntrain = 100;
p = 10;

sigmoidal = @(x)(1./(1+exp(-x)));

costPb1 = @(x,y,w,lambda)-sum(log(sigmoidal(y.*(x*w))))+lambda*sum(w.*w);
gradPb1 = @(x,y,w,lambda)-x'*(y.*(1-sigmoidal(y.*(x*w))))+2*lambda*w;
hessianPb1 = @(x,y,w,lambda)x'* sparse(1:size(x,1),1:size(x,1),sigmoidal(y.*(x*w)).*(1-sigmoidal(y.*(x*w))))*x + 2*lambda*eye(size(x,2));
crossErrorPb1 = @(x,y,xtrain,w)length(find(2*(sigmoidal(x*w)>.5)-1~=y)) / numel(y);



%% Problem (1) and generation of data

lastError = 0;
% we loop until we get a random dataset with nice properties
while (lastError < 0.05 || lastError > 0.3)
    generateData
    w = zeros(p,1);
    [ bestLambda ] = performCrossValidation(xTrain,yTrain,crossErrorPb1,costPb1,gradPb1,hessianPb1,w )
    
    % We retrain using the full training set
    testSetError = @(x,y,w)crossErrorPb1(xTest,yTest,x,w);
    
    [wPb1,scorePb1,errorPb1] = newtonRaphson(xTrain,yTrain,testSetError,costPb1,gradPb1,hessianPb1,bestLambda,w);
    lastError = errorPb1(end)
end


%% Problem (2)

% linear kernel
kernel_fun = @(x,y) x*y' ;

Gram = compute_kernel(xTrain,kernel_fun);
scoreC = @(x,y,w,lambda)-sum(log(sigmoidal(y.*(Gram*w))))+lambda*w'*Gram*w;
jacobianC = @(x,y,w,lambda) gradient2(x,y,w,lambda,kernel_fun,Gram);
hessianC = @(x,y,w,lambda) hessian2(x,y,w,lambda,kernel_fun,Gram);
testSetError = @(x,y,w) predictError2( xTest, yTest, xTrain, w, kernel_fun );

w = zeros(Ntrain,1);
[wPb2,scorePb2_linear,errorPb2_linear] = newtonRaphson(xTrain,yTrain,testSetError,scoreC,jacobianC,hessianC,bestLambda,w);

% gaussian kernel
sigma=10;
kernel_fun = @(x,y) exp(-norm(x-y)^2/(2*sigma^2)) ;

Gram = compute_kernel(xTrain,kernel_fun);
scoreC = @(x,y,w,lambda)-sum(log(sigmoidal(y.*(Gram*w))))+lambda*w'*Gram*w;
jacobianC = @(x,y,w,lambda) gradient2(x,y,w,lambda,kernel_fun,Gram);
hessianC = @(x,y,w,lambda) hessian2(x,y,w,lambda,kernel_fun,Gram);
crossValidError = @(x,y,xTrain,w) predictError2( x, y, xTrain, w, kernel_fun );
testSetError = @(x,y,w) predictError2( xTest, yTest, xTrain, w, kernel_fun );

% cross validation to select best lambda for gaussian kernel
w = zeros(Ntrain*4/5,1);
bestLambdaGaussian = bestLambda;
%[ bestLambdaGaussian ] = performCrossValidation(xTrain,yTrain,crossValidError,scoreC,jacobianC,hessianC,w )

w = zeros(Ntrain,1);
[wPb2,scorePb2_gaussian,errorPb2_gaussian] = newtonRaphson(xTrain,yTrain,testSetError,scoreC,jacobianC,hessianC,bestLambdaGaussian,w);



%% Problem (3)
% linear kernel
kernel_fun = @(x,y) x*y' ;
[wPb3,scorePb3_linear,errorPb3_linear] = newtonRaphson3(xTrain, yTrain, xTest, yTest, bestLambda, kernel_fun);

% gaussian kernel
sigma=10;
kernel_fun = @(x,y) exp(-norm(x-y)^2/(2*sigma^2)) ;
[wPb3,scorePb3_gaussian,errorPb3_gaussian] = newtonRaphson3(xTrain, yTrain, xTest, yTest, bestLambdaGaussian, kernel_fun);


%% Comparisons plots
cmap = hsv(5);
figure;
plot(scorePb1,'Color', cmap(1,:)); hold on;
plot(scorePb2_linear,'Color', cmap(2,:)); hold on;
plot(scorePb2_gaussian,'Color', cmap(3,:)); hold on;
plot(scorePb3_linear,'Color', cmap(4,:)); hold on;
plot(scorePb3_gaussian,'Color', cmap(5,:)); hold on;
legend('(1)', '(2) linear kernel', '(2) gaussian kernel', ...
    '(3) linear kernel', '(3) gaussian kernel');
ylabel('Objective'); xlabel('iterations');
title(sprintf('Ntrain=%d p=%d',Ntrain,p));
print(sprintf('Cost_%d_%d.png',Ntrain,p),'-dpng');

figure;
plot(errorPb1,'Color', cmap(1,:)); hold on;
plot(errorPb2_linear,'Color', cmap(2,:)); hold on;
plot(errorPb2_gaussian,'Color', cmap(3,:)); hold on;
plot(errorPb3_linear,'Color', cmap(4,:)); hold on;
plot(errorPb3_gaussian,'Color', cmap(5,:)); hold on;
legend('(1)', '(2) linear kernel', '(2) gaussian kernel', ...
    '(3) linear kernel', '(3) gaussian kernel');
ylabel('% test error'); xlabel('iterations');
title(sprintf('Ntrain=%d p=%d',Ntrain,p));
print(sprintf('Error_%d_%d.png',Ntrain,p),'-dpng');

%% Save generated data
fileName = sprintf('Ntrain%d-Ntest%d-p%d.mat' , Ntrain, Ntest, p);
save(fileName, 'xTrain', 'yTrain', 'xTest', 'yTest')