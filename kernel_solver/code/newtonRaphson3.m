function [w,score,error] = newtonRaphson3(xTrain, yTrain, xTest, yTest, lambda, kernel_fun)
%% Newton-Raphson algorithm for problem (3)
k = 1;
Ntrain = size(xTrain,1);
Gram = compute_kernel(xTrain,kernel_fun);

sigmoidal = @(x)(1./(1+exp(-x)));

scoreC = @(x,y,w,lambda)-sum(log(sigmoidal(y.*(Gram*(-w/2/lambda)))))+lambda*(-w/2/lambda)' * Gram * (-w/2/lambda);

jacobianC = @(x,y,w,lambda) y .* log( -y .* w ./ (1+y.*w)) - Gram * w / 2 / lambda;
hessianC = @(x,y,w,lambda) diag(1 ./ y ./ w  - 1 ./ (1 + y .* w)) - Gram / 2 / lambda;

errorC = @(x,y,w) predictError2( xTest, yTest, xTrain, -w/2/lambda, kernel_fun );

eps = 1e-4;
w = - eps .* yTrain;

score = scoreC(xTrain,yTrain,w,lambda); % initial score
error = errorC(xTrain,yTrain,w); % initial error
maxIterations = 40;

while k < maxIterations
    k = k +1;
    w_prev = w;
    
    % maximization problem => gradient ascent
    w        = w_prev - hessianC(xTrain,yTrain,w_prev,lambda) \ jacobianC(xTrain,yTrain,w,lambda); 
    
    % projection
    for i = 1:numel(w)
       if (w(i) * yTrain(i) <= -1)
           w(i) = (-1 + eps) * yTrain(i);
       elseif (w(i) * yTrain(i) >= 0)
           w(i)= ( - eps) * yTrain(i);
       end
    end
    
    score(k) = scoreC(xTrain,yTrain,w,lambda);
    error(k) = errorC(xTrain,yTrain,w);    
        
    % convergence criterion
    if sqrt(sum((w-w_prev).^2))/sqrt(sum(w.^2))<.01
        break
    end
end
end