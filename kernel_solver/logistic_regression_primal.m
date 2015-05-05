function problem=logistic_regression_primal(x,y)

sigmoid = @(x)(1./(1+exp(-x)));

problem.objective = @(w,c)-sum(log(sigmoid(y.*(x*w))))+c*sum(w.*w);
problem.g =         @(w,c)-x'*(y.*(1-sigmoid(y.*(x*w))))+2*c*w;
problem.H =         @(w,c)x'* sparse(1:size(x,1),1:size(x,1),sigmoid(y.*(x*w)).*(1-sigmoid(y.*(x*w))))*x + 2*c*eye(size(x,2));