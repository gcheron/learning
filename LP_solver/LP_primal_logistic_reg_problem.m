function problem=LP_primal_logistic_reg_problem(x,y)
y=y==1 ;
flogistic=@(x)(1./(1+exp(-x)));

problem.objective =                @(w,lambda)-sum(y.*log(flogistic(x*w)) + (1-y).*log(1-flogistic(x*w)))+lambda*sum(w(1:end-1).*w(1:end-1)); % do not regularized the bias
problem.g  =                       @(w,lambda)(x'*(flogistic(x*w)-y)+[lambda*w(1:end-1) ; 0]);
problem.H  =                       @(w,lambda)(computeHess(w,lambda,x,flogistic));
end

function H=computeHess(w,lambda,x,flogistic)
Idl=eye(length(w))*lambda;
Idl(end)=0; % do not regularized the bias
ypred = flogistic(x*w) ;
R=diag(ypred.*(1-ypred));
H=(x'*R*x)+Idl;
end