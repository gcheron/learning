function SVM_dual_problem=LP_dual_SVM_problem(x,y)
K=x'*x;
g = zeros(size(K,1),1);
H = zeros(size(K));

SVM_dual_problem.objective =                @(alpha,c)0.5*alpha'*diag(y)*K*diag(y)*alpha - sum(alpha);
SVM_dual_problem.violated_constraints =     @(alpha,c)  sum(alpha<0) + sum(alpha>c);
SVM_dual_problem.objective_bar =            @(alpha,c,t)(t*SVM_dual_problem.objective(alpha) - sum(log(sum(alpha))) - sum(log(alpha)) - sum(log(c-alpha)));
SVM_dual_problem.g  =                       @(alpha,c,t)(computeGrad(alpha,c,K,y,g));
SVM_dual_problem.H  =                       @(alpha,c,t)computeHess(K,y,H);
end

function g=computeGrad(alpha,c,K,y,g)
for i=1:numel(alpha)
    g(i) = y(i)*sum(alpha.*y.*(K(i,:)')) - 1 - 1/alpha(i)- 1/(c-alpha(i));
end
end


function H=computeHess(K,y,H)
for i = 1 :size(K,1)
    H(i,:) = y(i)*y'.*K(i,:);
end
end