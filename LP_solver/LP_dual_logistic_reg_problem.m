function problem=LP_dual_logistic_reg_problem(K,y)
y(y~=1)=-1 ;
flogistic=@(x)(1./(1+exp(-x)));

problem.objective =                @(alpha,lambda) - sum(log(flogistic(y.*(K*alpha))))+lambda*alpha'*K*alpha ;
problem.g  =                       @(alpha,lambda) - K*(y.*(1-flogistic(y.*(K*alpha)))) + 2*lambda* (K*alpha);
problem.H  =                       @(alpha,lambda) (computeHess(alpha,lambda,K,flogistic,y));
end

function H=computeHess(alpha,lambda,K,flogistic,y)
fsig = flogistic(y.*(K*alpha)) ;
fsig = fsig .* (1-fsig);
H = (K*diag(fsig)*K)  + 2*lambda * K ;
end

% check Hessian
% x = rand(8,3); K=x*x';
% y = [ones(4,1) ; -ones(4,1)]; alpha = rand(8,1) ; lambda = 0.1 ; 
% H2=zeros(8);
% for p = 1:8
%     for k=1:8
%         H2(p,k) = sum(K(p,:)'.*K(k,:)'.*flogistic(y.*(K*alpha)).*(1-flogistic(y.*(K*alpha)))) + 2*lambda * K(p,k) ;
%     end
% end
% (H-H2)