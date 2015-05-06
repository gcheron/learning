function SVM_problem=LP_primal_SVM_barrier_problem(x,y,n,m)
% Bias should be integrated into x: x = [x; ones(1,m)];
assert(sum(x(end,:)~=1) == 0)

M = repmat(y',n,1) .* x ;
g=zeros(n+m,1);
H=zeros(n+m);

SVM_problem.objective =             @(w,c)(0.5 * w(1:n)'* w(1:n) + c*sum(w(n+1:end))) ;
SVM_problem.violated_constraints =  @(w,c) (min(y.*(w(1:n)'*x)'-1+w(n+1:end))<0 || min(w(n+1:end))<0);
SVM_problem.objective_bar =         @(w,c,t)(t*SVM_problem.objective(w,c) - sum(log(y.*(w(1:n)'*x)'-1+w(n+1:end))) - sum(log(w(n+1:end)))); 
SVM_problem.g  =                    @(w,c,t)(computeGrad(w,c,t,M,n,g));
SVM_problem.H  =                    @(w,c,t)(computeHess(w,t,x,y,M,n,H));

end

function g=computeGrad(w,c,t,M,n,g)

wp=w(1:n);
z = w(n+1:end) ;

g(1:n) = t*wp - M * (1 ./ (M'*wp-1+z));
g(n+1:end) = t*c - 1/((wp'*M)'-1+z) - 1/z;

end


function H=computeHess(w,t,x,y,M,n,H)

wp = w(1:n);
z = w(n+1:end);

H(1:n,1:n) = t*eye(n)+M*diag((1./(M'*wp-1+z)).^2)*M';
H(n+1:end,n+1:end) = diag(1./(y.*(wp'*x)'-1+z).^2+1./z.^2);

for i=1:size(z,1)
    H(1:n,n+i) = y(i)*x(:,i)/(y(i)*x(:,i)'*wp-1+z(i))^2;
    H(n+i,1:n) = H(1:n,n+i);
end

end