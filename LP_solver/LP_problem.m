function problem=LP_problem(A,b)

problem.objective=@(x,c,t) (c'*x);
problem.objective_bar=@(x,c,t) (t*c'*x-sum(log(b-A*x)));
problem.violated_constraints =@(x,c) (min(b - A*x)<=0);
problem.g=@(x,c,t)(t*c+A'*(1./(b-A*x)));
problem.H=@(x,c,t)(A'*diag((1./(b-A*x)).^2)*A);

end