function [x_star,s_star]=LP_phase_I(A,b)
% Phase I: computes strictly feasible starting point x0 for barrier method
% Here standard form Linear Programming (LP) case is considered:
% find x0 which is a strictly feasible solution of the inequalities:
% A*x-b <= 0
% solving:
% minimize_(x,s) s
% subject to     Ax-b <= s
%
% Starting with s0 > max(A*x-b), we minimize it trying to find a s<0


[m,n] = size(A);
x0 = zeros(n,1); % x0 is in the domain of the constraints (R^n)
s0 = max(A*x0-b)+1 ; % s0 is larger then the maximum constraint

A = [A,-ones(m,1)]; % to substract s to each equation: Ax-b-s <= 0
x0= [x0;s0];        % concatenate s0 to x0 
c = [zeros(n,1); 1];% minimize s only in the objective

problem.objective=@(x,c) (c'*x);
problem.objective_bar=@(x,c,t) (t*c'*x-sum(log(b-A*x)));
problem.violated_constraints =@(x,c) (min(b - A*x)<=0);
problem.g=@(x,c,t)(t*c+A'*(1./(b-A*x)));
problem.H=@(x,c,t)(A'*diag((1./(b-A*x)).^2)*A);

[res,~] = LP_barrier_solver(problem,c,x0); % leads to a strictly feasible problem

s_star = res(n+1);
x_star =res(1:n);
if (s_star>0)
    error('There is no strictly feasible solution!\n');
end
