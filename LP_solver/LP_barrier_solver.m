function [x_star,f_star] = LP_barrier_solver(problem,c,x0)
% Inequality form Linear Programming (LP) Solver using Barrier method
% minimize c'*x
% subj. to Ax<=b
%
% Large mu: less outer iterations but more inner (Newton) iterations

% Barrier parameters
mu = 20 ;
t0= 1 ;

% stop criterions
ep = 1e-6 ; % tolerance

%init
t=t0;
x=x0;
n=length(x0);
while true
    
    % centering step (minimize t*c'*x-sum(log(b-A*x))
    x=LP_analytic_center(problem,c,t,x);
    
    dgap = n/t ; % duality gap
    
    if dgap < ep % stop criterion
        break
    end
    
    % increase t
    t=t*mu;
end

x_star = x ;
f_star = problem.objective(x_star,c) ;

% corrected dual
%lambda_star = (-1 ./ (A*x_star - b) / t) .* (1 + (A*delta) ./ (b - A*x_star));
