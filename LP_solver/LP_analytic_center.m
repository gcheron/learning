function x_star= LP_analytic_center(problem,c,t,x0)
% Analytic Center of standard form Linear Programming (LP).
% minimize f0=problem.objective_bar        (e.g.) t*c'*x-sum(log(b-A*x))
% (unconstrained problem)
% using damped Newton method with newton step computed by second order
% approximation because there is no constraint in the minimization problem
% x_0 is supposed to be strictly feasible

% stop criterions
ep = 1e-4 ; % tolerance
max_it = 100 ; % maximum number of iterations

% line search parameters
alpha = 0.01 ; % decrease in f acceptance
beta = 0.5 ; % the higher beta, the less crude the search will be


if (problem.violated_constraints(x0,c)) % x0 is not strictly feasible
    fprintf('not feasible');
    return
end

f=@(x)(problem.objective_bar(x,c,t)); % objective (Barrier)
gf=@(x)(problem.g(x,c,t)); % gradient
Hf=@(x)(problem.H(x,c,t)); % hessian

% init
x=x0;
for it = 1:max_it
    g=gf(x);
    H=Hf(x);
    
    % newton step by second order approximation
    delta = - H\g;
    
    lambda_2 = g'*-delta; % newton decrement
    
    if lambda_2 / 2 <= ep % stop criterion
        break ;
    end
    
    % backtracking line search
     t=1;
    while problem.violated_constraints(x+t*delta,c) % ensure that line search stays in feasible set
        t=beta*t ;
    end
    
    while (f(x+t*delta) >= f(x) + alpha*t*g'*delta) % line search
        t=beta*t ;
        if t < eps
            x_star=x;
            return
            %error('In line search: decrease ep (tolerance value)');
        end
    end
    
    x = x + t * delta ; % x step
    %assert(min(b - A*x) > 0) % x is still feasible
end

if it == max_it
    error ('Maximum number of iterations has been reached');
end

% optimal value
x_star=x;

end



