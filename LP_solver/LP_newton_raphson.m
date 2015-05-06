function x_star= LP_newton_raphson(problem,c,x0)
% minimize the unconstrained problem
% using Newton-Raphson method

% stop criterions
ep = 1e-4 ; % tolerance
max_it = 100 ; % maximum number of iterations

f=@(x)(problem.objective(x,c)); % objective
gf=@(x)(problem.g(x,c)); % gradient
Hf=@(x)(problem.H(x,c)); % hessian

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
    
    x = x + delta ; % x step
end

if it == max_it
    warning ('Maximum number of iterations has been reached');
end

% optimal value
x_star=x;

end



