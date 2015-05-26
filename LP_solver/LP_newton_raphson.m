function x_star= LP_newton_raphson(problem,c,x0,param)
% minimize the unconstrained problem
% using Newton-Raphson method

% stop criterions
ep = 1e-4 ; % tolerance
max_it = 100 ; % maximum number of iterations

if nargin > 3
    if isfield(param,'ep')
        ep = param.ep;
    end
    if isfield(param,'max_it')
        max_it = param.max_it ;
    end
end


f=@(x)(problem.objective(x,c)); % objective
gf=@(x)(problem.g(x,c)); % gradient
Hf=@(x)(problem.H(x,c)); % hessian

% init
x=x0;
lambda_2=Inf ;
for it = 1:max_it
    lambda_2_prev=lambda_2;
    
    g=gf(x);
    H=Hf(x);
    
    % newton step by second order approximation
    delta = - H\g;
    
    lambda_2 = g'*-delta; % newton decrement
    
    if lambda_2 / 2 <= ep || lambda_2 > lambda_2_prev || isnan(lambda_2) % stop criterion
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



