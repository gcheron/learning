function [w,score,error] = newtonRaphson(X,Y,errorC,scoreC,jacobianC,hessianC,lambda, w)
% Newton-Raphson algorithm
k = 1;
score = scoreC(X,Y,w,lambda); % initial score
error = errorC(X,Y,w); % initial error
maxIterations = 40;

while k < maxIterations
    k = k +1;
    w_prev = w;
    
    % update w (Newton-Raphson)
    w        = w_prev - hessianC(X,Y,w_prev,lambda) \ jacobianC(X,Y,w,lambda); 
    score(k) = scoreC(X,Y,w,lambda);
    error(k) = errorC(X,Y,w);
        
    % convergence criterion
    if sqrt(sum((w-w_prev).^2))/sqrt(sum(w.^2))<.01
        break
    end
end
end