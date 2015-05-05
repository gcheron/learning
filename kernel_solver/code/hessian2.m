function [ res ] = hessian2( x,y,w,lambda,kernel_fun,Gram)

sigmoidal = @(x)(1./(1+exp(-x)));
if (nargin == 5)
    Gram = compute_kernel(x,kernel_fun);
end
sigVect = sigmoidal(y.*(Gram*w));
sigVect = sigVect .* (1-sigVect);
res = Gram * diag(sigVect) * Gram + 2*lambda*Gram;
end

