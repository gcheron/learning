function [ res ] = gradient2( x,y,w,lambda,kernel_fun,Gram )

sigmoidal = @(x)(1./(1+exp(-x)));
if (nargin == 5)
    Gram = compute_kernel(x,kernel_fun);
end
sigVect = (1-sigmoidal(y.*(Gram*w)));
res = Gram * (-y .*sigVect) + 2*lambda*Gram*w;

end

