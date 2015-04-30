function [w,b]=train_lg_newton_raphson_multiclass(X,Y,param)

%-----default parameters-----
maxit=500;
lambda=0;
stopth=10^-8;
displayevolution=0;
%----------------------------

if nargin > 2
    
    if isfield(param,'lambda')
        lambda=param.lambda;
    end
    if isfield(param,'maxit')
        maxit=param.maxit;
    end
    if isfield(param,'stopth')
        stopth=param.stopth;
    end
    if isfield(param,'displayevolution')
        displayevolution=param.displayevolution;
    end
end

D=size(X,2);
N=size(X,1) ; assert(N==length(Y));


flogistic=@(x)(1./(1+exp(-x)));
X=[X ones(N,1)];
Y=Y==1 ;
Idl=eye(D+1)*lambda;Idl(end)=0;
w=zeros(D+1,1);
for i=1:maxit
    wprev=w;
    ypred = flogistic(X*w) ;
    
    %     R=diag(ypred.*(1-ypred));
    %     H=(X'*R*X)+Idl;
    %     G=X'*(ypred-Y)+[lambda*w(1:end-1) ; 0];
    %     w = w-H\G;
    
    w = w-((X'*diag(ypred.*(1-ypred))*X)+Idl)\(X'*(ypred-Y)+[lambda*w(1:end-1) ; 0]);
    
    if displayevolution
        fprintf('it: %d -- diffnorm=%e\n',i,norm(w-wprev))
    end
    
    if sum(isnan(w)+isinf(w))>0
        fprintf('Non-convergence for lambda=%.9f\n',lambda);
        break ;
    elseif norm(w-wprev) < stopth
        break ;
    end
end

b=w(end);
w=w(1:end-1);


% 2-D visualization
% fieldsize=-20:0.1:20;
% [xx,yy] = meshgrid(fieldsize,fieldsize);
% val = flogistic([xx(:) yy(:)]*w+b) ; val = reshape(val,length(fieldsize),length(fieldsize)) ;
% contour(xx,yy,val,0.5);


end

