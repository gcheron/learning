function [w,b]=train_lg_SGD(X,Y,param)

%-----default parameters-----
maxit=50000;
lr=1e-3;
batch_size=100;
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
    if isfield(param,'batch_size')
        batch_size=param.batch_size;
    end
    if isfield(param,'lr')
        displayevolution=param.lr;
    end
end

D=size(X,2);
N=size(X,1) ; assert(N==length(Y));


flogistic=@(x)(1./(1+exp(-x)));

rperm = randperm(N);
X=X(rperm,:);
Y=Y(rperm);

X=[X ones(N,1)];
Y=Y==1 ;
w=zeros(D+1,1);
b_ids = 1:batch_size:N ; % batch ids
nb_batches=length(b_ids);

for i=1:maxit
    if i > nb_batches
        ii=1;
    else
        ii=i;
    end
    c_ids = b_ids(ii):min(b_ids(ii) + batch_size-1,N);
    Xb = X(c_ids,:);
    Yb = Y(c_ids);
    wprev=w;
    ypred = flogistic(Xb*w) ;
       
    G=Xb'*(ypred-Yb)+[lambda*w(1:end-1) ; 0];
    w = w-lr/(batch_size)*G;
    
    
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
fprintf('Stop after %d iterations\n',i);


b=w(end);
w=w(1:end-1);


% 2-D visualization
% fieldsize=-20:0.1:20;
% [xx,yy] = meshgrid(fieldsize,fieldsize);
% val = flogistic([xx(:) yy(:)]*w+b) ; val = reshape(val,length(fieldsize),length(fieldsize)) ;
% contour(xx,yy,val,0.5);


end

