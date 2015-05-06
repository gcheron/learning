function [w,b]=train_lg_newton_raphson_multiclass(X,Y,param)

assert(size(Y,1)==size(X,1));


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
nb_classes=size(Y,2);

flogistic=@(x)(1./(1+exp(-x)));
X=[X ones(N,1)];
Y=Y==1 ;
Idl=eye(D+1)*lambda;Idl(end)=0;
w=zeros((D+1)*nb_classes,1);
H=zeros((D+1)*nb_classes,(D+1)*nb_classes);
g=zeros((D+1)*nb_classes,1);
Ic=eye(nb_classes);
for i=1:maxit
    wprev=w;
    
    w_c=reshape(w,[D+1 nb_classes]);
    
    num = exp(X*w_c);
    den = sum(num,2) ;
    y_pred = bsxfun(@times,num,1./den) ;
    
    for j = 1:nb_classes
        id_j = 1+(j-1)*(D+1):j*(D+1) ;
        
        w_j = w_c(:,j);
        y_j = Y(:,j);
        y_pred_j = y_pred(:,j);
        
        
        g(id_j)=X'*(y_pred_j-y_j)+[lambda*w_j(1:end-1) ; 0];
         
          
          
        for k = 1:nb_classes
            id_k = 1+(k-1)*(D+1):k*(D+1) ;
            H(id_j,id_k)=compute_hess_k_j(k,j,N,X,y_pred,Ic,Idl);
            
            %R=diag(y_pred(:,k).*(1-y_pred(:,j)));
            %H2(id_j,id_k) = (X'*R*X)+Idl ;
        end
        
        
        
        
    end
    
    
    w = w-H\g;
    
    if displayevolution
        fprintf('it: %d -- diffnorm=%e\n',i,norm(w(:)-wprev(:)))
    end
    
    if sum(isnan(w(:))+isinf(w(:)))>0
        fprintf('Non-convergence for lambda=%.9f\n',lambda);
        break ;
    elseif norm(w(:)-wprev(:)) < stopth
        break ;
    end
end

w_c=reshape(w,[D+1 nb_classes]);
b=w_c(end,:);
w=w_c(1:end-1,:);


% check Hessian
function H_j_k=compute_hess_k_j(k,j,N,X,y_pred,Ic,Idl)
H_j_k=Idl;
for i = 1:N
    H_j_k = H_j_k + y_pred(i,k).*(Ic(k,j)-y_pred(i,j))*X(i,:)'*X(i,:);
end


% 2-D visualization
% fieldsize=-20:0.1:20;
% [xx,yy] = meshgrid(fieldsize,fieldsize);
% val = flogistic([xx(:) yy(:)]*w+b) ; val = reshape(val,length(fieldsize),length(fieldsize)) ;
% contour(xx,yy,val,0.5);





