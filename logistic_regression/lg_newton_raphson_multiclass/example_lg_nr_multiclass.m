% 4 gaussians

rng(0)

X1 = bsxfun(@times,randn(1000,2),[5 2]);
X1 = bsxfun(@plus,X1,[1 -1]);
X2 = bsxfun(@times,randn(1000,2),[2.5 4.5]);
X2 = bsxfun(@plus,X2,[1 10]);
X3 = bsxfun(@times,randn(1000,2),[5 2]);
X3 = bsxfun(@plus,X3,[10 7]);
X4 = bsxfun(@times,randn(1000,2),[2.5 2]);
X4 = bsxfun(@plus,X4,[-4 5]);


X = [X1;X2;X3;X4] ;
Y = [[ones(size(X1,1),1) ; zeros(size(X2,1)+size(X3,1)+size(X4,1),1)] ...
    [zeros(size(X1,1),1); ones(size(X2,1),1) ; zeros(size(X3,1)+size(X4,1),1)] ...
    [zeros(size(X1,1)+size(X2,1),1); ones(size(X3,1),1) ; zeros(size(X4,1),1)] ...
    [zeros(size(X1,1)+size(X2,1)+size(X3,1),1); ones(size(X4,1),1)] ...
    ] ;

param.lambda=1000;
param.maxit=5;
param.displayevolution=1;
[w,b]=train_lg_newton_raphson_multiclass(X,Y,param);

w_c=[w;b];

fieldsize=-30:0.1:30;
[xx,yy] = meshgrid(fieldsize,fieldsize);
num = exp([xx(:) yy(:) ones(length(yy(:)),1)]*w_c);
den = sum(num,2) ;
val = bsxfun(@times,num,1./den) ;
[~,classprior] = max(val,[],2); 
classprior = reshape(classprior,length(fieldsize),length(fieldsize)) ;
contourf(xx,yy,classprior,[1 1]);
hold on
contourf(xx,yy,classprior,[2 2]);
contourf(xx,yy,classprior,[3 3]);
contourf(xx,yy,classprior,[4 4]);
map = [0, 0, 0.5 ;...
    0.5, 0, 0; ...
    0, 0.5, 0 ;...
    0.5, 0.5, 0];
colormap(map)

scatter(X1(:,1),X1(:,2),'b')

scatter(X2(:,1),X2(:,2),'r')
scatter(X3(:,1),X3(:,2),'g')
scatter(X4(:,1),X4(:,2),'y')
axis equal
