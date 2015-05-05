% Two gaussian
X1 = bsxfun(@times,randn(1000,2),[5 2]);
X1 = bsxfun(@plus,X1,[1 -1]);
X2 = bsxfun(@times,randn(1000,2),[2.5 4.5]);
X2 = bsxfun(@plus,X2,[1 10]);




scatter(X2(:,1),X2(:,2),'r')
hold on
scatter(X1(:,1),X1(:,2),'b')
axis equal

x = [X1;X2] ;
y = [ones(size(X1,1),1) ; -ones(size(X2,1),1)] ;


c=1000; w0 = zeros(size(x,2),1);
problem=logistic_regression_primal(x,y);
w_star= damped_newton_method(problem,c,w0);

flogistic=@(x)(1./(1+exp(-x)));
fieldsize=-20:0.1:20;
[xx,yy] = meshgrid(fieldsize,fieldsize);
val = flogistic([xx(:) yy(:)]*w_star+b) ; val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5] );