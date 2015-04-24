% Two gaussian
X1 = bsxfun(@times,randn(1000,2),[5 2]);
X1 = bsxfun(@plus,X1,[1 -1]);
X2 = bsxfun(@times,randn(1000,2),[2.5 4.5]);
X2 = bsxfun(@plus,X2,[1 10]);



scatter(X1(:,1),X1(:,2),'b')
hold on
scatter(X2(:,1),X2(:,2),'r')
axis equal

X = [X1;X2] ;
Y = [ones(size(X1,1),1) ; -ones(size(X2,1),1)] ;

[w,b]=train_lg_newton_raphson(X,Y);


flogistic=@(x)(1./(1+exp(-x)));
fieldsize=-20:0.1:20;
[xx,yy] = meshgrid(fieldsize,fieldsize);
val = flogistic([xx(:) yy(:)]*w+b) ; val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5] );