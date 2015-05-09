rng(0)

% Two gaussian
X1 = bsxfun(@times,randn(1000,2),[5 2]);
X1 = bsxfun(@plus,X1,[1 -1]);
X2 = bsxfun(@times,randn(1000,2),[2.5 4.5]);
X2 = bsxfun(@plus,X2,[1 10]);

close all

scatter(X1(:,1),X1(:,2),'b')
hold on
scatter(X2(:,1),X2(:,2),'r')
axis equal
set(gcf,'renderer','zbuffer');

X = [X1;X2] ;
Y = [ones(size(X1,1),1) ; -ones(size(X2,1),1)] ;

fieldsize=-20:0.1:20;
[xx,yy] = meshgrid(fieldsize,fieldsize);
xGrid=[xx(:) yy(:)];


% linear Kernel

K_lin = X*X' ;
Kgrid=xGrid*X';

% without CV
svm = svmtrain_libsvm(Y,[(1:size(K_lin,1))' K_lin], '-s 0 -t 4 -c 100 -w1 1 -w-1 1 -q');
val=test_svm_kernel(Kgrid,svm);
val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5],'b');

% with cv
fun = svm_get_cv_fun();
[svm,output_info]=cv_main_c_beta(K_lin,Y,fun,'beta_range', 2.^(-1:1:1), 'cost_range',[0.01 0.1 1])
val=test_svm_kernel(Kgrid,svm);
val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5],'--b','LineWidth',2);

% RBF Kernel
sigma=0.01;
K_rbf=zeros(size(K_lin));
for i=1:size(K_rbf,1)
    u=X(i,:);
    K_rbf(i,:) = exp(-sigma*(sum(bsxfun(@minus,X,u).^2,2))) ;
end
Kgrid=zeros(size(Kgrid));
for i=1:size(xGrid,1)
    u=xGrid(i,:);
    Kgrid(i,:) = exp(-sigma*(sum(bsxfun(@minus,X,u).^2,2))) ;
end
% without CV
svm = svmtrain_libsvm(Y,[(1:size(K_rbf,1))' K_rbf], '-s 0 -t 4 -c 100 -w1 1 -w-1 1 -q');
val=test_svm_kernel(Kgrid,svm);
val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5],'r' );

% with cv
[svm,output_info]=cv_main_c_beta(K_rbf,Y,fun,'beta_range', 2.^(-1:1:1), 'cost_range',[0.01 0.1 1])
val=test_svm_kernel(Kgrid,svm);
val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5],'--r','LineWidth',2);

hold off

