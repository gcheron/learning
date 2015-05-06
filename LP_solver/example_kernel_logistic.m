% Two gaussian

rng(0)

X1 = bsxfun(@times,randn(1000,2),[5 2]);
X1 = bsxfun(@plus,X1,[1 2]);
X2 = bsxfun(@times,randn(1000,2),[2.5 3.5]);
X2 = bsxfun(@plus,X2,[1 10]);
X3 = bsxfun(@times,randn(200,2),[5 2]);
X3 = bsxfun(@plus,X3,[1 2]);
X4 = bsxfun(@times,randn(200,2),[2.5 3.5]);
X4 = bsxfun(@plus,X4,[1 10]);

scatter(X3(:,1),X3(:,2),'bx')
hold on
scatter(X4(:,1),X4(:,2),'rx')
axis equal
set(gcf,'renderer','zbuffer');


xTrain = [X1;X2] ;
xTrain = [xTrain ones(size(xTrain,1),1)] ;
yTrain = [ones(size(X1,1),1) ; -ones(size(X2,1),1)] ;

xTest = [X3;X4] ;
xTest = [xTest ones(size(xTest,1),1)] ;
yTest = [ones(size(X3,1),1) ; -ones(size(X4,1),1)] ;


fieldsize=-20:0.1:20;
[xx,yy] = meshgrid(fieldsize,fieldsize);
xGrid = [xx(:) yy(:) ones(length(yy(:)),1)];

flogistic=@(x)(1./(1+exp(-x)));

% PB1 (primal same as logistic Newton-Raphson folder)
c=0.001; w0=zeros(size(xTrain,2),1) ;
PB1=LP_primal_logistic_reg_problem(xTrain,yTrain);
tic;
w_pb1= LP_damped_newton_method(PB1,c,w0);
train_time_primal = toc;
tic;
w_pb1_nr= LP_newton_raphson(PB1,c,w0);
train_time_primal_nr = toc;

val = flogistic(xGrid*w_pb1) ; val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5],'r','LineWidth',2);
val = flogistic(xGrid*w_pb1_nr) ; val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5],'--r','LineWidth',4);

% PB2: dual (Kernel logistic regression)
% linear kernel
c=0.001; alpha0=zeros(size(xTrain,1),1) ;

K = xTrain*xTrain' ;
PB2=LP_dual_logistic_reg_problem(K,yTrain);

tic;
alpha_pb2= LP_damped_newton_method(PB2,c,alpha0);
train_time_dual_lin = toc;
tic;
alpha_pb2_nr= LP_newton_raphson(PB2,c,alpha0);
train_time_dual_lin_nr = toc;

Kgrid=xGrid*xTrain';
sumProduct_grid = zeros(size(xGrid,1),1);
for i=1:size(xGrid,1)
    sumProduct_grid(i) =  Kgrid(i,:) * alpha_pb2 ;
end
Ktest_lin=xTest*xTrain';
sumProduct_lin = zeros(size(xTest,1),1);
for i=1:size(xTest,1)
    sumProduct_lin(i) =  Ktest_lin(i,:) * alpha_pb2 ;
end

sumProduct_grid_nr = zeros(size(xGrid,1),1);
for i=1:size(xGrid,1)
    sumProduct_grid_nr(i) =  Kgrid(i,:) * alpha_pb2_nr ;
end
sumProduct_lin_nr = zeros(size(xTest,1),1);
for i=1:size(xTest,1)
    sumProduct_lin_nr(i) =  Ktest_lin(i,:) * alpha_pb2_nr ;
end


val = flogistic(sumProduct_grid) ; val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5],'b' );
val = flogistic(sumProduct_grid_nr) ; val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5],'--b','LineWidth',2 );


% PB3: dual (Kernel logistic regression)
% RBF kernel
c=0.001; alpha0=zeros(size(xTrain,1),1) ;
sigma = 0.001;
K = zeros(size(K)) ;
for i=1:size(K,1)
    u=xTrain(i,:);
    K(i,:) = exp(-sigma*(sum(bsxfun(@minus,xTrain,u).^2,2))) ;
end

PB3=LP_dual_logistic_reg_problem(K,yTrain);

tic;
alpha_pb3= LP_damped_newton_method(PB3,c,alpha0);
train_time_dual_rbf = toc;
tic;
alpha_pb3_nr= LP_newton_raphson(PB3,c,alpha0);
train_time_dual_rbf_nr = toc;


Kgrid=zeros(size(xGrid,1),size(xTrain,1));
for i=1:size(xGrid,1)
    u=xGrid(i,:);
    Kgrid(i,:) = exp(-sigma*(sum(bsxfun(@minus,xTrain,u).^2,2))) ;
end
Ktest_rbf=zeros(size(xTest,1),size(xTrain,1));
for i=1:size(xTest,1)
    u=xTest(i,:);
    Ktest_rbf(i,:) = exp(-sigma*(sum(bsxfun(@minus,xTrain,u).^2,2))) ;
end


sumProduct_grid = zeros(size(xGrid,1),1);
for i=1:size(xGrid,1)
    sumProduct_grid(i) =  Kgrid(i,:) * alpha_pb3 ;
end
sumProduct_rbf = zeros(size(xTest,1),1);
for i=1:size(xTest,1)
    sumProduct_rbf(i) =  Ktest_rbf(i,:) * alpha_pb3 ;
end
sumProduct_grid_nr = zeros(size(xGrid,1),1);
for i=1:size(xGrid,1)
    sumProduct_grid_nr(i) =  Kgrid(i,:) * alpha_pb3_nr ;
end
sumProduct_rbf_nr = zeros(size(xTest,1),1);
for i=1:size(xTest,1)
    sumProduct_rbf_nr(i) =  Ktest_rbf(i,:) * alpha_pb3_nr ;
end


val = flogistic(sumProduct_grid) ; val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5],'g' );
val = flogistic(sumProduct_grid_nr) ; val = reshape(val,length(fieldsize),length(fieldsize)) ;
contour(xx,yy,val,[0.5 0.5],'--g','LineWidth',2 );

% error damped newton
sc_primal=flogistic(xTest*w_pb1);
sc_dual_lin=flogistic(sumProduct_lin);
sc_dual_rbf=flogistic(sumProduct_rbf);

error_primal = sum((sc_primal < 0.5) == (yTest==1))
error_dual_lin = sum((sc_dual_lin < 0.5) == (yTest==1))
error_dual_rbf = sum((sc_dual_rbf < 0.5) == (yTest==1))

train_time_primal
train_time_dual_lin
train_time_dual_rbf


% error newton raphson
sc_primal_nr=flogistic(xTest*w_pb1_nr);
sc_dual_lin_nr=flogistic(sumProduct_lin_nr);
sc_dual_rbf_nr=flogistic(sumProduct_rbf_nr);

error_primal_nr = sum((sc_primal_nr < 0.5) == (yTest==1))
error_dual_lin_nr = sum((sc_dual_lin_nr < 0.5) == (yTest==1))
error_dual_rbf_nr = sum((sc_dual_rbf_nr < 0.5) == (yTest==1))

train_time_primal_nr
train_time_dual_lin_nr
train_time_dual_rbf_nr


