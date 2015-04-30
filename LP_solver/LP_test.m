% % Test LP
% 
% m = 400;
% n = 200;
% 
% A = randn(m,n);
% x0 = rand(n,1);
% b = A*x0 + rand(m,1) ;
% c = rand(n,1);
% 
% 
% problem=LP_problem(A,b);
% [x0,~]=LP_phase_I(A,b);
% [x_star,f_star] = LP_barrier_solver(problem,c,x0);
% 
% 
% 
% p_s = c'*x_star ;
% 
% % CVX comparison
% cvx_begin
% variable x(n,1)
% minimize (c'*x)
% subject to
% A*x<=b
% cvx_end
% 
% fprintf('\n\nMy  p*=%0.9f\n',p_s);
% fprintf('CVX p*=%0.9f\n',cvx_optval);

% Test SVM

n_dim = 2 ; %100;
n = n_dim+1;
m = 500;

rng(0)

% separable example
x = [randn(n_dim,m/2) randn(n_dim,m/2)-5];
x = [x; ones(1,m)];
y = [ones(m/2,1); -ones(m/2,1)];

% bivariate example
muc1 = zeros(n_dim,1) ;
muc2 = randn(n_dim,1) / log(n_dim);
sigc1 = rand(1,n_dim);
sigc2 = rand(1,n_dim);
covc1 = diag(sigc1);
covc2 = diag(sigc2);
y = [ones(m/2,1); -ones(m/2,1)];
xc1 = mvnrnd(repmat(muc1',m/2,1),covc1);
xc2 = mvnrnd(repmat(muc2',m/2,1),covc2);
x = [xc1' xc2' ; ones(1,m)];


% w(1:n) = w ;  w(n+1:end) = z
w0 = [zeros(n,1);ones(m,1) * 2];w_our_init=w0;
c = 0.1 ;
a0 = c/2*ones(m,1);

problem=LP_primal_SVM_problem(x,y,n,m);
problem_dual=LP_dual_SVM_problem(x,y);

[w_star,f_star] = LP_barrier_solver(problem,c,w0);
[wd_star,d_star] = LP_barrier_solver(problem_dual,c,a0);

cvx_begin quiet
    variable w_cvx(n+m,1)
    minimize (0.5 * w_cvx(1:n)'* w_cvx(1:n) + c*sum(w_cvx(n+1:end)))
    subject to
        y.*(w_cvx(1:n)'*x)'>= 1 - w_cvx(n+1:end);
        w_cvx(n+1:end) >= 0;
cvx_end

% compare to libsvm
addpath('/sequoia/data1/gcheron/general-code/libsvm-3.18/matlab/')
parameter_string = sprintf('-s 0 -t 0 -c %.9f',c);
model = svmtrain(y,x(1:end-1,:)', parameter_string);
w_lib = [model.SVs' * model.sv_coef ; -model.rho];
if model.Label(1) == -1
    w_lib=-w_lib ;
end


fprintf('\n\nMy  p*=%0.9f\n',f_star);
fprintf('CVX p*=%0.9f\n',cvx_optval);

clf
scatter(x(1,y==1),x(2,y==1)) ; hold on
scatter(x(1,y~=1),x(2,y~=1),'r')

xmax = max(x(1:2,:),[],2) +1;
xmin = min(x(1:2,:),[],2) -1;

[xx1,xx2] = meshgrid(xmin(1):.01:xmax(1), xmin(2):.01:xmax(2));
xx=[xx1(:)'; xx2(:)'; ones(1,length(xx2(:)))];
sc=w_star(1:3)'*xx; sc=reshape(sc,[size(xx2,1) size(xx2,2)]);
contour(xx1,xx2,sc,[0 0],'g') 

sc=w_cvx(1:3)'*xx; sc=reshape(sc,[size(xx2,1) size(xx2,2)]);
contour(xx1,xx2,sc,[0 0],'r') 

sc=w_lib'*xx; sc=reshape(sc,[size(xx2,1) size(xx2,2)]);
contour(xx1,xx2,sc,[0 0],'b') 



