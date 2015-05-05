function K=compute_kernel(X,fun)

n=size(X,1) ;
K = zeros (n,n) ;

for i = 1:n
    for j=1:n
        K(i,j) = fun(X(i,:),X(j,:));
    end
end