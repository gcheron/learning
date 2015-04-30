function fscores=test_logistic_regression(X,w,b)
flogistic=@(x)(1./(1+exp(-x)));
sc=X*w+b;
if sum(isnan(sc)+isinf(sc))>0
    fscores=NaN+ones(size(X,1),1);
else
    fscores  = flogistic(sc) ;
end
end

