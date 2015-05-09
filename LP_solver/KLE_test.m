function conf=KLE_test(alpha,kernel)
flogistic=@(x)(1./(1+exp(-x)));
conf=flogistic(kernel * alpha);