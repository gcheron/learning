function [ pctError ] = predictError2( xTest, yTest, xTrain, w, kernel_fun )

sigmoidal = @(x)(1./(1+exp(-x)));

sumProduct = zeros(size(xTest,1),1);
for i=1:size(xTest,1)
    for j=1:size(xTrain,1)
        sumProduct(i) = sumProduct(i) + w(j) * kernel_fun(xTest(i,:),xTrain(j,:));
    end
end


pctError = length(find(2*(sigmoidal(sumProduct)>.5)-1~=yTest)) / numel(yTest);
end

