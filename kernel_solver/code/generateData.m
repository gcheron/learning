
fileName = sprintf('Ntrain%d-Ntest%d-p%d.mat' , Ntrain, Ntest, p);

if exist(fileName, 'file')
    load(fileName); %we use the split that already exist for these settings
    return;
end

muClass1 = zeros(p,1) ;
muClass2 = randn(p,1) / log(p);

%%
% Random covariance matrix are generated using Wishart distribution
covClass1 = wishrnd(eye(p),round(p/2));
covClass2 = wishrnd(eye(p),round(p/2));
% covClass1 = eye(p)*4;
% covClass2 = eye(p);

yTrain = [ones(Ntrain/2,1); -ones(Ntrain/2,1)];
xTrainClass1 = mvnrnd(repmat(muClass1',Ntrain/2,1),covClass1);
xTrainClass2 = mvnrnd(repmat(muClass2',Ntrain/2,1),covClass2);
xTrain = [[xTrainClass1 ; xTrainClass2]];
%xTrain = [[xTrainClass1 ; xTrainClass2] ones(Ntrain,1)];

yTest = [ones(Ntest/2,1); -ones(Ntest/2,1)];
xTestClass1 = mvnrnd(repmat(muClass1',Ntest/2,1),covClass1);
xTestClass2 = mvnrnd(repmat(muClass2',Ntest/2,1),covClass2);
xTest = [[xTestClass1 ; xTestClass2] ];
%xTest = [[xTestClass1 ; xTestClass2] ones(Ntest,1)];