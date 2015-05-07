function folds = split_data_idx(labels,K)

pos = find(labels==1);
neg = find(labels~=1);

rng(3); % set seed

randpos = randperm(length(pos));
randneg = randperm(length(neg));

posp = pos(randpos);
negp = neg(randneg);

posperfold = floor(length(posp)/K);
negperfold = floor(length(negp)/K);

train_sets = arrayfun(@(x) [ posp( (x-1)*posperfold + 1 : x*posperfold ) ; negp( (x-1)*negperfold + 1 : x*negperfold ) ] , 1:K , 'UniformOutput' , false );

nsamples=length(labels);
folds={};
for k=1:K
    ind_val=train_sets{k};
    ind_train=setdiff([1:nsamples]',ind_val);
    labels_val = labels(ind_val);
    labels_train = labels(ind_train);
    
    folds.ind_val{k}=ind_val;
    folds.ind_train{k}=ind_train;
    folds.labels_val{k}=labels_val;
    folds.labels_train{k}=labels_train;
end


end

