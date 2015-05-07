function  [ind_train,ind_val,training_set_labels,validation_set_labels] = ...
    split_data_idx_rand(labels,trainp,min_wanted)

if nargin < 3
    min_wanted = 0 ;
end

nbsamples = length(labels) ;

nbneg=sum(labels~=1) ;
nbpos=sum(labels==1) ;

assert(2*min_wanted<=nbneg && 2*min_wanted<=nbpos);

nbtrain = round(nbsamples*trainp) ;

trainposnum = -1 ; trainegnum = -1 ; validposnum = -1 ; validnegnum = -1 ;

while (trainposnum < min_wanted || trainegnum < min_wanted || validposnum < min_wanted || validnegnum < min_wanted)
    perm = randperm(nbsamples) ;
    
    ind_train = perm(1:nbtrain) ;
    ind_val = perm(nbtrain+1:end) ;
    
    training_set_labels = labels(ind_train);
    validation_set_labels = labels(ind_val);
    
    trainposnum = sum(training_set_labels==1) ;
    trainegnum = sum(training_set_labels~=1) ;
    validposnum = sum(validation_set_labels==1) ;
    validnegnum = sum(validation_set_labels~=1) ;
end
end
