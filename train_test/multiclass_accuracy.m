function [result,per_sample_accuracy,per_class_accuracy]=multiclass_accuracy(scores,labels)

nb_classes = size(labels,2) ;
[~,GT_classes] = max(labels,[],2); % ground truth
[~,test_classes] = max(scores,[],2); % prediction

% per sample accuracy
TP=sum(GT_classes==test_classes) ;
per_sample_accuracy = TP/length(GT_classes) ;

% per class accuracy
class_accuracy=-ones(nb_classes,1);
for c=1:nb_classes
    c_id = (GT_classes==c);
    TPc=sum(test_classes(c_id) == c) ;
    class_accuracy(c)=TPc/sum(c_id);
end
per_class_accuracy = mean(class_accuracy);

% confusion matrix
conf_mat=-ones(nb_classes,nb_classes);
conf_unnorm=-ones(nb_classes,nb_classes);
for c=1:nb_classes
    conf_mat(c,:) = hist(test_classes(GT_classes==c),1:nb_classes) ;
    conf_unnorm(c,:) =conf_mat(c,:) ;
    conf_mat(c,:)=conf_mat(c,:)/sum(conf_mat(c,:)) ;
end

result.scores = scores;
result.GT_classes = GT_classes;
result.test_classes = test_classes ;
%result.actionnames = actionnames ;
result.conf_mat = conf_mat ;
result.conf_unnorm = conf_unnorm ;
result.per_sample_accuracy = per_sample_accuracy ;
result.per_class_accuracy = per_class_accuracy ;
result.class_accuracy = class_accuracy ;
result.TP = TP ;
