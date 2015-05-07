function modelmulticlasssplit_from_experimentsplit(exsplitstringpath, savepath)
% e.g. :
% modelmulticlasssplit_from_experimentsplit('/sequoia/data1/gcheron/ICCV15/JHMDB/splitlists/experimentsplits/*train1.txt','/sequoia/data1/gcheron/JHMDB/splitlists/JHMDB_train1.txt')

splitlist = dir(exsplitstringpath);
numclasses = length(splitlist);
[expdir,~,~] = fileparts(exsplitstringpath);


% open exp splist
fs=zeros(numclasses,1);
numfil=0;
for i = 1 :numclasses
    splitpath=sprintf('%s/%s',expdir,splitlist(i).name);
    fs(i) = fopen (splitpath);
    if i==1
        numfil=get_filenumber(splitpath);
    else
        assert(numfil==get_filenumber(splitpath))
    end
end

% collect labels and sample names
labels = zeros(numfil,numclasses);
for i = 1 : numclasses
    if i==1
        [samplelist,lab]=getsamples(fs(i) ,numfil);
    else
        [sampl,lab]=getsamples(fs(i),numfil);
        for j=1:numfil
            assert(strcmp(samplelist(j),sampl(j))==1);
        end
    end
    labels(:,i)=lab ;
end
labels(labels ~= 1) = 0;
assert(sum(sum(labels,2)~=1)==0) % only one positive label per sample


% save new multiclass split
f=fopen(savepath,'w');
for s=1:numfil
    fprintf(f,'%s %d\n',samplelist{s},find(labels(s,:)==1));
end
fclose(f);

function [samplelist,labels]=getsamples(split,numfil)
% pre-allocate memory
samplelist=cell(numfil,1) ;
labels = zeros(numfil,1);
[sample,label] = strtok(fgetl(split));
ii=0; % number of loaded samples
while ischar(sample)
    ii=ii+1;
    samplelist{ii}=sample;
    labels(ii) = str2num(label) ;
    %fprintf('Collect Sample: %d out of %d : %s\n',ii,numfil,sample)
    [sample,label] = strtok(fgetl(split));
end
fclose(split);
assert(numfil == ii);

function numfil=get_filenumber(splitpath)
split = fopen(splitpath);
aa = fgetl(split);numfil=0;
while ischar(aa), numfil=numfil+1; aa = fgetl(split); end ;
fclose(split);



