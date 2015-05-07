function [samplelist,labels]=get_sample_list(splitpath,featdirraw,nbclasses)

% count number of samples
numfil=get_filenumber(splitpath);

% open image list
split = fopen(splitpath) ;

% pre-allocate memory
samplelist=cell(numfil,1) ;
labels = zeros(numfil,nbclasses);
[sample,label] = strtok(fgetl(split));
ii=0; % number of loaded samples
while ischar(sample)
    ii=ii+1;
    samplelist{ii}=[featdirraw '/' sample '.mat'];
    labels(ii,str2num(label)) = 1 ;
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
