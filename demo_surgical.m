% this is the demo for relative hmm on surgical video data
clear;
clc;
source_file='C:\Users\qzhang53\Documents\MATLAB\relative_hmm\HDPHMMData_Surgical_Video_Student.mat';
load(source_file, 'data_struct', 'video_list', 'clip_group', 'clip_label', 'video_label');
% convert struct to cell
num_data=length(data_struct);
data=cell(1, num_data);
% we may need to normalize the data
for i=1: num_data
    data{i}=data_struct(i).obs;%./repmat(eps+sum(data_struct(i).obs), size(data_struct(i).obs, 1), 1);
end
% build the relative information
clip_video=zeros(1, num_data);
tmp=cumsum(clip_group);
video_label2=video_label;
clip_video(1: clip_group(1))=1;
for i=2: length(clip_group)
    clip_video(tmp(i-1)+1: tmp(i))=i;
    video_label2(i)=all(clip_label(clip_video==i)==1) | all(clip_label(clip_video==i)==2);
end
clip_good=find(video_label2(clip_video)==1 & video_label(clip_video)==1 & clip_label==1);
% use only parts of it
clip_good=clip_good(rand(size(clip_good))>.5);
clip_bad=find(video_label2(clip_video)==1 & video_label(clip_video)==2 & clip_label==2);
clip_bad=clip_bad(rand(size(clip_bad))>.7);
relative_set=zeros(num_data);
for i=clip_good
    relative_set(i, clip_bad)=1;
end
relative_set=sparse(relative_set);
% check the ratio
fprintf(1, 'Number of relative information: %f\n', full(sum(relative_set(:)))/numel(relative_set));
clear i;
save('demo_student_data');
% perform the algorithm
model=relative_hmm(data, relative_set, 20);
