clear;
clc;
C=zeros(72, 100);
C_size=zeros(1, 100);
source_dir='C:\Users\qzhang53\Documents\MATLAB\Dataset\surgical_video_student\';
load('C:\Users\qzhang53\Documents\MATLAB\relative_hmm\student\student_new_result.mat')
cd(source_dir);
video_name=dir('*_descriptor.mat');
for i=1: length(video_histogram)
    % load feature
    load(video_name(i).name, 'sparse_hog', 'sparse_coordinate');
    frame_flag=find(sum(video_histogram{i})==1);
    fprintf(1, '%d valid frames founded for video %s\n', length(frame_flag), video_name(i).name);
    for t=frame_flag
        idx=find(video_histogram{i}(:, t));
        C(:, idx)=C(:, idx)+sparse_hog(:, sparse_coordinate(4, :)==t);
        C_size(idx)=C_size(idx)+1;
    end
end

for i=1: 100
    C(:, idx)=C(:, idx)/C_size(idx);
end
