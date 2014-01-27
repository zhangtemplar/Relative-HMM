% this file transfer the model learned from the student dataset to the
% hospital dataset
clear;
clc;
root_dir='C:\Users\qzhang53\Documents\MATLAB\relative_hmm';
source_dir='C:\Users\qzhang53\Documents\MATLAB\Dataset\hospital\';
cd(root_dir);
dense_setting;
%% ========================================================================
% load feature
%{
cd(source_dir);
source_pattern='_c3_hog.txt';
file_list=dir(['*' source_pattern]);
for i=1: length(file_list)
    [sparse_coordinate, sparse_hog]=readSingleLaptevDescriptor(file_list(i).name, hog_dim, hog_head);
    save([file_list(i).name(1: end-length(source_pattern)) '_descriptor'], 'sparse_coordinate', 'sparse_hog');
    clear sparse_coordinate sparse_hog;
end
%}
number_cluster=100;
pca_dim=0;
normalize_hist=1;
% use related point
related_point=0;
feature_type='sparse_hog';
coordinate_type='sparse_coordinate';
load('C:\Users\qzhang53\Documents\MATLAB\Dataset\surgical_video_student\student_data.mat', 'C')
video_histogram=loadFeatureMat2TemporalHistogram(source_dir, '*_descriptor.mat',...
    {feature_type coordinate_type}, [], [], C, related_point, normalize_hist, 1);
video_label=[4,3,3,4,5,5,1,4,5,3,4,5,2,4,1,1,4,2,4,1,3,3,2,1,5,2,3,2,2,2,1,...
    1,1,1,1,1,5,5,3,3,1,1,4,4,3,3,3,1,5,1,4,2,4,3,4,3,1,1,4,2,5,2,3,2,2,5,1,4,1,5,3];
video_label=video_label(setdiff(1: length(video_label), [13 19 30 62]));
save('hospital_data', 'video_histogram', 'video_label');
%% ========================================================================
% dense the feature
frame_rate=10;
data=cell(size(video_histogram));
for i=1: length(data)
    data{i}=zeros(number_cluster, ceil(size(video_histogram{i}, 2)/frame_rate));
    for j=1: floor(size(video_histogram{i}, 2)/frame_rate)
        data{i}(:, j)=sum(video_histogram{i}(:, j*frame_rate+(1-frame_rate: 0)), 2);
    end
    if j~=size(data{i}, 2)
        data{i}(:, j+1)=sum(video_histogram{i}(:, j*frame_rate+1: end), 2);
    end
    % rescale it
    data{i}=data{i}/10;
end
%% ========================================================================
% apply the model
load('C:\Users\qzhang53\Documents\MATLAB\relative_hmm\student\student_new_result.mat', 'model')
path=cell(length(model), length(data));
log_lik=zeros(2, length(data));
for j=1: length(model)
    prior=log(model(j).pi_init+eps);
    transmat=log(model(j).pi+eps);
    obsmat=log(model(j).theta+eps);
    for i=1: length(data)
        [path{j, i} log_lik(j, i)]=viterbi_path(prior, transmat, obsmat*data{i});
    end
end
tmp=log_lik(1, :)-log_lik(2, :);
plot_style={'r+', 'go', 'b*', 'cs', 'md'};
figure;
hold on;
for i=1: 5
    idx=find(video_label==i);
    plot(i+randn(size(idx))/2, tmp(idx), plot_style{i});
end
hold off;
axis([0.5 5.5 min(tmp)-5, max(tmp+5)]);
set(gca, 'xtick', 1: 5);
xlabel('Post Graduate Year (PGY)', 'fontsize', 18);
ylabel('Score', 'fontsize', 18);
set(gca, 'position', [0.1 0.1 0.9 0.9]);

figure;
axes('position', [0.05 0.05 0.95 0.95]);
hold on;
for i=1: 5
    idx=find(video_label==i);
    plot(randn(1, length(idx)), tmp(idx), plot_style{i});
end
hold off;
