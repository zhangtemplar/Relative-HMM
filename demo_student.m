% this is the demo for relative hmm on surgical video data
clear;
clc;
root_dir='C:\Users\qzhang53\Documents\MATLAB\relative_hmm';
source_dir='C:\Users\qzhang53\Documents\MATLAB\Dataset\surgical_video_student\';
cd(source_dir);
% feature extraction
extract_feature=0;
% split fold
split_fold=0;
number_cluster=100;
pca_dim=0;
normalize_hist=1;
related_point=0;
feature_type='sparse_hog';
coordinate_type='sparse_coordinate';
% hmm related
num_state=10;
% number of training pairs
num_pair=0.8;
%% ========================================================================
if extract_feature || ~exist('student_data.mat', 'file')
    disp('feature is extracted');
    % generate histogram
    [video_data video_name video_label action_size]= dense_load_feature_mat(...
        source_dir, '*_descriptor.mat', {feature_type coordinate_type}, 0.01, 1);
    % build the clustering and pca
    % pca_dim=30; sigma=100; 
    [ans C coef latent]=dense_build_codebook(video_data{1},...
        number_cluster, pca_dim, 50);
    clear ans video_data;
    % do histogram for each temporal histogram
    % not use related points but normalize the histogram
    video_histogram=loadFeatureMat2TemporalHistogram(source_dir, '*_descriptor.mat',...
        {feature_type coordinate_type}, [], coef, C, related_point, normalize_hist, 1);
    save('student_data', 'video_histogram', 'video_name', 'video_label',...
        'action_size', 'C', 'coef', 'latent');
else
    disp('feature is loaded');
    load('student_data');
end
%% ========================================================================
% dense the histogram
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
% relative information
if split_fold || ~exist('student_train.mat', 'file')
    video_label=false(size(data));
    % for video label, first day and last day are used for training. first day
    % is always worse than the last day
    first_week={[1 2 3] [39 40 41] [74 75 76] [110 111 112] [137 138 139] [167 168 169],...
        [200 201 202] [231 232 233] [267 268 269] [302 303 304] [338 339 340] [362 363 364],...
        [402 403] [434 435] [468 469 470] [501 502 503]};
    last_week={[36,37,38],[71,72,73],[107,108,109],[134,135,136],[166],[197,198,199],...
        [228,229,230],[264,265,266],[299,300,301],[335,336,337],[359,360,361],...
        [399,400,401],[431,432,433],[465,466,467],[498,499,500],[544,545,546]};
    video_label(cell2mat(first_week))=1;
    video_label(cell2mat(last_week))=1;
    relative_set=false(length(data), length(data));
    % for i=1: length(first_week)
    %     relative_set(last_week{i}, first_week{i})=1;
    % end
    subject_index=unique(data_info(:, 2));
    for i=1: length(subject_index)
        % we assume the skill is improved during process
        day_index=unique(data_info(data_info(:, 2)==subject_index(i), 1));
        % we only use the last two trials for each day, i.e., first trial
        % is just a warm up
        day_size=zeros(size(day_index));
        for j=1: length(day_index)
            day_size(j)=sum(data_info(:, 2)==subject_index(i) & data_info(:, 1)==day_index(j));
        end
        day_index=day_index(day_size>1);
        % build up the relative matrix
        for j=1: length(day_index)-1
            left_index=find(data_info(:, 2)==subject_index(i) & data_info(:, 1)==day_index(j));
            right_index=find(data_info(:, 2)==subject_index(i) & data_info(:, 1)==day_index(j+1));
            relative_set(right_index, left_index)=1;
        end
    end
    relative_set(rand(size(relative_set))<num_pair)=0;
else
    load('student_train.mat', 'video_label', 'relative_set', 'subject_index', 'data_info'); 
end
%% ========================================================================
% baseline method training
%{
t1=tic;
[model_list model_idx model_score]=relative_hmm(data, relative_set, num_state);
t1=toc(t1);
% testing
model=model_list(model_idx);
prior=log(model.pi_init);
transmat=log(model.pi);
obsmat=log(model.theta);
path=cell(size(data));
log_lik=zeros(size(data));
for i=1: length(data)
    [path{i} log_lik(i)]=viterbi_path(prior, transmat, obsmat*data{i});
    data_length(i)=size(data{i}, 2);
end
tmp=log_lik./data_length;
axes('position', [0.05 0.05 0.95 0.95]);
plot(tmp);
hold on;
for i=1: length(subject_index)-1
    idx=find(data_info(:, 2)==subject_index(i));
    plot(idx(end)*ones(1,2)+0.5, [min(tmp) max(tmp)], 'r');
end
hold off;
axis([0 547 min(tmp) max(tmp)]);
save('student_result_base', 'model', 'data', 'first_week', 'last_week', 'relative_set', 'path', 'log_lik');
%}
%% ========================================================================
% training with improved model
t2=tic;
[model_list, model_idx, model_score]=relative_hmm_two(data, relative_set, num_state);
t2=toc(t2);
% testing
model=model_list(:, model_idx);
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
axes('position', [0.05 0.05 0.95 0.95]);
plot(tmp);
hold on;
for i=1: length(subject_index)-1
    idx=find(data_info(:, 2)==subject_index(i));
    plot(idx(end)*ones(1,2)+0.5, [min(tmp) max(tmp)], 'r');
end
hold off;
axis([0 547 min(tmp) max(tmp)]);
% generaet confusion matrix
test_result=zeros(action_size);
% compare within subjects
for i=1: length(subject_index)
    % we assume the skill is improved during process
    day_index=unique(data_info(data_info(:, 2)==subject_index(i), 1));
    % build up the relative matrix
    for j=2: length(day_index)
        left_index=find(data_info(:, 2)==subject_index(i) & data_info(:, 1)<day_index(j))';
        right_index=find(data_info(:, 2)==subject_index(i) & data_info(:, 1)==day_index(j))';
        for l=right_index
            test_result(l, left_index)=tmp(l)>=tmp(left_index);
        end
    end
end
num_pair=0;
for i=1: length(subject_index)
    % we assume the skill is improved during process
    day_index=unique(data_info(data_info(:, 2)==subject_index(i), 1));
    % build up the relative matrix
    for j=2: length(day_index)
        left_index=find(data_info(:, 2)==subject_index(i) & data_info(:, 1)<day_index(j))';
        right_index=find(data_info(:, 2)==subject_index(i) & data_info(:, 1)==day_index(j))';
        num_pair=num_pair+length(left_index)*length(right_index);
    end
end
fprintf(1, 'Accuracy is %d/%d=%f\n', sum(test_result(:)), num_pair, sum(test_result(:))/num_pair);
save('student_result_new', 'model', 'data', 'data_info', 'relative_set', 'path', 'log_lik');
%% ========================================================================
% plot the result
%{
for i=1: length(first_week)
     subplot(211); 
%      hold on;
%      stairs(path{1, first_week{i}(1)}, 'r'); 
     stairs(path{2, first_week{i}(1)}, 'g'); 
%      hold off;
     axis([1 length(path{2, first_week{i}(1)}) 1 10]); 
     subplot(212); 
%      hold on;
     stairs(path{1, last_week{i}(end)}, 'r'); 
%      stairs(path{2, last_week{i}(end)}, 'g'); 
%      hold off;
     axis([1 length(path{1, last_week{i}(end)}) 1 10]);
     waitforbuttonpress();
end
% for the state transition path
h=figure();
h1=axes('parent', h, 'position', [0.05 0.55 0.9 0.4]);
h2=axes('parent', h, 'position', [0.05 0.05 0.9 0.4]);
for i=1: length(path)
    stairs(h1, path{1, i}, 'linewidth', 2);
    title('Model 1', 'fontsize', 12, 'parent', h1);
    axis(h1, [1 length(path{1, i}) 0 11]);
    stairs(h2, path{2, i}, 'linewidth', 2);
    title('Model 2', 'fontsize', 12, 'parent', h2);
    axis(h2, [1 length(path{2, i}) 0 11]);
    set(h, 'position', [100 100 640 480]);
    print('-deps', '-r96', num2str(i, 'path_%03d'));
    print('-dtiff', '-r96', num2str(i, 'path_%03d'));
end
% for the model itself
h=figure();
h1=axes('parent', h, 'position', [0.05 0.05 0.4 0.9]);
h2=axes('parent', h, 'position', [0.55 0.05 0.4 0.9]);
imshow(model(1).pi, 'parent', h1);
title('Model 1', 'fontsize', 12, 'parent', h1);
colormap(h1, 'jet');
for i=1: 10
    for j=1: 10
        text(i-.5, j, num2str(model(1).pi(i, j), '%.2f'), 'fontsize', 9, 'parent', h1);
    end
end
imshow(model(2).pi, 'parent', h2);
title('Model 1', 'fontsize', 12, 'parent', h2);
colormap(h2, 'jet');
for i=1: 10
    for j=1: 10
        text(i-.5, j, num2str(model(2).pi(i, j), '%.2f'), 'fontsize', 9, 'parent', h2);
    end
end
set(h, 'position', [100 100 800 400]);
%}
