% this file performs simulations for relative hmm
clear;
clc;
%% ========================================================================
% parameters and constants
num_model=6;
option=cell(1, num_model);
data_struct=cell(1, num_model);
num_dim=10;
num_state=5;
num_frame=100;
num_sequence=300;
num_train=50;
num_test=num_sequence-num_train;
root_dir='C:\Users\qzhang53\Documents\MATLAB\relative_hmm\simulation\';
cd(root_dir);
%% ========================================================================
% generate data
for i=1: num_model
    [params settings]=generateParams(num_dim, num_state, 1, 0, 0, root_dir);
    T=round(num_frame*(.8+.4*rand(1, num_sequence)));
    if i==1
        [data_struct{i} ans option{i}]=generateData(params, settings, T);
    else
        [data_struct{i} ans option{i}]=generateData(params, settings, T, option{1}.theta);
    end
    % mkdir(num2str(i, 'model%d'));
    movefile('*.tif', num2str(i, 'model%d\\'));
end
data_train=cell(1, num_model*num_train);
data_test=cell(1, num_model*num_test);
k=1;
kk=1;
for i=1: num_model
    for j=1: num_train
        data_train{k}=data_struct{i}(j).obs;
        k=k+1;
    end
    for j=1: num_test
        data_test{kk}=data_struct{i}(num_train+j).obs;
        kk=kk+1;
    end
end
% generate relative information
relative_set=zeros(num_model*num_train);
% relative_order=[1 2 5 3 6 4];
relative_order=[2 5 1 6 3 4];
for i=1: num_model-1
    relative_set(1+(relative_order(i)-1)*num_train: relative_order(i)*num_train,...
        (relative_order(i+1)-1)*num_train+1: relative_order(i+1)*num_train)=1;
end
relative_set(rand(size(relative_set))<0.9)=0;
fprintf('Number of constraint:%d\n', sum(relative_set(:)));
imshow(relative_set); title('Pair-wise order information is provided as white pixels');
clear i j k kk;
save('simulation_data');
return;
%% ========================================================================
% start
[model idx]=relative_hmm(data_train, relative_set, 10);
prior=log(model(idx).pi_init+eps);
transmat=log(model(idx).pi+eps);
obsmat=log(model(idx).theta+eps);
save('simulation_data_result', 'model', 'relative_set');
%% ========================================================================
% test
train_path=cell(size(data_train));
train_loglik=zeros(size(data_train));
train_length=zeros(size(data_train));
for i=1: length(data_train)
    [train_path{i} train_loglik(i)]=viterbi_path(prior, transmat, obsmat*data_train{i});
    train_length(i)=length(train_path{i});
end
train_result=zeros(num_model*num_train);
for j=1: num_model
    for i=1+(relative_order(j)-1)*num_train: relative_order(j)*num_train
        for k=j+1: num_model
            train_result(i, 1+(relative_order(k)-1)*num_train: relative_order(k)*num_train)=train_loglik(i)>=train_loglik(1+(relative_order(k)-1)*num_train: relative_order(k)*num_train);
        end
    end
end
train_truth=zeros(size(train_result));
for i=1: num_model-1
    for j=i+1: num_model
        test_truth(1+(relative_order(i)-1)*num_train: relative_order(i)*num_train,...
            1+(relative_order(j)-1)*num_train: relative_order(j)*num_train)=1;
    end
end
save('simulation_data_result', 'train_path', 'train_loglik', 'train_result', '-append');
% =========================================================================
test_path=cell(size(data_train));
test_loglik=zeros(size(data_test));
test_length=zeros(size(data_test));
for i=1: length(data_test)
    [test_path{i} test_loglik(i)]=viterbi_path(prior, transmat, obsmat*data_test{i});
    test_length(i)=length(test_path{i});
end
test_result=zeros(num_model*num_test);
for j=1: num_model
    for i=1+(relative_order(j)-1)*num_test: relative_order(j)*num_test
        for k=j+1: num_model
            test_result(i, 1+(relative_order(k)-1)*num_test: relative_order(k)*num_test)=test_loglik(i)>=test_loglik(1+(relative_order(k)-1)*num_test: relative_order(k)*num_test);
        end
    end
end
test_truth=zeros(size(test_result));
for i=1: num_model-1
    for j=i+1: num_model
        test_truth(1+(relative_order(i)-1)*num_test: relative_order(i)*num_test,...
            1+(relative_order(j)-1)*num_test: relative_order(j)*num_test)=1;
    end
end
save('simulation_data_result', 'test_path', 'test_loglik', 'test_result', '-append');
% =========================================================================
% reorder the model for better visualization
relative_set2=zeros(size(relative_set));
train_result2=zeros(size(train_result));
test_result2=zeros(size(test_result));
for i=1: num_model
    for j=1: num_model
        relative_set2((i-1)*num_train+1: i*num_train, (j-1)*num_train+1: j*num_train)=...
            relative_set(1+(relative_order(i)-1)*num_train: relative_order(i)*num_train,...
            1+(relative_order(j)-1)*num_train: relative_order(j)*num_train);
        train_result2((i-1)*num_train+1: i*num_train, (j-1)*num_train+1: j*num_train)=...
            train_result(1+(relative_order(i)-1)*num_train: relative_order(i)*num_train,...
            1+(relative_order(j)-1)*num_train: relative_order(j)*num_train);
        test_result2((i-1)*num_test+1: i*num_test, (j-1)*num_test+1: j*num_test)=...
            test_result(1+(relative_order(i)-1)*num_test: relative_order(i)*num_test,...
            1+(relative_order(j)-1)*num_test: relative_order(j)*num_test);
    end
end
relative_set=relative_set2;
train_result=train_result2;
test_result=test_result2;
clear relative_set2 train_result2 test_result2;
%% ========================================================================
% start
[model_list model_idx model_score]=relative_hmm_two(data_train, relative_set, 10);
model=model_list(:, model_idx);
prior=[log(model(1).pi_init+eps), log(model(2).pi_init+eps)];
transmat=cat(3, log(model(1).pi+eps), log(model(2).pi+eps));
obsmat=cat(3, log(model(1).theta+eps), log(model(2).theta+eps));
% save('simulation_data_result', 'model', 'relative_set');
%% ========================================================================
% test
train_path=cell(2, length(data_train));
train_loglik=zeros(2, length(data_train));
train_length=zeros(size(data_train));
for j=1: 2
    for i=1: length(data_train)
        [train_path{j, i} train_loglik(j, i)]=viterbi_path(prior(:, j), transmat(:, :, j), obsmat(:, :, j)*data_train{i});
        train_length(i)=length(train_path{j, i});
    end
end
train_loglik=train_loglik(1, :)-train_loglik(2, :);
train_result=zeros(num_model*num_train);
for j=1: num_model
    for i=1+(relative_order(j)-1)*num_train: relative_order(j)*num_train
        for k=j+1: num_model
            train_result(i, 1+(relative_order(k)-1)*num_train: relative_order(k)*num_train)=train_loglik(i)>=train_loglik(1+(relative_order(k)-1)*num_train: relative_order(k)*num_train);
        end
    end
end
train_truth=zeros(size(train_result));
for i=1: num_model-1
    for j=i+1: num_model
        test_truth(1+(relative_order(i)-1)*num_train: relative_order(i)*num_train,...
            1+(relative_order(j)-1)*num_train: relative_order(j)*num_train)=1;
    end
end
% save('simulation_data_result', 'train_path', 'train_loglik', 'train_result', '-append');
% =========================================================================
test_path=cell(2, length(data_test));
test_loglik=zeros(2, length(data_test));
test_length=zeros(size(data_test));
for j=1: 2
    for i=1: length(data_test)
        [test_path{j, i} test_loglik(j, i)]=viterbi_path(prior(:, j), transmat(:, :, j), obsmat(:, :, j)*data_test{i});
        test_length(i)=length(test_path{j, i});
    end
end
test_loglik=test_loglik(1, :)-test_loglik(2, :);
test_result=zeros(num_model*num_test);
for j=1: num_model
    for i=1+(relative_order(j)-1)*num_test: relative_order(j)*num_test
        for k=j+1: num_model
            test_result(i, 1+(relative_order(k)-1)*num_test: relative_order(k)*num_test)=test_loglik(i)>=test_loglik(1+(relative_order(k)-1)*num_test: relative_order(k)*num_test);
        end
    end
end
test_truth=zeros(size(test_result));
for i=1: num_model-1
    for j=i+1: num_model
        test_truth(1+(relative_order(i)-1)*num_test: relative_order(i)*num_test,...
            1+(relative_order(j)-1)*num_test: relative_order(j)*num_test)=1;
    end
end
% save('simulation_data_result', 'test_path', 'test_loglik', 'test_result', '-append');
% =========================================================================
% reorder the model for better visualization
relative_set2=zeros(size(relative_set));
train_result2=zeros(size(train_result));
test_result2=zeros(size(test_result));
train_loglik2=train_loglik;
test_loglik2=test_loglik;
for i=1: num_model
    train_loglik2((i-1)*num_train+1: i*num_train)=train_loglik(1+(relative_order(i)-1)*num_train: relative_order(i)*num_train);
    test_loglik2((i-1)*num_test+1: i*num_test)=test_loglik(1+(relative_order(i)-1)*num_test: relative_order(i)*num_test);
    for j=1: num_model
        relative_set2((i-1)*num_train+1: i*num_train, (j-1)*num_train+1: j*num_train)=...
            relative_set(1+(relative_order(i)-1)*num_train: relative_order(i)*num_train,...
            1+(relative_order(j)-1)*num_train: relative_order(j)*num_train);
        train_result2((i-1)*num_train+1: i*num_train, (j-1)*num_train+1: j*num_train)=...
            train_result(1+(relative_order(i)-1)*num_train: relative_order(i)*num_train,...
            1+(relative_order(j)-1)*num_train: relative_order(j)*num_train);
        test_result2((i-1)*num_test+1: i*num_test, (j-1)*num_test+1: j*num_test)=...
            test_result(1+(relative_order(i)-1)*num_test: relative_order(i)*num_test,...
            1+(relative_order(j)-1)*num_test: relative_order(j)*num_test);
    end
end
train_truth=zeros(size(train_result));
test_truth=zeros(size(test_result));
for i=1: num_model-1
    train_truth((i-1)*num_train+1: i*num_train, i*num_train+1: end)=1;
    test_truth((i-1)*num_test+1: i*num_test, i*num_test+1: end)=1;
end
relative_set=relative_set2;
train_result=train_result2;
test_result=test_result2;
clear relative_set2 train_result2 test_result2;
%% ========================================================================
h=figure;
h1=axes('position', [0 0 0.5 0.45], 'parent', h);
imshow(train_result2, 'parent', h1); title('Result for Training Data');
h2=axes('position', [0.5 0 0.5 0.45], 'parent', h);
imshow(test_result2, 'parent', h2); title('Result for Testing Data');
h3=axes('position', [0 0.5 0.5 0.45], 'parent', h);
imshow(relative_set2, 'parent', h3); title('Prior Information');
h4=axes('position', [0.5 0.5 0.5 0.45], 'parent', h);
imshow(train_truth, 'parent', h4); title('Prior Information');

h=figure;
h1=axes('position', [0.05 0.05 0.9 0.35], 'parent', h);
plot(h1, train_loglik2); title('ln(p(x,z|\theta_1)/p(x,z|\theta_2)) of Training Data');
hold(h1, 'on');
for i=1: num_model-1
    plot(h1, [i*num_train+0.5, i*num_train+0.5], [min(train_loglik2) max(train_loglik2)], 'r');
end
hold(h1, 'off');
h2=axes('position', [0.05 0.55 0.9 0.35], 'parent', h);
plot(h2, test_loglik2); title('ln(p(x,z|\theta_1)/p(x,z|\theta_2)) of Testing Data');
hold(h2, 'on');
for i=1: num_model-1
    plot(h2, [i*num_test+0.5, i*num_test+0.5], [min(test_loglik2) max(test_loglik2)], 'r');
end
hold(h2, 'off');
