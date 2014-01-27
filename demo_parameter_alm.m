% demo_parameter
clear;
clc;
load('C:\Users\qzhang53\Documents\MATLAB\relative_hmm\time\simulation_data');
f1=fopen('simulation_new_result.txt', 'w');
number_pair_train=[0.9875 0.975 0.95 0.9 0.8 0.7 0.6 0.5];% 0.4 0.3 0.2 0.1];
time_info=zeros(size(number_pair_train));
for ii=1: length(number_pair_train)
    %% ========================================================================
    % ground truth
    relative_order=[2 5 1 6 3 4];
    try
        disp('load data');
        load(['C:\Users\qzhang53\Documents\MATLAB\relative_hmm\simulation\',...
            num2str(ii, 'simulation_hmm_result_%02d')], 'relative_set');
    catch
        disp('data not found, create a new one instead');
        relative_set=zeros(num_model*num_train);
        for i=1: num_model-1
            relative_set(1+(relative_order(i)-1)*num_train: relative_order(i)*num_train,...
                (relative_order(i+1)-1)*num_train+1: relative_order(i+1)*num_train)=1;
        end
        relative_set(rand(size(relative_set))<number_pair_train(ii))=0;
    end
    fprintf(f1, 'Experiment %d\tPrior %d\t', ii, sum(relative_set(:)));
    %% ========================================================================
    % train
    t=tic;
    [model_list model_idx model_score]=relative_hmm_alm(data_train, relative_set, 10);
    time_info(ii)=toc(t);
    fprintf(f1, 'Time %f\n', ii, time_info(ii));
    model=model_list(:, model_idx);
    prior=[log(model(1).pi_init+eps), log(model(2).pi_init+eps)];
    transmat=cat(3, log(model(1).pi+eps), log(model(2).pi+eps));
    obsmat=cat(3, log(model(1).theta+eps), log(model(2).theta+eps));
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
    %% ====================================================================
    % save result
    fprintf(f1, 'New: Prior %f(%f), Train %f(%f), Test %f(%f)\n', sum(relative_set(:)), sum(relative_set(:))/37500, sum(train_result(:)), sum(train_result(:))/37500, sum(test_result(:)), sum(test_result(:))/937500);
    save(num2str(ii, 'simulation_alm_result_%02d'), 'model_list', 'model_idx', 'model_score', 'relative_set',...
        'train_path', 'train_loglik', 'train_result', 'train_path', 'test_loglik', 'test_result');
    model=model_list(:, 1);
    prior=[log(model(1).pi_init+eps), log(model(2).pi_init+eps)];
    transmat=cat(3, log(model(1).pi+eps), log(model(2).pi+eps));
    obsmat=cat(3, log(model(1).theta+eps), log(model(2).theta+eps));
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
    %% ====================================================================
    % save result
    fprintf(f1, 'Hmm: Prior %f(%f), Train %f(%f), Test %f(%f)\n', sum(relative_set(:)), sum(relative_set(:))/37500, sum(train_result(:)), sum(train_result(:))/37500, sum(test_result(:)), sum(test_result(:))/937500);
    save(num2str(ii, 'simulation_hmm_result_%02d'), 'model_list', 'model_idx', 'model_score', 'relative_set',...
        'train_path', 'train_loglik', 'train_result', 'train_path', 'test_loglik', 'test_result');
    drawnow('update')
end
fclose(f1);
sendEmail2Self('Result for improved', sprintf('%f, ', time_info), 'simulation_new_result.txt');
