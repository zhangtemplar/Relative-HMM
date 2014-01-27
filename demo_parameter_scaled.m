% demo_parameter
clear;
clc;
load('simulation_data');
f1=fopen('simulation_old2_result.txt', 'w');
number_pair_train=[0.9875 0.975 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1];
for ii=1: length(number_pair_train)
    %% ========================================================================
    % ground truth
    relative_set=zeros(num_model*num_train);
    relative_order=[2 5 1 6 3 4];
    for i=1: num_model-1
        relative_set(1+(relative_order(i)-1)*num_train: relative_order(i)*num_train,...
            (relative_order(i+1)-1)*num_train+1: relative_order(i+1)*num_train)=1;
    end
    relative_set(rand(size(relative_set))<number_pair_train(ii))=0;
    fprintf(f1, 'Experiment %d\tPrior %d\t', ii, sum(relative_set(:)));
    %% ========================================================================
    % train
    [model_list model_idx model_score]=relative_hmm(data_train, relative_set, 10, [], 1);
    prior=log(model_list(model_idx).pi_init+eps);
    transmat=log(model_list(model_idx).pi+eps);
    obsmat=log(model_list(model_idx).theta+eps);
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
    fprintf(f1, 'Training %d\t', ii, sum(train_result(:)));
    %% ========================================================================
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
    fprintf(f1, 'Testing %d\n', ii, sum(test_result(:)));
    %% ====================================================================
    % save result
    save(num2str(ii, 'simulation_old2_result_%02d'), 'model_list', 'model_idx', 'model_score', 'relative_set',...
        'train_path', 'train_loglik', 'train_result', 'train_path', 'test_loglik', 'test_result');
end
fclose(f1);
