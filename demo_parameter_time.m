% demo_parameter
clear all;
clc;
load('C:\Users\qzhang53\Documents\MATLAB\relative_hmm\simulation\simulation_data');
number_pair_train=[0.9875 0.975 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1];
time_info=zeros(size(number_pair_train));
for ii=1: length(number_pair_train)
    %% ========================================================================
    % ground truth
    relative_order=[2 5 1 6 3 4];
    try
        disp('load data');
        load(['C:\Users\qzhang53\Documents\MATLAB\relative_hmm\simulation\',...
            num2str(ii, 'simulation_new_result_%02d')], 'relative_set');
    catch ME
        disp('data not found, create a new one instead');
        relative_set=zeros(num_model*num_train);
        for i=1: num_model-1
            relative_set(1+(relative_order(i)-1)*num_train: relative_order(i)*num_train,...
                (relative_order(i+1)-1)*num_train+1: relative_order(i+1)*num_train)=1;
        end
        relative_set(rand(size(relative_set))<number_pair_train(ii))=0;
    end
    relative_hmm(data_train, relative_set, 10, [], 'improved', 'alm', 0, [], [], 10);
    relative_hmm(data_train, relative_set, 10, [], 'improved', 'ip', 0, [], [], 10);
end
