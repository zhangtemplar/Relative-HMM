% this is the main entrance of the code
% Input
% --[required]-------------------------------------------------------------
%   data            {1xn} where each cell is a [dxT] matrix with
%                   d-feature and T-frame
%   relative_set
%                   [nxn] the prior information, e.g., (i,j)=1, Seq. i is
%                   better than Seq. j
%   num_state
%                   [1x1] scalar for the number of states
% --[optional]-------------------------------------------------------------
%   model           [] structure for hmm. This is typically used for a warm
%                   start of the model
%   do_method       ['ip'] choose the model to use: 
%                   'ip' for interior point method for improved model
%                   'alm' for augmented lagrange multiplier for improved
%                   model too.
%                   'base' for base model
%                   Please choose 'ip' first unless you find the algorithm
%                   is too slow, then use 'alm'.
%   update_observation       
%                   [1] do we update the observation model or not. Set to
%                   1, as long as the model can be updated.
%   do_scale        [0] do we rescale the data according to its length.
%                   Not very useful.
%   svm_C           [1000] the weight for the misclassification. Very
%                   important and check the C for SVM
%   gap_likelihood  [10] the gap for loglikelihood. Very important and
%                   check the margin in SVM
%   max_iter        [100] the maximal number of iterations allowed
% Output
%   model_list      [1xm] for base method or [2xm] for improved method,
%                   which contains the list of models obtained
%   idx             [1x1] the best model in the list
%   model_score     [1xm] the objective function with each score
function [model_list idx model_score]=relative_hmm_improve(...
    data, relative_set, num_state, model,...
    do_method, update_observation, do_scale,...
    svm_C, gap_likelihood, max_iter)
    %% ====================================================================
    % parse the input
    if nargin<3
        error('usage: model=relative_hmm(data, prior, # state)');
    end
    if nargin<4
        model=[];
    end
    % chhoose the model we use
    if nargin<5 || isempty(do_method)
        do_method='ip';
    end
    % choose the solver for updating the model
    if nargin<6 || isempty(update_observation)
        update_observation=1;
    end
    % do we normalize the likelihood given the video length
    if nargin<7 || isempty(do_scale)
        do_scale=0;
    end
    % the weight to the slack variables
    if nargin<8 || isempty(svm_C)
        svm_C=1000;
    end
    % what is the gap for two comparing pair of loglikelihood
    if nargin<9 || isempty(gap_likelihood)
        gap_likelihood=10;
    end
    % maximal number of iteration for main algorithms
    if nargin<10 || isempty(max_iter)
        max_iter=1000;
    end
    %% ====================================================================
    % algorithm
    if strcmp(do_method, 'base')
        [model_list idx model_score]=relative_hmm_base(...
            data, relative_set, num_state, model,...
            update_observation, do_scale,...
            svm_C, gap_likelihood, max_iter);
    elseif strcmp(do_method, 'alm')
        [model_list idx model_score]=relative_hmm_alm(...
            data, relative_set, num_state, model,...
            update_observation, do_scale,...
            svm_C, gap_likelihood, max_iter);
    else % default case, use ip
        [model_list idx model_score]=relative_hmm_alm(...
            data, relative_set, num_state, model,...
            update_observation, do_scale,...
            svm_C, gap_likelihood, max_iter);
    end
end
