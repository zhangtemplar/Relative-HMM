%% ========================================================================
% compute the statiscs of the path, including the initial state, state
% counts and accumulative observation. Espeically, as the probability is
% biased to the shorter sequence, we normalize the probability by dividing
% its length
function [n m x]=count_state(path, data, model, do_normalize)
    num_dim=size(data, 1);
    num_state=size(model.pi, 1);
    n=zeros(1, num_state);
    m=zeros(num_state, num_state);
    x=zeros(num_dim, num_state);
    n(1, path(1))=1;
    x(:, path(1))=x(:, 1);
    for t=2: size(path, 2)
        m(path(t-1), path(t))=m(path(t-1), path(t))+1;
        x(:, path(t))=x(:, path(t))+data(:, t);
    end
    x=x';
    if do_normalize
        m=m/size(data, 2);
        x=x/size(data, 2);
    end
end
