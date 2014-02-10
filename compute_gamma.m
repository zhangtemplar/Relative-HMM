%% ========================================================================
% this function compute the expectation of states in the path
function [gamma0 gamma pi_init]=compute_gamma(alpha, beta)
    num_state=size(alpha{1}, 1);
    gamma0=zeros(num_state, 1);
    gamma=cell(size(alpha));
    for i=1: length(alpha)
        gamma{i}=alpha{i}.*beta{i};
        gamma{i}=gamma{i}./repmat(sum(gamma{i}), num_state, 1);
        gamma0=gamma0+gamma{i}(:, 1);
    end
    if nargout>2
        pi_init=gamma0./sum(gamma0);
    end
end
