%% ========================================================================
% this function computes the expectation of state transitions
function [epsilon pi epsilons]=compute_epsilon(alpha, beta, p_xt_zt, model)
    num_state=size(alpha{1}, 1);
    epsilons=cell(size(alpha));
    epsilon=zeros(num_state, num_state);
    for k=1: length(alpha)
        epsilons{k}=zeros(num_state, num_state);
        for t=2: size(alpha{k}, 2)
            tmp=model.pi.*(alpha{k}*(beta{k}.*p_xt_zt{k})');
            % tmp=alpha{k}(:, t-1)*beta{k}(:, t)'.*model.pi.*repmat(p_xt_zt{k}(:, t), 1, num_state);
            epsilons{k}=epsilons{k}+tmp./repmat(sum(tmp), size(tmp, 1), 1);
        end
        epsilon=epsilon+epsilons{k};
    end
    if nargout>1
        pi=epsilon./repmat(sum(epsilon, 2), 1, size(epsilon, 2));
    end
end
