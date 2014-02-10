%% ========================================================================
% this function computes the backward algorithm
function [alpha loglik]=forward_message(p_xt_zt, model)
    alpha=zeros(size(p_xt_zt));
    alpha(:, 1)=model.pi_init.*p_xt_zt(:, 1);
    alpha(:, 1)=alpha(:, 1)/sum(alpha(:, 1));
    loglik=0;
    for i=2: size(p_xt_zt, 2)
        alpha(:, i)=model.pi'*alpha(:, i-1).*p_xt_zt(:, i);
        z=sum(alpha(:, i));
        loglik=loglik+log(z);
        alpha(:, i)=alpha(:, i)/z;
    end
end
