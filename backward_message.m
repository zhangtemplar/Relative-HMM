%% ========================================================================
% this function computes the backward algorithm
function beta=backward_message(p_xt_zt, model)
    num_state=size(p_xt_zt, 1);
    beta=zeros(size(p_xt_zt));
    beta(:, end)=1/num_state;
    for i=size(p_xt_zt, 2)-1: -1: 1
        beta(:, i)=model.pi*(beta(:, i+1).*p_xt_zt(:, i+1));
        beta(:, i)=beta(:, i)/sum(beta(:, i));
    end
end
