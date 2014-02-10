%% ========================================================================
% this function computes the emission prob. with multinomial model
function p_xt_zt=emission_multinomial(data, theta_log, use_log)
    if nargin<3 || isempty(use_log)
        use_log=0;
    end
    % p_xt_zt{i}=zeros(size(theta_log, 2), size(data{i}, 2));
    p_xt_zt=theta_log*data;
%     p_xt_zt=p_xt_zt-repmat(mean(p_xt_zt), size(p_xt_zt, 1), 1);
    if use_log~=1
        p_xt_zt=exp(p_xt_zt);
    end
%     p_xt_zt=p_xt_zt./repmat(sum(p_xt_zt), size(p_xt_zt, 1), 1);
end
