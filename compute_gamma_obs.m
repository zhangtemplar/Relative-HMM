%% ========================================================================
% this function compute expected observation
function [gamma_x theta gamma_xs]=compute_gamma_obs(gamma, data)
    num_state=size(gamma{1}, 1);
    num_dim=size(data{1}, 1);
    gamma_xs=cell(size(gamma));
    gamma_x=zeros(num_dim, num_state);
    for i=1: length(data)
        gamma_xs{i}=data{i}*gamma{i}';
        gamma_x=gamma_x+gamma_xs{i};
    end
    if nargout>1
        theta=gamma_x./repmat(sum(gamma_x), size(gamma_x, 1), 1);
        theta=theta';
    end
end
