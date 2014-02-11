%% ========================================================================
% this function solves subproblem 2
function w2=update_model(w, u, lambda, mu, num_state, num_dim)
    w=reshape(w, num_state, []);
    u=reshape(u, num_state, []);
    lambda=reshape(lambda, num_state, []);
    options=optimset('Algorithm', 'interior-point', 'GradObj', 'on',...
        'GradConstr', 'on', 'Hessian', 'user-supplied', 'Display', 'off',...
        'TolCon', 1e-3); 
    w2=zeros(size(w));
    % for state transition
    lb=zeros(num_state, 1);
    ub=ones(num_state, 1);
    Aeq=ones(1, num_state);
    for i=1: num_state+1
        % for model 1
        ww=w(:, i);
        uu=u(:, i);
        ll=lambda(:, i);
        options.HessFcn=@(x,l)diag((mu*(1-log(x+eps)+uu)+ll)./(x.^2+eps));
        w2(:, i)=fmincon(@(x)obj_func(x, uu, ll, mu), ww, [], [], Aeq, 1, lb, ub, [], options);
        % for model 2
        ww=w(:, i+num_state+1+num_dim);
        uu=u(:, i+num_state+1+num_dim);
        ll=lambda(:, i+num_state+1+num_dim);
        options.HessFcn=@(x,l)diag((mu*(1-log(x+eps)+uu)+ll)./(x.^2+eps));
        w2(:, i+num_state+1+num_dim)=fmincon(@(x)obj_func(x, uu, ll, mu), ww, [], [], Aeq, 1, lb, ub, [], options);
    end
    % for observation prob.
    for i=1: num_state
        % for model 1
        ww=w(i, num_state+2: num_state+num_dim+1)';
        uu=u(i, num_state+2: num_state+num_dim+1)';
        ll=lambda(i, num_state+2: num_state+num_dim+1)';
        options.HessFcn=@(x,l)diag((mu*(1-log(x+eps)+uu)+ll)./(x.^2+eps));
        w2(i, num_state+2: num_state+num_dim+1)=fmincon(@(x)obj_func(x, uu, ll, mu), ww, [], [], Aeq, 1, lb, ub, [], options)';
        % for model 2
        ww=w(i, 2*num_state+3+num_dim: 2*num_state+2+2*num_dim)';
        uu=u(i, 2*num_state+3+num_dim: 2*num_state+2+2*num_dim)';
        ll=lambda(i, 2*num_state+3+num_dim: 2*num_state+2+2*num_dim)';
        options.HessFcn=@(x,l)diag((mu*(1-log(x+eps)+uu)+ll)./(x.^2+eps));
        w2(i, 2*num_state+3+num_dim: 2*num_state+2+2*num_dim)=fmincon(@(x)obj_func(x, uu, ll, mu), ww, [], [], Aeq, 1, lb, ub, [], options)';
    end
    w2=w2(:);
end
%% ========================================================================
function [f g]=obj_func(w, u, lambda, mu)
    lw=log(w+eps);
    f=lambda'*(u-lw)+sum((u-lw).^2)*mu/2;
    g=(mu*(lw-u)-lambda)./(w+eps);
end