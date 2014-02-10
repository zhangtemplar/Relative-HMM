%{
this function implements relative hmms.
Input
    data:           the training sequences. {1xN} cell and each cell is [dxT]
        matrix;
    relative_set:   the information is relative set. [NxN] (sparse) matrix. 
        (i,j)=1 means the likelihood of Sequence i is larger than that of j;
    num_state:      the number of states in the model
    model:          hidden markov model (only multinomial supported now).
        [optional]
        pi_init:    the initial probability [Kx1] vector;
        pi:         the probability transition matrix [kxk]. (i,j) is the
            probability transiting from State i to State j
        theta:      the observation model [kxd] matrix;
Output
    model:          check the input
History
    We add slack variables for the cases of nonseperable cases
%}   
function [model_list idx model_score]=relative_hmm_base(...
    data, relative_set, num_state, model,...
    update_observation, do_scale, svm_C, gap_likelihood, max_iter)
    %% ====================================================================
    % constant
    max_iter_init=20;
    % the tolerance of loglikelihood
    tol_lik=1e-3;
    % the tolerance of convergence
    tol_fun=1e-3;
    %% ====================================================================
    % variables
    % num_state=size(model.pi, 1);
    num_dim=size(data{1}, 1);
    num_data=length(data);
    num_relatives=sum(relative_set(:)==1);
    % randomize the model
    if nargin<4 || isempty(model)
        model.pi_init=rand(num_state, 1);
        model.pi_init=model.pi_init/sum(model.pi_init);
        model.pi=rand(num_state, num_state);
        model.pi=model.pi./repmat(sum(model.pi, 2), 1, num_state);
        model.theta=rand(num_state, num_dim);
        model.theta=model.theta./repmat(sum(model.theta, 2), 1, num_dim);
    end
    alpha=cell(size(data));
    beta=cell(size(data));
    path=cell(size(data));
    p_xt_zt=cell(size(data));
    stats=struct('n', zeros(num_state, 1), 'm', zeros(num_state, num_state),...
        'x', zeros(num_dim, num_state), 't', 1);
    stats=repmat(stats, size(data));
    if do_scale
        for i=1: length(data)
            stats(i).t=size(data{i}, 2);
        end
    end
    if update_observation
        lb=-inf(num_state*(num_state+num_dim+1), 1);
        ub=zeros(num_state*(num_state+num_dim+1), 1);
    else
        lb=-inf(num_state*(num_state+1), 1);
        ub=zeros(num_state*(num_state+1), 1);
    end
    lb=[lb; zeros(num_relatives, 1)];
    ub=[ub; inf(num_relatives, 1)];
    % for the relative constraints
    if update_observation
        Aeq=zeros(num_state*2+1, num_state*(num_state+1+num_dim));
        Aeq(1, 1: num_state)=1;
        for i=1: num_state
            Aeq(i+1, num_state*(1: num_state)+i)=1;
            Aeq(i+1+num_state, num_state*(1: num_dim)+i+num_state*num_state)=1;
        end
        num_variable=num_state*(num_state+1+num_dim);
    else
        Aeq=zeros(num_state+1, num_state*(num_state+1));
        Aeq(1, 1: num_state)=1;
        for i=1: num_state
            Aeq(i+1, num_state*(1: num_state)+i)=1;
        end
        num_variable=num_state*(num_state+1);
    end
    Aeq=sparse(Aeq);
    model_list=repmat(model, 1, max_iter);
    model_score=zeros(1, max_iter);
    % optimize the nonlinear problem. we use intrior point method and
    % provide analytic form for the gradient, Hessian.
    fmincon_options=optimset('Algorithm', 'interior-point', 'GradObj', 'on',...
        'GradConstr', 'on', 'Hessian', 'user-supplied', 'Display', 'off',...
        'TolCon', 1e-3); 
    f1=fopen('converge.txt', 'w');
    %% ====================================================================
    % init with ordinary HMM
    model=train_hmm(data, model, max_iter_init, tol_lik);
    pi_init=log(model.pi_init+eps);
    pi=log(model.pi+eps);
    theta=log(model.theta+eps);
    for i=1: length(data)
        p_xt_zt{i}=emission_multinomial(data{i}, theta, 1);
        path{i}=viterbi_path(pi_init, pi, p_xt_zt{i});
        [stats(i).n stats(i).m stats(i).x]=count_state(path{i}, data{i}, model, do_scale);
    end
    %% ====================================================================
    % apply the relative constraint
    % when we use multi-nomial observation model, the problem is a linear
    % programming
    iter=1;
    step=inf;
    model_list(:, iter)=model;
    model_prev=model;
    if update_observation
        x=log([model.pi_init; model.pi(:); model.theta(:)]+eps);
    else
        x=log([model.pi_init; model.pi(:)]+eps);
    end
    % add the slack variables
    x=[x; zeros(num_relatives, 1)];
    while 1
        %% ----------------------------------------------------------------
        %{
        E step: find the optimal path
        %}
        % the optimal path acctually changes a lot 
        for i=1: length(data)
            p_xt_zt{i}=emission_multinomial(data{i}, theta, 1);
            path{i}=viterbi_path(pi_init, pi, p_xt_zt{i});
            [stats(i).n stats(i).m stats(i).x]=count_state(path{i}, data{i}, model, do_scale);
        end
	    % formulate the problem
        if update_observation
            % the observation model is optimized at the same time
            % only available for the multinomial. It could be very slow
            [f A b]=formulate_problem(stats, relative_set, gap_likelihood);
        else
            % the observation model is fixed.
            [f A b]=formulate_problem2(stats, relative_set, gap_likelihood, theta);
        end
        model_score(iter)=sum(A*x(1: num_variable)<0);
        fprintf(1, 'Initialization: Iter\t%02d, Objectve\t%f, Constraint\t%f,Step\t%f\n',...
            iter, -f'*x(1: num_variable), model_score(iter), step);
        A=[A, -eye(num_relatives)];
        f=[f; -svm_C*ones(num_relatives, 1)];
        %% ----------------------------------------------------------------
        %{
        M step: find the new model under the relative constraint
        %}
        % we provide information for the objective gradient, constraint
        % gradient and constraint hessian
        fmincon_options.HessFcn=@(x,lambda)(sparse(diag([Aeq'*lambda.eqnonlin.*exp(x(1: num_variable)); zeros(num_relatives, 1)])));
        fmincon_options.TolFun=nonlinear_objective(x, -f)*tol_fun;
        % do the optimization
        [x,fval,exitflag]=fmincon(@(val)nonlinear_objective(val, -f), x, A, b, [], [], lb, ub,...
            @(val)nonlinear_constraint(val, Aeq), fmincon_options);
        model.pi_init=exp(x(1: num_state));
        pi_init=x(1: num_state);
        model.pi=reshape(exp(x(num_state+1: num_state*(num_state+1))), num_state, num_state);
        pi=reshape(x(num_state+1: num_state*(num_state+1)), num_state, num_state);
        if update_observation
            model.theta=reshape(exp(x(num_state*(num_state+1)+1: num_state*(num_state+num_dim+1))), num_state, num_dim);
            theta=reshape(x(num_state*(num_state+1)+1: num_state*(num_state+num_dim+1)), num_state, num_dim);
        end
        %% ----------------------------------------------------------------
        % check the convergence
        step=(norm(model_prev.pi_init-model.pi_init)+...
            norm(model_prev.pi(:)-model.pi(:))+...
            norm(model_prev.theta(:)-model.theta(:)));
        if (iter>=max_iter || step<tol_lik || exitflag<0 || model_score(iter)>=num_relatives)
            break;
        end
        iter=iter+1;
        model_list(iter)=model;
        model_prev=model;
    end
    model_list=model_list(1: iter);
    [val idx]=max(model_score);
    fprintf(1, 'Optimization finished: best model is obtained at Iter %d where %d constraints are statisfied', idx, val);
    fclose(f1);
end

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

% this function formulate the problem into a LP
function [f A b]=formulate_problem(stats, relative_set, gap_likelihood)
    num_state=size(stats(1).x, 1);
    num_dim=size(stats(1).x, 2);
    % =====================================================================
    % objective function
    n=zeros(size(stats(1).n));
    m=zeros(size(stats(1).m));
    x=zeros(size(stats(1).x));
    for i=1: length(stats)
        n=n+stats(i).n;
        m=m+stats(i).m*stats(i).t;
        x=x+stats(i).x*stats(i).t;
    end
    f=[n'; m(:); x(:)];
    % =====================================================================
    % linear inequality
    % we can also use the expectation of the states instead of the optimal 
    % state for the constraints
    A=zeros(sum(relative_set(:)==1), num_state*(num_state+1+num_dim));
    b=-gap_likelihood*ones(size(A, 1), 1);
    k=1;
    % for the observation model we don't consider the differences of
    % probability due to observation
    for i=1: size(relative_set, 1)
        for j=1: size(relative_set, 2)
            if (relative_set(i, j)==1)
                A(k, :)=[stats(j).n-stats(i).n, stats(j).m(:)'-stats(i).m(:)', stats(j).x(:)'-stats(i).x(:)'];
                k=k+1;
            end
        end
    end
    A=sparse(A);
end

% a simpler formulation of the problem. We remove probability due to
% observation in the constraint.
function [f A b]=formulate_problem2(stats, relative_set, gap_likelihood, theta)
    num_state=size(stats(1).x, 1);
    % num_dim=size(stats(1).x, 2);
    % =====================================================================
    % objective function
    n=zeros(size(stats(1).n));
    m=zeros(size(stats(1).m));
    % x=zeros(size(stats(1).x));
    for i=1: length(stats)
        n=n+stats(i).n;
        m=m+stats(i).m*stats(i).t;
        % x=x+stats(i).x*stats(i).t;
    end
    f=[n'; m(:)];
    % =====================================================================
    % linear inequality
    % we can also use the expectation of the states instead of the optimal 
    % state for the constraints
    A=zeros(sum(relative_set(:)==1), num_state*(num_state+1));
    b=zeros(size(A, 1), 1);
    k=1;
    % for the observation model we don't consider the differences of
    % probability due to observation
    for i=1: size(relative_set, 1)
        for j=1: size(relative_set, 2)
            if (relative_set(i, j)==1)
                A(k, :)=[stats(j).n-stats(i).n, stats(j).m(:)'-stats(i).m(:)'];
                b(k)=sum(sum((stats(i).x-stats(j).x).*theta))-gap_likelihood;
                k=k+1;
            end
        end
    end
    A=sparse(A);
end

% the nonlinear constraint
function [c ceq gc gceq]=nonlinear_constraint(x, Aeq)
    n=length(x)-size(Aeq, 2);
    m=size(Aeq, 2);
    gc=[];
    gceq=[Aeq'.*repmat(exp(x(1: m)), 1, size(Aeq, 1)); zeros(n, size(Aeq, 1))];
    c=[];
    ceq=Aeq*exp(x(1: m))-1;
end

% the objective functions
function [val f]=nonlinear_objective(x, f)
    val=f'*x;
end
