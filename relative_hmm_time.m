%{
this function implements relative hmms model two. Note, the data on the
left of operator is used for Category 1 and data on the right is used for
Category 2
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
function [model_list idx model_score]=relative_hmm_time(data, relative_set, num_state, model)
    %% ====================================================================
    % constant
    % maximal number of iteration for hmm
    max_iter_init=20;
    % maximal number of iteration for main algorithms
    max_iter=100;
    % do we normalize the likelihood given the video length
    do_scale=0;
    % do we require the probability vector sums to 1
    % do_normalize=0;
    % the weight to the slack variables
    svm_C=1000;
    % what is the gap for two comparing pair of loglikelihood
    gap_likelihood=10;
    % do we incooprating the observation model into the formulation
    update_observation=1;
    % the tolerance of loglikelihood
    tol_lik=1e-3;
    % the tolerance of convergence
    tol_fun=1e-3;
    % initial temprature for sa algorithms. The temporature at t iteration
    % is computed as T/ln(t), with t>=1
    temp_init=100;
    % ---------------------------------------------------------------------
    % use alm algorithm
    use_alm=0;
    % the maximal number of iteration for alm algorithm
    max_iter_alm=100;
    % step size for alm. smaller step size means lower convergence speed
    % but better final convergence. We choose rho=1.5 as compromise. In
    % addition, rho>0.
    rho=1.5;
    % tolarence for alm convergence
    tol=1e-3;
    %% ====================================================================
    % variables
    % num_state=size(model.pi, 1);
    num_dim=size(data{1}, 1);
    num_data=length(data);
    num_relatives=sum(relative_set(:)==1);
    index_positive=find(sum(relative_set==1, 2))';
    index_negative=find(sum(relative_set==1, 1));
    % randomize the model
    if nargin<4 || isempty(model)
        model.pi_init=rand(num_state, 1);
        model.pi_init=model.pi_init/sum(model.pi_init);
        model.pi=rand(num_state, num_state);
        model.pi=model.pi./repmat(sum(model.pi, 2), 1, num_state);
        model.theta=rand(num_state, num_dim);
        model.theta=model.theta./repmat(sum(model.theta, 2), 1, num_dim);
    end
    % we need two model
    if length(model)==1
        model=repmat(model, 2, 1);
    end
    path=cell(2, num_data);
    p_xt_zt=cell(2, num_data);
    pi_init=zeros(num_state, 2);
    pi=zeros(num_state, num_state, 2);
    theta=zeros(num_state, num_dim, 2);
    stats=struct('n', zeros(num_state, 1), 'm', zeros(num_state, num_state),...
        'x', zeros(num_dim, num_state), 't', 1);
    stats=repmat(stats, 2, num_data);
    if do_scale
        for i=1: length(data)
            stats(i).t=size(data{i}, 2);
        end
    end
    if update_observation
        lb=-inf(2*num_state*(num_state+num_dim+1), 1);
        ub=zeros(2*num_state*(num_state+num_dim+1), 1);
    else
        lb=-inf(2*num_state*(num_state+1), 1);
        ub=zeros(2*num_state*(num_state+1), 1);
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
        num_variable2=num_state*(num_state+1+num_dim);
        Aeq=[Aeq zeros(size(Aeq)); zeros(size(Aeq)), Aeq];
    else
        Aeq=zeros(num_state+1, num_state*(num_state+1));
        Aeq(1, 1: num_state)=1;
        for i=1: num_state
            Aeq(i+1, num_state*(1: num_state)+i)=1;
        end
        num_variable2=num_state*(num_state+1);
        Aeq=[Aeq zeros(size(Aeq)); zeros(size(Aeq)), Aeq];
    end
    num_variable=num_variable2*2;
    Aeq=sparse(Aeq);
    H=zeros(size(Aeq, 2)+num_relatives);
    H(1: size(Aeq, 2), 1: size(Aeq, 2))=eye(size(Aeq, 2));
    H=sparse(H);
    model_list=repmat(model, 1, max_iter);
    model_score=zeros(1, max_iter);
    % optimize the nonlinear problem. we use intrior point method and
    % provide analytic form for the gradient, Hessian.
    fmincon_options=optimset('Algorithm', 'interior-point', 'GradObj', 'on',...
        'GradConstr', 'on', 'Hessian', 'user-supplied', 'Display', 'off',...
        'TolCon', 1e-3); 
    z_idx1=1: num_state+1;
    z_idx2=num_state+2: num_state+1+num_dim;
    z_idx3=num_state+2+num_dim: 2*num_state+2+num_dim;
    z_idx4=2*num_state+3+num_dim: 2*num_state+2+2*num_dim;
    %% ====================================================================
    % init with ordinary HMM
    % for Model 1
    model(1)=train_hmm(data(index_positive), model(1), max_iter_init, tol_lik);
    pi_init(:, 1)=log(model(1).pi_init+eps);
    pi(:, :, 1)=log(model(1).pi+eps);
    theta(:, :, 1)=log(model(1).theta+eps);
    for i=1: length(data)
        p_xt_zt{1, i}=emission_multinomial(data{i}, theta(:, :, 1), 1);
        path{1, i}=viterbi_path(pi_init(:, 1), pi(:, :, 1), p_xt_zt{1, i});
        [stats(1, i).n stats(1, i).m stats(1, i).x]=count_state(path{1, i}, data{i}, model(1), do_scale);
    end
    % model 2
    model(2)=train_hmm(data(index_negative), model(2), max_iter_init, tol_lik);
    pi_init(:, 2)=log(model(2).pi_init+eps);
    pi(:, :, 2)=log(model(2).pi+eps);
    theta(:, :, 2)=log(model(2).theta+eps);
    for i=1: length(data)
        p_xt_zt{2, i}=emission_multinomial(data{i}, theta(:, :, 2), 1);
        path{2, i}=viterbi_path(pi_init(:, 2), pi(:, :, 2), p_xt_zt{2, i});
        [stats(2, i).n stats(2, i).m stats(2, i).x]=count_state(path{2, i}, data{i}, model(2), do_scale);
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
        x=log([model(1).pi_init; model(1).pi(:); model(1).theta(:);...
            model(2).pi_init; model(2).pi(:); model(2).theta(:)]+eps);
    else
        x=log([model(1).pi_init; model(1).pi(:); model(2).pi_init; model(2).pi(:)]+eps);
    end
    % add the slack variables
    x=[x; zeros(num_relatives, 1)];
    tmp_conf=0;
    if 1 % while 1
        %% ----------------------------------------------------------------
        %{
        E step: find the optimal path
        %}
        % the optimal path acctually changes a lot 
        for j=1: 2
            for i=1: length(data)
                p_xt_zt{j, i}=emission_multinomial(data{i}, theta(:, :, j), 1);
                path{j, i}=viterbi_path(pi_init(:, j), pi(:, :, j), p_xt_zt{j, i});
                [stats(j, i).n stats(j, i).m stats(j, i).x]=count_state(path{j, i}, data{i}, model(j), do_scale);
            end
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
        prev_conv=A*x(1: num_variable);
        log_lik=-f'*x(1: num_variable);
        now_conv=svm_C*sum(max(prev_conv-b, 0))+log_lik;
        model_score(iter)=sum(prev_conv<0);
        %{
        fprintf(1, 'Initialization: Iter\t%02d, Objectve\t%f, Log likelihood\t%f,Constraint\t%d\n',...
            iter, now_conv, log_lik, model_score(iter));
        %}
        A_sparse=sparse([A, -eye(num_relatives)]);
        f_full=[f; -svm_C*ones(num_relatives, 1)];
        %% ----------------------------------------------------------------
        %{
        M step: find the new model under the relative constraint
        %}
        tstart=tic;
        % use alm algorithm
        x1=update_model(x, Aeq, H, f, A_sparse, b, lb, ub, svm_C,...
            num_state, num_dim, z_idx1, z_idx2, z_idx3, z_idx4,...
            max_iter_alm, tol, update_observation, num_relatives, rho);
        t1=toc(tstart);
        tstart=tic;
        % use Matlab nonlinear minimization routine
        fmincon_options.HessFcn=@(x,lambda)(sparse(diag(...
            [Aeq'*lambda.eqnonlin.*exp(x(1: num_variable)); zeros(num_relatives, 1)])));
        fmincon_options.TolFun=nonlinear_objective(x, -f_full)*tol_fun;
        % do the optimization
        [x2,fval,exitflag]=fmincon(@(val)nonlinear_objective(val, -f_full), x, A_sparse, b, [], [], lb, ub,...
            @(val)nonlinear_constraint(val, Aeq), fmincon_options);
        t2=toc(tstart);
        for j=1: 2
            model(j).pi_init=exp(x(num_variable2*(j-1)+(1: num_state)));
            pi_init(:, j)=x(num_variable2*(j-1)+(1: num_state));
            model(j).pi=reshape(exp(x(num_variable2*(j-1)+(num_state+1: num_state*(num_state+1)))), num_state, num_state);
            pi(:, :, j)=reshape(x(num_variable2*(j-1)+(num_state+1: num_state*(num_state+1))), num_state, num_state);
            if update_observation
                model(j).theta=reshape(exp(x(num_variable2*(j-1)+(num_state*(num_state+1)+1: num_state*(num_state+num_dim+1)))), num_state, num_dim);
                theta(:, :, j)=reshape(x(num_variable2*(j-1)+(num_state*(num_state+1)+1: num_state*(num_state+num_dim+1))), num_state, num_dim);
            end
        end
        %% ----------------------------------------------------------------
        % check the convergence
        %{
        if (iter>=max_iter || abs((now_conv-prev_conv)/now_conv)<tol ||...
                exitflag<0 || model_score(iter)>=num_relatives)
            fprintf(1, 'Optimization finished: Iter,%d\tStep,%f\tExit,%d\n', iter, now_conv, exitflag);
            break;
        end
        iter=iter+1;
        model_list(:, iter)=model;
        prev_conv=now_conv;
        %}
    end
    % =====================================================================
    %{
    model_list=model_list(:, 1: iter);
    [val idx]=sort(model_score, 'descend');
    for i=idx
        if norm(model_list(1, i).pi_init-model_list(2, i).pi_init)+...
            norm(model_list(1, i).pi(:)-model_list(2, i).pi(:))+...
            norm(model_list(1, i).theta(:)-model_list(2, i).theta(:))>=1e-6
            break;
        end
    end
    val=model_score(i);
    idx=i;
    % we need to reject the trivial models
    fprintf(1, 'Best model is obtained at Iter %d where %d constraints are statisfied', idx, val);
    %}
    fprintf(1, 'Computational time for %d pairs is %f for ALM and %f for old.\n', num_relatives, t1, t2);
end
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
%% ========================================================================
% this function formulate the problem into a LP
function [f A b]=formulate_problem(stats, relative_set, gap_likelihood)
    num_state=size(stats(1).x, 1);
    num_dim=size(stats(1).x, 2);
    index_positive=find(sum(relative_set==1, 2))';
    index_negative=find(sum(relative_set==1, 1));
    % =====================================================================
    % objective function
    f=zeros(2*num_state*(num_state+num_dim+1), 1);
    n=zeros(size(stats(1, 1).n));
    m=zeros(size(stats(1, 1).m));
    x=zeros(size(stats(1, 1).x));
    for i=index_positive
        n=n+stats(1, i).n;
        m=m+stats(1, i).m*stats(1, i).t;
        x=x+stats(1, i).x*stats(1, i).t;
    end
    f(1: num_state*(num_state+num_dim+1))=[n'; m(:); x(:)];
    n=zeros(size(stats(2, 1).n));
    m=zeros(size(stats(2, 1).m));
    x=zeros(size(stats(2, 1).x));
    for i=index_negative
        n=n+stats(2, i).n;
        m=m+stats(2, i).m*stats(2, i).t;
        x=x+stats(2, i).x*stats(2, i).t;
    end
    f(num_state*(num_state+num_dim+1)+1: end)=[n'; m(:); x(:)];
    % =====================================================================
    % linear inequality
    % we can also use the expectation of the states instead of the optimal 
    % state for the constraints
    A=zeros(sum(relative_set(:)==1), 2*num_state*(num_state+1+num_dim));
    b=-gap_likelihood*ones(size(A, 1), 1);
    k=1;
    % for the observation model we don't consider the differences of
    % probability due to observation
    for i=1: size(relative_set, 1)
        for j=1: size(relative_set, 2)
            if (relative_set(i, j)==1)
                A(k, :)=[stats(1, j).n-stats(1, i).n, stats(1, j).m(:)'-stats(1, i).m(:)', stats(1, j).x(:)'-stats(1, i).x(:)', ...
                    stats(2, i).n-stats(2, j).n, stats(2, i).m(:)'-stats(2, j).m(:)', stats(2, i).x(:)'-stats(2, j).x(:)'];
                k=k+1;
            end
        end
    end
    A=sparse(A);
end
%% ========================================================================
% a simpler formulation of the problem. We remove probability due to
% observation in the constraint.
function [f A b]=formulate_problem2(stats, relative_set, gap_likelihood, theta)
    error('Not yet implemented');
end
%% ========================================================================
% the nonlinear constraint
function [c ceq gc gceq]=nonlinear_constraint(x, Aeq)
    n=length(x)-size(Aeq, 2);
    m=size(Aeq, 2);
    gc=[];
    gceq=[Aeq'.*repmat(exp(x(1: m)), 1, size(Aeq, 1)); zeros(n, size(Aeq, 1))];
    c=[];
    ceq=Aeq*exp(x(1: m))-1;
end
%% ========================================================================
% the objective functions
function [val f]=nonlinear_objective(x, f)
    val=f'*x;
end
%% ========================================================================
% this function performs plain hmm
function model=train_hmm(data, model, max_iter_init, tol_lik)
    iter=1;
    prev_lik=-inf;
    converged=false;
    p_xt_zt=cell(1, length(data));
    alpha=cell(1, length(data));
    beta=cell(1, length(data));
    fprintf(1, 'Initialize HMM 1\n');
    while (~converged)
        loglik=0;
        % E step
        theta(:, :, 1)=log(model.theta+eps);
        for i=1: length(data)
            p_xt_zt{i}=emission_multinomial(data{i}, theta(:, :, 1));
            [alpha{i} lik]=forward_message(p_xt_zt{i}, model);
            beta{i}=backward_message(p_xt_zt{i}, model);
            loglik=loglik+lik;
        end
        % M step
    	[gamma0 gamma model.pi_init]=compute_gamma(alpha, beta);
    	[epsilon model.pi]=compute_epsilon(alpha, beta, p_xt_zt, model);
    	[gamma_x model.theta]=compute_gamma_obs(gamma, data);
        % check convergence
        fprintf(1, 'Initialization: Iter\t%02d, Likelihood %f\n', iter, loglik);
        if (iter>max_iter_init || loglik-prev_lik<tol_lik)
            converged=true;
        end
        iter=iter+1;
        prev_lik=loglik;
    end
end
%% ========================================================================
% this function implements the algorithm for updating the model
function x_new=update_model(x, Aeq, H, f, A_sparse, b, lb_qp, ub_qp, svm_C,...
    num_state, num_dim, z_idx1, z_idx2, z_idx3, z_idx4, max_iter_alm, tol,...
    update_observation, num_relatives, rho)
    %% ----------------------------------------------------------------
    %{
    M step: find the new model 
    %}
    % initialize the multipliers
    lambda=x(1: size(Aeq, 2));
    mu=1/norm(lambda);
    lambda=lambda*mu;
    iter_alm=1;
    % -----------------------------------------------------------------
    % apply the ALM
    while 1
        % -------------------------------------------------------------
        % solve z with QP
        % luckily we have a closed form solution for it
        z=exp(x(1: size(Aeq, 2))+lambda/mu);
        z=reshape(z, num_state, []);
        z(:, z_idx1)=z(:, z_idx1)./repmat(sum(z(:, z_idx1)), num_state, 1);
        z(:, z_idx3)=z(:, z_idx3)./repmat(sum(z(:, z_idx3)), num_state, 1);
        if update_observation
            z(:, z_idx2)=z(:, z_idx2)./repmat(sum(z(:, z_idx2), 2), 1, num_dim);
            z(:, z_idx4)=z(:, z_idx4)./repmat(sum(z(:, z_idx4), 2), 1, num_dim);
        end
        z=log(z(:)+eps);
        % -------------------------------------------------------------
        % solve x with QP
        % Profile indicates that this QP takes most of the time. Some
        % Google search shows that QP in Matlab is very slow. So try
        % MOSEK. We find MOSEK has some bugs and try CPLEX
        x_new=quadprog(H, [-f+lambda-mu*z; svm_C*ones(num_relatives, 1)]/mu,...
            A_sparse, b, [], [], lb_qp, ub_qp, [], struct('Display', 'off'));
        % we could a dynamic rho here, i.e., it increases fast at
        % beginning and slow at the end
        new_conv=norm(x_new(1: size(Aeq, 2))-z)/norm(z);
        lambda=lambda+mu*(x_new(1: size(Aeq, 2))-z);
        mu=mu*rho;
        % -------------------------------------------------------------
        % check convergence and update multipliers
        if new_conv<tol || iter_alm>max_iter_alm
            % fprintf(1, '\tALM Iter %d\tConv %f\tMu %f\n', iter_alm, new_conv, mu);
            break;
        end
        iter_alm=iter_alm+1;
    end
end
