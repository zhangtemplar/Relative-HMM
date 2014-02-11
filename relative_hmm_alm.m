%{
this function implements relative hmms model two. Note, the data on the
left of operator is used for Category 1 and data on the right is used for
Category 2
For optimization, we use augmented lagrange multipliers, where the
nonlinear equality constraints are elimated. The original formulation is
found to be slower than the primal-dual interior point method. In this
version, we combine the alm iteration with the orignal one.
We shall have two different formulations:
    1. nonlinear objective function with linear inequality constraints;
    2. least square problem with linear inequality constraints. for this
    one we need to do some approximation/relaxation.
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
    V4: we use simulated anealing to solve the local optimality problem
    V3: for speed consideration, we decided to use IBM cplex. MOSEK has
    certain bugs.
    V2: we add slack variables z to sperate the nonlinear constraints from
    the orginal problem, i.e., z=e^x.
    V1: We add slack variables for the cases of nonseperable cases
%}   
function [model_list idx model_score]=relative_hmm_alm(...
            data, relative_set, num_state, model,...
            update_observation, do_scale, svm_C, gap_likelihood, max_iter)
    %% ====================================================================
    % constant
    % maximal number of iteration for hmm
    max_iter_init=20;
    % the tolerance of loglikelihood
    tol_lik=1e-3;
    % the tolerance of convergence
    tol_fun=1e-3;
    % initial temprature for sa algorithms. The temporature at t iteration
    % is computed as T/ln(t), with t>=1
    temp_init=100;
    % ---------------------------------------------------------------------
    % the maximal number of iteration for alm algorithm
    max_iter_alm=100;
    % step size for alm. smaller step size means lower convergence speed
    % but better final convergence. We choose rho=1.5 as compromise. In
    % addition, rho>0.
    rho=1.25;
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
    % optimize the nonlinear problem. we use intrior point method and
    % provide analytic form for the gradient, Hessian.
    fmincon_options=optimset('Algorithm', 'interior-point', 'GradObj', 'on',...
        'GradConstr', 'on', 'Hessian', 'user-supplied', 'Display', 'off',...
        'TolCon', 1e-3); 
    z_idx1=1: num_state+1;
    z_idx2=num_state+2: num_state+1+num_dim;
    z_idx3=num_state+2+num_dim: 2*num_state+2+num_dim;
    z_idx4=2*num_state+3+num_dim: 2*num_state+2+2*num_dim;
    f1=fopen('converge.txt', 'w');
    %% ====================================================================
    % init with ordinary HMM
    % for Model 1
    if nargin<4 || isempty(model)
        model.pi_init=rand(num_state, 1);
        model.pi_init=model.pi_init/sum(model.pi_init);
        model.pi=rand(num_state, num_state);
        model.pi=model.pi./repmat(sum(model.pi, 2), 1, num_state);
        model.theta=rand(num_state, num_dim);
        model.theta=model.theta./repmat(sum(model.theta, 2), 1, num_dim);
        model=repmat(model, 2, 1);
        model(1)=train_hmm(data(index_positive), model(1), max_iter_init, tol_lik);
        model(2)=train_hmm(data(index_negative), model(2), max_iter_init, tol_lik);
    end
    % create the slack variables
    pi_init(:, 1)=log(model(1).pi_init+eps);
    pi(:, :, 1)=log(model(1).pi+eps);
    theta(:, :, 1)=log(model(1).theta+eps);
    % model 2
    pi_init(:, 2)=log(model(2).pi_init+eps);
    pi(:, :, 2)=log(model(2).pi+eps);
    theta(:, :, 2)=log(model(2).theta+eps);
    %% ====================================================================
    % apply the relative constraint
    % when we use multi-nomial observation model, the problem is a linear
    % programming
    model_list=repmat(model, 1, max_iter);
    model_score=zeros(1, max_iter);
    iter=1;
    model_list(:, iter)=model;
    if update_observation
        x=log([model(1).pi_init; model(1).pi(:); model(1).theta(:);...
            model(2).pi_init; model(2).pi(:); model(2).theta(:)]+eps);
    else
        x=log([model(1).pi_init; model(1).pi(:); model(2).pi_init; model(2).pi(:)]+eps);
    end
    % add the slack variables
    x=[x; zeros(num_relatives, 1)];
    lambda=x(1: size(Aeq, 2));
    mu=1.25/norm(lambda);
    lambda=lambda*mu;
    prev_conv=inf;
    new_conv=inf;
    tic;
    fprintf(1, 'Iteration |Objective |Log Likeli|Constraint|Gap\n');
    while 1
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
        tmp_conv=A*x(1: num_variable);
        log_lik=-f'*x(1: num_variable);
        now_conv=svm_C*sum(max(tmp_conv-b, 0))+log_lik;
        model_score(iter)=sum(tmp_conv<0);
        fprintf(1, '%10d|%.4e|%.4e|%10d|%.4e\n', iter, now_conv, log_lik, model_score(iter), new_conv);
        A_sparse=sparse([A, -eye(num_relatives)]);
        %% ----------------------------------------------------------------
        %{
        M step: find the new model under the relative constraint
        %}
        % -------------------------------------------------------------
        % solve x with QP
        % Profile indicates that this QP takes most of the time. Some
        % Google search shows that QP in Matlab is very slow. So try
        % MOSEK. We find MOSEK has some bugs and try CPLEX
        x=cplexqp(H, [-f+lambda-mu*x(1: size(Aeq, 2)); svm_C*ones(num_relatives, 1)]/mu,...
            A_sparse, b, [], [], lb, ub, [], struct('Display', 'off'));
        % -------------------------------------------------------------
        % solve z with QP
        z=exp(x(1: size(Aeq, 2))+lambda/mu);
        z=reshape(z, num_state, []);
        z(:, z_idx1)=z(:, z_idx1)./repmat(sum(z(:, z_idx1)), num_state, 1);
        z(:, z_idx3)=z(:, z_idx3)./repmat(sum(z(:, z_idx3)), num_state, 1);
        if update_observation
            z(:, z_idx2)=z(:, z_idx2)./repmat(sum(z(:, z_idx2), 2), 1, num_dim);
            z(:, z_idx4)=z(:, z_idx4)./repmat(sum(z(:, z_idx4), 2), 1, num_dim);
        end
        % refine the result
        % this is a nonconvex minimization with linear constraint, in
        % addition, the problem is seperable
        % for state transition
        z=update_model(z(:)+eps, x(1: size(Aeq, 2)), lambda, mu, num_state, num_dim);
        z=log(z(:)+eps);
        % -----------------------------------------------------------------
        % update the multiplier
        new_conv=norm(x(1: size(Aeq, 2))-z);
        lambda=lambda+mu*(x(1: size(Aeq, 2))-z);
        mu=mu*rho;
        x(1: size(Aeq, 2))=z;
        % -----------------------------------------------------------------
        % format the model
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
        % check the temporature. Note we use this scheme as a simiplication
        % and accerlation. Since there is no randomization in the nonlinear
        % minization function, thus reject the model and redo minization
        % will simply return that model again.
        % we the use the descent of the first two iteration for
        % initialization temporature
        if iter==2
            temp_init=abs(now_conv-prev_conv);
            fprintf(1, 'Initial temperature: %f\n', temp_init);
        elseif now_conv>=prev_conv && iter>.1*max_iter
            accep_ratio=1/(1+exp((now_conv-prev_conv)*log(iter)/temp_init));
            % Check the acceptance of the new mode, if it is accepted, then
            % we continue with the accepted one; otherwise, we 
            if rand(1)>accep_ratio
                break;
            end
        end
        iter=iter+1;
        prev_conv=now_conv;
        model_list(:, iter)=model;
        % check the convergence
        if iter>=max_iter || (new_conv<tol && model_score(iter)>=num_relatives)
            fprintf(1, '%10d|%.4e|%.4e|%10d|%.4e\n', iter, now_conv, log_lik, model_score(iter), new_conv);
            break;
        end
    end
    t2=toc;
    % =====================================================================
    % find the best models.
    model_list=model_list(:, 1: iter);
    [val idx]=sort(model_score, 'descend');
    % just double check to avoid trivial solutions, i.e., two are the same
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
    fprintf(1, 'Best model is obtained at Iter %d where %d constraints are statisfied\n', idx, val);
    fprintf(1, 'Total time is %f for %d iterations\n', t2, iter);
    fclose(f1);
end

