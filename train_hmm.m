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
