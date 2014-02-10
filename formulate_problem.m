%% ========================================================================
% this function formulate the problem into a LP
function [f A b]=formulate_problem(stats, relative_set, gap_likelihood, update_observation)
    if nargin<4 || isempty(update_observation)
        update_observation=1;
    end
    if update_observation
        [f A b]=formulate_problem1(stats, relative_set, gap_likelihood);
    else
        [f A b]=formulate_problem2(stats, relative_set, gap_likelihood)
    end
end
%% ========================================================================
function [f A b]=formulate_problem1(stats, relative_set, gap_likelihood)
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
