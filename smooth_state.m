function [y loglik]=smooth_state(obs, trans, prior)
% smooth_state
% Input
%   obs:    KxT the probability matrix for the observation. In our case,
%       this is the output from random forest;
%   trans:  KxK the transition probability matrix. (i,j) is the probability
%       from state i to j
%   prior:  Kx1 the initial transition probability matrix
% Output
%   y:      1xT the smoothed state path
%   loglik: the loglik of the path, you can view as the confidence score.
% K is the number of states and T is the length of sequence
obs=log(obs+eps);
[y loglik]=viterbi_path(prior, trans, obs);
