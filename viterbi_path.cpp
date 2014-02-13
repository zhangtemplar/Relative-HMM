// this function implements vertbi algorithm for finding optimal path
/*
 * usage: [path loglik]= viterbi_path(prior, transmat, obslik)
 * Input:
 *  prior:  [kx1] the intial transition vector in log scale
 *  transmat: [kxk] the transition probability matrix in log scale, where 
 *  transmat(i,j) is the probability from State i to State j
 *  obslik: [kxt] the observation probability for each state in log scale
 * Output
 *  path:   [1xt]the optimal state path
 *  loglik: [1x1]the log likelihood associated with the optimal path
 */
#include "mex.h"
#include "matrix.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /*
     *check the input first
     */
    if (nrhs!=3 || nlhs<1)
    {
        mexErrMsgTxt("Usage: [path loglik]= viterbi_path(prior, transmat, obslik)");
    }
    // length
    int T=mxGetN(prhs[2]);
    // # states
    int Q=mxGetM(prhs[2]);
    // the pointers
    double *prior=mxGetPr(prhs[0]);
    double *transmat=mxGetPr(prhs[1]);
    double *obslik=mxGetPr(prhs[2]);
    // some variables
    double *delta=new double[Q*T];
    double *psi=new double[Q*T];
    if (delta==0 || psi==0)
    {
        delete []delta;
        delete []psi;
        mexErrMsgTxt("Memory allocation error");
    }
    // for the result
    plhs[0]=mxCreateDoubleMatrix(1, T, mxREAL);
    if (plhs[0]==0)
    {
        delete []delta;
        delete []psi;
        mexErrMsgTxt("Memory allocation error");
    }
    double *path=mxGetPr(plhs[0]);
    /*
     * use the dynamic programming
     */
    int t=0;
    int j=0;
    int i=0;
    for (j=0; j<Q; j++)
    {
        delta[j]=prior[j]+obslik[j];
        psi[j]=-1;
    }
    double tmp;
    for (t=1; t<T; t++)
    {
        for (j=0; j<Q; j++)
        {
            delta[j+t*Q]=-mxGetInf();
            psi[j+t*Q]=-1;
            for (i=0; i<Q; i++)
            {
                tmp=delta[i+(t-1)*Q]+transmat[i+j*Q];
                if (tmp>delta[j+t*Q])
                {
                    delta[j+t*Q]=tmp;
                    psi[j+t*Q]=i;
                }
            }
            delta[j+t*Q]+=obslik[j+t*Q];
        }
    }
    /*
     * back trace to find the optimal path
     */
    tmp=-mxGetInf();
    for (j=0; j<Q; j++)
    {
        if (tmp<delta[j+(T-1)*Q])
        {
            tmp=delta[j+(T-1)*Q];
            path[T-1]=j;
        }
    }
    for (t=T-2; t>=0; t--)
    {
        path[t]=psi[(int) (path[t+1]+(t+1)*Q)];
    }
    /*
     * compute the loglik
     */
    if (nlhs>1)
    {
        tmp=prior[(int) path[0]]+obslik[(int) path[0]];
        for (t=1; t<T; t++)
        {
            tmp+=transmat[(int)(path[t-1]+path[t]*Q)]+obslik[(int)(path[t]+t*Q)];
        }
        plhs[1]=mxCreateDoubleScalar(tmp);
    }
    /*
     * in matlab index starts from 1
     */
    for (t=0; t<T; t++)
    {
        path[t]=path[t]+1;
    }
}