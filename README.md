Relative-HMM
============

This repository holds the code for Qiang Zhang; Baoxin Li, "Relative Hidden Markov Models for Evaluating Motion Skill," Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference on , vol., no., pp.548,555, 23-28 June 2013

Abstract: This paper is concerned with a novel problem: learning temporal models using only relative information. Such a problem arises naturally in many applications involving motion or video data. Our focus in this paper is on videobased surgical training, in which a key task is to rate the performance of a trainee based on a video capturing his motion. Compared with the conventional method of relying on ratings from senior surgeons, an automatic approach to this problem is desirable for its potential lower cost, better objectiveness, and real-time availability. To this end, we propose a novel formulation termed Relative Hidden Markov Model and develop an algorithm for obtaining a solution under this model. The proposed method utilizes only a relative ranking (based on an attribute of interest) between pairs of the inputs, which is easier to obtain and often more consistent, especially for the chosen application domain. The proposed algorithm effectively learns a model from the training data so that the attribute under consideration is linked to the likelihood of the inputs under the learned model. Hence the model can be used to compare new sequences. Synthetic data is first used to systematically evaluate the model and the algorithm, and then we experiment with real data from a surgical training system. The experimental results suggest that the proposed approach provides a promising solution to the real-world problem of motion skill evaluation from video.

URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6618921&isnumber=6618844

Author: Qiang Zhang
Email: qzhang53@asu.edu,zhangtemplar@gmail.com

Disclaimer: You are free to use them for your own research, provided that you cite the source explicitly in your work, do not make any redistribution or use them for commercial purposes. However, you need to use these codes at your own risk. If you find any problems or have any comments, please contact me through email. If you use these codes in your work, please cites my paper as described above.

Prerequisites: to use this package, you will need Matlab (tested on Matlab 2011a to 2013b, on Windows 7 64bit), the HMMall toolbox (available at http://www.cs.ubc.ca/~murphyk/Software/HMM/hmm_download.html) and IBM CPLEX studio (if you don't have it, please replace cplexqp from CPLEX toolbox with quadprog from Matlab builtin).

Usage: to start, please try the code demo_pair. The major function is implemented in relative_hmm, which is in turn a wrapper to relative_hmm_base, relative_hmm_ip and relative_hmm_alm, which actually implements the methods presented in the paper. For more information please refer to the relative_hmm and the paper.
