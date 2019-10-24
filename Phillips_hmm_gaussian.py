#!/usr/bin/env python3

#Phillips_hmm_gaussian.py --nodev --iterations 1 --clusters_file gaussian_hmm_smoketest_clusters.txt --data_file points.dat --print_params
#Forward/backward maybe correct, do train_model and test

import numpy as np
if not __file__.endswith('_hmm_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = '/u/cs246/data/em/' #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args, train_xs):
    if args.cluster_num:
        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num,2,2))
            for k in range(0,args.cluster_num):
                sigmas[k] = np.cov(train_xs.T)
                #sigmas[k] = np.diagflat(sigmas[k])
        else:
            sigmas = np.zeros((2,2))
            sigmas = np.cov(train_xs.T)
            

        transitions = np.zeros((args.cluster_num,args.cluster_num)) #transitions[i][j] = probability of moving from cluster i to cluster j
        initials = np.zeros(args.cluster_num) #probability for starting in each state
        #TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)
        K = args.cluster_num
        N = (len(train_xs))
        for k in range(0,K):
            initials[k] = 1/K
            for j in range(0,K):
                transitions[k,j] = 1/K
        for k in range(0,K):
            j = np.random.randint(0,N)
            mus[k] = train_xs[j]
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)


    #TODO: Do whatever you want to pack mus, sigmas, initals, and transitions into the model variable (just a tuple, or a class, etc.)
    model = mus, sigmas, initials, transitions
    return model

def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log
    mus, sigmas, initials, transitions = extract_parameters(model)
    N = (len(data))
    K = (args.cluster_num)
    alphas = np.zeros((N,K))
    log_likelihood = 0.0


    # for t = 0
    if args.tied:
        for k in range(K):
            alphas[0,k] = initials[k]*multivariate_normal(mean=mus[k], cov=sigmas).pdf(data[0])
    else:
        for k in range(K):
            alphas[0,k] = initials[k]*multivariate_normal(mean=mus[k], cov=sigmas[k]).pdf(data[0])
    s = np.sum(alphas[0,:])
    log_likelihood += log(s)
    alphas[0,:] = alphas[0,:]/s

    #compute alphas
    for t in range(1,N):
        for k in range(K):
            for k2 in range(K):
                alphas[t,k] +=alphas[t-1,k2]*transitions[k2,k]*multivariate_normal(mean=mus[k], cov=sigmas[k]).pdf(data[t]) #multiplying by emission probability
        s = np.sum(alphas[t,:])
        alphas[t,:] = alphas[t,:]/s
        log_likelihood += log(s)
    #TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    #NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment
    #log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0,
    #and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything
    #different than what's in the notes). This was discussed in class on April 3rd.
    log_likelihood = log_likelihood/N
    return alphas, log_likelihood

def backward(model, data, args):
    from scipy.stats import multivariate_normal
    mus, sigmas, initials, transitions = extract_parameters(model)
    N = (len(data))
    K = (args.cluster_num)
    betas = np.zeros((N,K))
    normalizer = np.zeros(N)
    
    # for t = 0
    for k in range(0, K):
        betas[N-1,k] = 1
    betasOG = np.copy(betas)
    for k in range(0, K):
        betas[N-1, k] = betas[N-1, k] / np.sum(betasOG[N-1, :])    
    #compute betas
    for t in range(N-2,-1,-1):
        for k in range(0, K):
            for k2 in range(0, K):
                if args.tied:
                    betas[t,k] += betas[t+1,k2]*transitions[k,k2]*multivariate_normal(mean=mus[k2], cov=sigmas).pdf(data[t+1])
                else:
                    betas[t,k] += betas[t+1,k2]*transitions[k,k2]*multivariate_normal(mean=mus[k2], cov=sigmas[k2]).pdf(data[t+1]) #multiplying by emission probability
        s = np.sum(betas[t,:])
        betas[t,:] = betas[t,:]/s
    return betas

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    
    #inintialization
    N = len(train_xs)
    K = args.cluster_num
    mus, sigmas, initials, transitions = extract_parameters(model)
    emissions = np.zeros((N,K))
    for i in range (0, args.iterations):
        alphas, ll = forward(model, train_xs, args)
        betas = backward(model, train_xs, args)
        xi = np.zeros((N,K,K))
        gamma = np.zeros((N,K))
        for t in range(0,N):
            for k in range(0, K):
                normalizeG = 0
                for ksum in range(0,K):
                    normalizeG += alphas[t,ksum] * betas[t,ksum]
                #normalizeG = (np.sum(alphas[t,:] * betas[t,:], axis=1)) #check
                gamma[t,k] = alphas[t,k]*betas[t,k]/normalizeG
                if not (t==0):
                    for k2 in range(0, K):
                        normalize = 0
                        for i in range(0, K):
                            for j in range (0, K):
                                if args.tied:
                                    P_xt_givenJ = multivariate_normal(mean=mus[j], cov=sigmas).pdf(train_xs[t])
                                else:
                                    P_xt_givenJ = multivariate_normal(mean=mus[j], cov=sigmas[j]).pdf(train_xs[t])
                                normalize += alphas[t-1, i]*betas[t,j]*transitions[i,j]*P_xt_givenJ
                        if args.tied:
                            emissionTK2 = multivariate_normal(mean=mus[k2], cov=sigmas).pdf(train_xs[t])
                        else:
                            emissionTK2 = multivariate_normal(mean=mus[k2], cov=sigmas[k2]).pdf(train_xs[t])
                        xi[t, k, k2] = alphas[t-1,k]*betas[t,k2]*transitions[k,k2] * emissionTK2 / normalize
        #M - STEP
        for k in range(0, K):
            initials[k] = gamma[0,k]
            for k2 in range(0,K):
                xiSum = 0
                normalizer = 0
                transitions[k,k2] = np.sum(xi[:,k,k2])/np.sum(gamma[:,k])
                
        for k in range(len(initials)):
            num = np.zeros((1,2))
            norm = 0
            for n in range(train_xs.shape[0]):
                num += gamma[n,k]*train_xs[n,:]
                norm += gamma[n,k]
                mus[k,:] = num/norm
            if not args.tied:
                for k in range(len(initials)):
                    num = np.zeros((2,2))
                    norm = 0
                    for n in range(train_xs.shape[0]):
                        num += gamma[n,k]*np.matmul(np.reshape(train_xs[n],(2,1))-np.reshape(mus[k,:],(2,1)),np.transpose(np.reshape(train_xs[n],(2,1))-np.reshape(mus[k,:],(2,1))))
                        norm += gamma[n,k]
                        sigmas[k,:,:] = num/norm
            else:
                norm = train_xs.shape[0]
                num = np.zeros((2,2))
                for k in range(len(initials)):
                    for n in range(train_xs.shape[0]):
                        num += gamma[n,k]*np.matmul(np.reshape(train_xs[n],(2,1))-np.reshape(mus[k,:],(2,1)),np.transpose(np.reshape(train_xs[n],(2,1))-np.reshape(mus[k,:],(2,1))))
                    sigmas = num/norm

    model2 = mus, sigmas, initials, transitions
    return model2

def average_log_likelihood(model, data, args):
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    #NOTE: yes, this is very simple, because you did most of the work in the forward function above
    ll = 0.0
    alphas, ll = forward(model, data, args)
    return ll

def extract_parameters(model):
    #TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    mus = model[0]
    sigmas = model[1]
    initials = model[2]
    transitions = model[3]
    return mus, sigmas, initials, transitions

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args, train_xs)
    model = train_model(model, train_xs, dev_xs, args)
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    mus, sigmas, initials, transitions = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))
    
##        ## GRAPHING CODE
##    import matplotlib.pyplot as plt
##    ll_train = np.zeros(10)
##    ll_dev = np.zeros(10)
##    count = np.zeros(10)
##    args.cluster_num = 2
##    count[0] = float('nan')            
##    for i in range(1,10):
##        print(i)
##
##        count[i] = i
##        args.iterations = i
##        modelx = init_model(args, train_xs)
##        modelx = train_model(modelx, train_xs, dev_xs, args)
##        ll_train[i] = average_log_likelihood(modelx, train_xs, args)
##        ll_dev[i] = average_log_likelihood(modelx, dev_xs, args)
##
##    fig, ax = plt.subplots()
##    ax.plot(count, ll_dev, label = 'Dev Data')
##    ax.plot(count, ll_train, label = 'Training Data')
##
##
##    plt.legend()
##    ax.set(xlabel='Iterations', ylabel='Log-Likelihood',
##           title='Iterations vs Log-Likelihood')
##    ax.grid()
##
##    fig.savefig("Log-Likelihood_vs_Iterations_(with_args_tied_3_clusters).png")
##    plt.show()
##    args.tied == True
##    ll_train = np.zeros(10)
##    ll_dev = np.zeros(10)
##    count = np.zeros(10)
##    args.iterations = 2
##    count[0] = float('nan')
##
##    for i in range(1,6):
##        print(i)
##        count[i] = i
##        args.cluster_num = i
##        modelx = init_model(args, train_xs)
##        modelx = train_model(modelx, train_xs, dev_xs, args)
##        ll_train[i] = average_log_likelihood(modelx, train_xs, args)
##        ll_dev[i] = average_log_likelihood(modelx, dev_xs, args)
##        ###CODE FOR PLOTTING
##    fig, ax = plt.subplots()
##    ax.plot(count, ll_dev, label = 'Dev Data')
##    ax.plot(count, ll_train, label = 'Training Data')
##
##
##    plt.legend()
##    ax.set(xlabel='Num_Clusters', ylabel='Log-Likelihood',
##           title='Num_Clusters vs Log-Likelihood at 4 iterations')
##    ax.grid()
##    if args.tied:
##        fig.savefig("Log-Likelihood_vs_Num_Clusters (WITH ARGS.TIED).png")
##    else:
##        fig.savefig("Log-Likelihood_vs_Num_Clusters (WITHOUT ARGS.TIED).png")
##    plt.show()
if __name__ == '__main__':
    main()
