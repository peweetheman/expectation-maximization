#!/usr/bin/env python3
import numpy as np

if not __file__.endswith('_em_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
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


def init_model(args, train_xs, train_ys):
    if args.cluster_num:
        K= args.cluster_num
        lambdas = np.zeros(args.cluster_num)
        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num,2,2))


            for k in range(0,args.cluster_num):
                sigmas[k] = np.cov(train_xs.T)
        else:
            sigmas = np.zeros((2,2))
            sigmas = np.cov(train_xs.T)
            
        for k in range(0,K):
            lambdas[k] = 1/K
        for k in range(0,K):
            j = np.random.randint(0,len(train_xs))
            mus[k] = train_xs[j]
    else: 
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)
    #TODO: do whatever you want to pack the lambdas, mus, and sigmas into the model variable (just a tuple, or a class, etc.)
    #NOTE: if args.tied was provided, sigmas will have a different shape
    model = (lambdas, mus, sigmas)
    return model

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #NOTE: you can use multivariate_normal like this:
    #probability_of_xn_given_mu_and_sigma = multivariate_normal(mean=mu, cov=sigma).pdf(xn)
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    lambdas_prime, mus_prime, sigmas_prime = extract_parameters(model)
    K = len(lambdas_prime)
    N = len(train_xs)
    probability_of_Xn = np.zeros(N)
    probability_of_Xn_given_Zk = np.zeros((N,K))
    probability_of_Zk_given_Xn = np.zeros((K,N))
    if not args.tied:
        for i in range(args.iterations):
            lambdas, mus, sigmas = lambdas_prime, mus_prime, sigmas_prime
            lambdas_prime, mus_prime, sigmas_prime = np.zeros(lambdas.shape), np.zeros(mus.shape), np.zeros(sigmas.shape)
            normalizer = np.zeros(K)
            probability_of_Xn = np.zeros(probability_of_Xn.shape)
            #E-STEP
            for n in range(N):
                for k in range(K):
                    probability_of_Xn_given_Zk[n,k] = multivariate_normal(mean=mus[k], cov=sigmas[k]).pdf(train_xs[n])
                    probability_of_Xn[n] += lambdas[k]*probability_of_Xn_given_Zk[n,k]
            for k in range(K):
                for n in range(N):
                    #if(probability_of_Xn[n] == 0):
                        #print(args.iterations)
                    probability_of_Zk_given_Xn[k,n] = (lambdas[k] * probability_of_Xn_given_Zk[n,k]) / probability_of_Xn[n]
                    normalizer[k]+=probability_of_Zk_given_Xn[k,n]
            #M-STEP
            for k in range(K):
                for n in range(N):
                    lambdas_prime[k] += probability_of_Zk_given_Xn[k,n]/N
                    mus_prime[k] += probability_of_Zk_given_Xn[k,n] * train_xs[n]/normalizer[k]
            normalizer = np.zeros(K)
            probability_of_Xn = np.zeros(probability_of_Xn.shape)
            for n in range(N):
                for k in range(K):
                    probability_of_Xn_given_Zk[n,k] = multivariate_normal(mean=mus[k], cov=sigmas[k]).pdf(train_xs[n])
                    probability_of_Xn[n] += lambdas[k]*probability_of_Xn_given_Zk[n,k]
            for k in range(K):
                for n in range(N):
                    probability_of_Zk_given_Xn[k,n] = (lambdas[k] * probability_of_Xn_given_Zk[n,k]) / probability_of_Xn[n]
                    normalizer[k]+=probability_of_Zk_given_Xn[k,n]
            for k in range(K):
                for n in range(N):
                    sigmas_prime[k] +=  probability_of_Zk_given_Xn[k,n]*(train_xs[n].reshape(2,1)-mus_prime[k].reshape(2,1))*(train_xs[n].reshape(1,2)-mus_prime[k].reshape(1,2)) / normalizer[k]


    else:
            lambdas, mus, sigmas = lambdas_prime, mus_prime, sigmas_prime
            lambdas_prime, mus_prime, sigmas_prime = np.zeros(lambdas.shape), np.zeros(mus.shape), np.zeros(sigmas.shape)
            normalizer = np.zeros(K)
            probability_of_Xn = np.zeros(probability_of_Xn.shape)
            #E-STEP
            for n in range(N):
                for k in range(K):
                    probability_of_Xn_given_Zk[n,k] = multivariate_normal(mean=mus[k], cov=sigmas).pdf(train_xs[n])
                    probability_of_Xn[n] += lambdas[k]*probability_of_Xn_given_Zk[n,k]
            for k in range(K):
                for n in range(N):
                    if(probability_of_Xn[n] == 0):
                        print(args.iterations)
                    probability_of_Zk_given_Xn[k,n] = (lambdas[k] * probability_of_Xn_given_Zk[n,k]) / probability_of_Xn[n]
                    normalizer[k]+=probability_of_Zk_given_Xn[k,n]
            #M-STEP
            for k in range(K):
                for n in range(N):
                    lambdas_prime[k] += probability_of_Zk_given_Xn[k,n]/N
                    mus_prime[k] += probability_of_Zk_given_Xn[k,n] * train_xs[n]/normalizer[k]
            normalizer = np.zeros(K)
            probability_of_Xn = np.zeros(probability_of_Xn.shape)
            for n in range(N):
                for k in range(K):
                    probability_of_Xn_given_Zk[n,k] = multivariate_normal(mean=mus[k], cov=sigmas).pdf(train_xs[n])
                    probability_of_Xn[n] += lambdas[k]*probability_of_Xn_given_Zk[n,k]
            for k in range(K):
                for n in range(N):
                    probability_of_Zk_given_Xn[k,n] = (lambdas[k] * probability_of_Xn_given_Zk[n,k]) / probability_of_Xn[n]
                    normalizer[k]+=probability_of_Zk_given_Xn[k,n]
            for k in range(K):
                for n in range(N):
                    sigmas_prime +=  probability_of_Zk_given_Xn[k,n]*(train_xs[n].reshape(2,1)-mus_prime[k].reshape(2,1))*(train_xs[n].reshape(1,2)-mus_prime[k].reshape(1,2)) / N
        
    model = (lambdas_prime, mus_prime, sigmas_prime)
    if(args.nodev==False):
        maxLL=-10000
        ll_train = np.zeros(30)
        ll_dev = np.zeros(30)
        count = np.zeros(30)
        args.iterations = 10
        best_clusters = 1
        print("finding good number of clusters....")
        for i in range(1,15):
            count[i] = i
            args.cluster_num = i
            modelx = init_model(args, train_xs, dev_xs)
            modelx = train_model(modelx, train_xs, dev_xs, args)
            ll_train[i] = average_log_likelihood(modelx, train_xs, args)
            ll_dev[i] = average_log_likelihood(modelx, dev_xs, args)
            if(ll_dev[i]>maxLL):
                maxLL = ll_dev[i]
                best_clusters = i
        print("choose this many clusters: ", best_clusters)
        print("finding good number of iterations...")
        maxLL=-10000
        ll_train = np.zeros(30)
        ll_dev = np.zeros(30)
        count = np.zeros(30)
        args.cluster_num = i
        best_iterations = 1
        for i in range(1,10):
            count[i] = i
            args.iterations = i
            modelx = init_model(args, train_xs, dev_xs)
            modelx = train_model(modelx, train_xs, dev_xs, args)
            ll_train[i] = average_log_likelihood(modelx, train_xs, args)
            ll_dev[i] = average_log_likelihood(modelx, dev_xs, args)
            if(ll_dev[i]>maxLL):
                maxLL = ll_dev[i]
                best_iterations = i
        print("best number of iterations :", best_iterations)
        print("final max ll: ", maxLL)
        modelx = init_model(args, train_xs, dev_xs)
        modelx = train_model(modelx, train_xs, dev_xs, args)
        return modelx
    return model

def average_log_likelihood(model, data, args):
    from math import log
    from scipy.stats import multivariate_normal
    lambdas, mus, sigmas = extract_parameters(model)
    ll = 0.0
    for n in range(0,len(data)):
        p_xn = 0
        for k in range(0,len(lambdas)):
            if not args.tied:
                p_xn += (lambdas[k]*multivariate_normal(mean=mus[k], cov=sigmas[k]).pdf(data[n]))
            else:
                p_xn += (lambdas[k]*multivariate_normal(mean=mus[k], cov=sigmas).pdf(data[n]))
        if(p_xn>0):
            ll += log(p_xn)
    return ll/len(data)

def extract_parameters(model):
    #TODO: extract lambdas, mus, and sigmas from the model and return them (same type and shape as in init_model)
    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    return lambdas, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
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
    model = init_model(args, train_xs, dev_xs)
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))



           
            ## GRAPHING CODE
            #import matplotlib.pyplot as plt
##            ll_train = np.zeros(30)
##            ll_dev = np.zeros(30)
##            count = np.zeros(30)
##            args.cluster_num = 5
##            args.tied = True
##            count[0] = float('nan')            
##            for i in range(1,30):
##                count[i] = i
##                args.iterations = i
##                modelx = init_model(args, train_xs, dev_xs)
##                modelx = train_model(modelx, train_xs, dev_xs, args)
##                ll_train[i] = average_log_likelihood(modelx, train_xs, args)
##                ll_dev[i] = average_log_likelihood(modelx, dev_xs, args)
##    

           ###CODE FOR PLOTTING
##            fig, ax = plt.subplots()
##            ax.plot(count, ll_dev, label = 'Dev Data')
##            ax.plot(count, ll_train, label = 'Training Data')
##
##
##            plt.legend()
##            ax.set(xlabel='Iterations', ylabel='Log-Likelihood',
##                   title='Iterations vs Log-Likelihood')
##            ax.grid()
##
##            fig.savefig("Log-Likelihood_vs_Iterations_(with_args_tied_5_clusters).png")
##            plt.show()
##            ll_train = np.zeros(30)
##            ll_dev = np.zeros(30)
##            count = np.zeros(30)
##            args.iterations = 10
##            count[0] = float('nan')
##            for i in range(1,30):
##                count[i] = i
##                args.cluster_num = i
##                modelx = init_model(args, train_xs, dev_xs)
##                modelx = train_model(modelx, train_xs, dev_xs, args)
##                ll_train[i] = average_log_likelihood(modelx, train_xs, args)
##                ll_dev[i] = average_log_likelihood(modelx, dev_xs, args)
##                ###CODE FOR PLOTTING
##            fig, ax = plt.subplots()
##            ax.plot(count, ll_dev, label = 'Dev Data')
##            ax.plot(count, ll_train, label = 'Training Data')
##
##
##            plt.legend()
##            ax.set(xlabel='Num_Clusters', ylabel='Log-Likelihood',
##                   title='Num_Clusters vs Log-Likelihood at 10 iterations')
##            ax.grid()
##            if args.tied:
##                fig.savefig("Log-Likelihood_vs_Num_Clusters (WITH ARGS.TIED).png")
##            else:
##                fig.savefig("Log-Likelihood_vs_Num_Clusters (WITHOUT ARGS.TIED).png")
##            plt.show()

if __name__ == '__main__':
    main()
