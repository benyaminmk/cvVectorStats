""" Author: Benyamin Meschede-Krasa 
cross validated confidence intervals for cv metrics """
import os
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

######################
######  PARAMS  ######
######################

##################################################
######                  MAIN                ######
##################################################

def cvJackknifeCI(fullDataStatistic, dataFun, dataTrials, alpha):
    """compute confidence intervals for cv statistic

    Parameters
    ----------
    fullDataStatistic : list
        list of statistics computed from `dataTrials` 
    dataFun : func
        callable function that transforms `dataTrials` to the
        statistic
    dataTrials : array (n_classes, n_trials, n_features)
        list of data from classes used to compute `fullDataStatistic`
    alpha : float
        alpha for confidence interval coverage (e.g. 0.05 for 95%CI)

    Returns
    -------
    CI : array (n_statistics, 2)
        upper and lower bounds for each statistic
    jacks : array (n_folds, n_statistics)
        folds from jackknifing (loo)
    """

    # NOTE: implementation only supports data cells with same numbers of trials unlike original implementation
    nFolds = dataTrials[0].shape[0] # Leave one trial out cross validation
    folds = np.arange(nFolds)
    jacks = np.zeros([nFolds, len(fullDataStatistic)]) 
    for foldIdx in folds:
        deleteTrials = [list(dataTrial) for dataTrial in dataTrials]
        for x in range(len(deleteTrials)):
            deleteTrials[x].pop(foldIdx)
        jacks[foldIdx,:] = dataFun(*deleteTrials)[:2]

    ps = nFolds*np.array(fullDataStatistic) - (nFolds-1)*jacks
    v = np.var(ps,axis=0) 
    
    multiplier = norm.ppf((1-alpha/2), 0, 1)
    CI = np.array([(fullDataStatistic - multiplier*np.sqrt(v/nFolds)), (fullDataStatistic + multiplier*np.sqrt(v/nFolds))])
    return CI, jacks