""" Author: Benyamin Meschede-Krasa 
test suite for functions using synthetic data and varying trial numbers"""
import numpy as np
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt

###################################################
######                  TESTS                ######
###################################################

def testDistance(testCI=True):
    from cvDistance import cvDistance
    CIMODE='jackknife'
    TRIAL_NUMS = [5,20]
    N_REPS = 10 #TODO clean 1000
    DISTANCES = np.linspace(0,10,50)
    N_DIM = 100

    distanceEst = np.zeros([len(TRIAL_NUMS),len(DISTANCES),N_REPS])
    distanceEstUnbiased = np.zeros([len(TRIAL_NUMS),len(DISTANCES),N_REPS])

    for t, nTrials in enumerate(TRIAL_NUMS):
        for distIdx, dist in enumerate(DISTANCES):
            nTrials = TRIAL_NUMS[t]
            for n in range(N_REPS):
                data1 = np.random.randn(nTrials,N_DIM)
                data2 = (dist/np.sqrt(N_DIM)) + np.random.randn(nTrials,N_DIM)

                # non cv euclidian distance 
                distanceEst[t,distIdx,n] = np.linalg.norm(np.mean(data1,axis=0)-np.mean(data2,axis=0))

                distanceEstUnbiased[t,distIdx,n] = cvDistance(data1, data2,CIMode=CIMODE)[1] # get euclidean distance, not squared distance
    statName = 'Distance'
    plotTrueVsEstimated( TRIAL_NUMS, distanceEst, distanceEstUnbiased, DISTANCES, statName )

    if testCI:
        # The following samples data from two multivariate distributions whose mean vectors
        # have varying distances from each other. A confidence interval for
        # the distance is estimated using 3 different methods and the
        # coverage is checked. 
        
        ALPHA=0.05
        N_REPS = 100
        N_DIM = 100
        DISTANCES = np.linspace(0,10,3)
        isCovered = np.zeros((len(DISTANCES), N_REPS))

        for distIdx, dist in enumerate(DISTANCES):
            for n in range(N_REPS):
                data1 = np.random.randn(20,N_DIM)
                data2 = (dist/np.sqrt(N_DIM)) + np.random.randn(20,N_DIM)

                [_, stat, CI, _] = cvDistance( data1, data2, subtractMean=False, CIMode='jackknife', CIAlpha=ALPHA)
                isCovered[distIdx,n] = np.logical_and(dist > CI[0,1], dist < CI[1,1])


        plotCICoverage( 'Jackknife', isCovered, DISTANCES )
    


def plotTrueVsEstimated(trialNums, estValuesStandard, estValuesUnbiased, trueValues, statName ):
    """plotter for repititions of cv metric estimates and standard estimates compared to ground truth

    Parameters
    ----------
    trialNums : array ()
        list of number of trials tested, aligns with first dimension of estimates
    estValuesStandard : array (n_trial_nums, n_true_values, n_repitions )
        standard method of estimating value
    estValuesUnbiased : array (n_trial_nums, n_true_values, n_repitions )
        cross-validated method of estimating a value
    trueValues : array (n_true_values)
        difference ground truths tested on
    statName : str
        name of metric/value, e.g. 'Distance'
    """
    f, axes = plt.subplots(1,len(trialNums))
    colors=['r','k']
    for t in range(len(trialNums)):
    
        mn = np.mean(estValuesStandard[t,:,:],axis=1)
        sd = np.std(estValuesStandard[t,:,:],axis=1)

        mn_un = np.mean(estValuesUnbiased[t,:,:],axis=1)
        sd_un = np.std(estValuesUnbiased[t,:,:],axis=1)
        
        axes[t].plot(trueValues, mn, color=colors[0], linewidth=2,label='Standard')
        axes[t].plot(trueValues, mn_un, color=colors[1], linewidth=2,label='Cross-Validated')
        axes[t].plot([0,max(trueValues)],[0,max(trueValues)],'--k',linewidth=2,label='Truth')

        axes[t].fill_between(trueValues, mn-sd, mn+sd, color= colors[0], alpha=0.5)
        axes[t].fill_between(trueValues, mn_un-sd_un, mn_un+sd_un, color= colors[1], alpha=0.5)

        axes[t].set_title(f"{trialNums[t]} Trials")
        axes[t].set_xlabel(f"True {statName}")
        axes[t].set_ylabel(f"Estimated {statName}")

    plt.tight_layout()
    plt.legend()

def plotCICoverage(ciName, coverage, trueValues):
    """plot how often truth was withing the confidence intervals for a given 
    method of quantifying uncertainty

    Parameters
    ----------
    ciName : str
        name of method for quantifying uncertainty e.g. 'jackknife'
    coverage : array (n_reps,n_true_values)
        boolean array of whether truth was covered for n_reps repititions
    trueValues : array (n_true_values,)
        ground truth
    """
    plt.figure()
    for d in range(len(trueValues)):
        CI = proportion_confint(sum(coverage[d,:]),
                                    len(coverage[d,:]),
                                    method='beta')
        p = sum(coverage[d,:])/len(coverage[d,:])
        
        plt.plot(trueValues[d],p,'ko',linewidth=2)
        plt.plot([trueValues[d],trueValues[d]],CI,'k-',linewidth=2)

        plt.xlabel('True Values')
        plt.ylabel('Coverage')
        plt.title(ciName)

        plt.xlim([trueValues[0]-1,trueValues[-1]+1])
        plt.ylim([0,1])
    plt.tight_layout()

if __name__ == "__main__":
    testDistance()
    plt.show()