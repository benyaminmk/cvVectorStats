""" Author: Benyamin Meschede-Krasa 
test suite for functions using synthetic data """
import os
import numpy as np
from scipy.stats import norm as normfit
import matplotlib.pyplot as plt

######################
######  PARAMS  ######
######################
TRIAL_NUMS = [5,20]
N_REPS = 1000
DISTANCES = np.linspace(0,10,50)
N_DIM = 100

#############################
######  PRECONDITIONS  ######
#############################



##################################################
######                  MAIN                ######
##################################################

def testDistance():
    from cvDistance import cvDistance

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

                distanceEstUnbiased[t,distIdx,n] = cvDistance(data1, data2)[1] # get euclidean distance, not squared distance
    statName = 'Distance'
    plotTrueVsEstimated( TRIAL_NUMS, distanceEst, distanceEstUnbiased, DISTANCES, statName )


def plotTrueVsEstimated(trialNums, estValuesStandard, estValuesUnbiased, trueValues, statName ):


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

if __name__ == "__main__":
    testDistance()
    plt.show()