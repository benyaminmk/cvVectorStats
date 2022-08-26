""" Author: Benyamin Meschede-Krasa 
cross validated distance, based on https://github.com/fwillett/cvVectorStats/blob/master/cvDistance.m """
import numpy as np
from cvCI import cvJackknifeCI

def cvDistance(class0,class1,subtractMean=False, CIMode='none',CIAlpha=0.05): #TODO implement CI
    """Estimate the distance between two distributions

    Parameters
    ----------
    class0 : ndarray (nTrials,nFeatures)
        samples from distributions to be compared 
    class1 : _type_
        _description_
    subtractMean : bool, optional
        If subtractMean is true, this will center each vector
        before computing the size of the difference, by default False
    CIMode : str
        method for computing confidence intervals. Currently only 'jackknife'
        is implmented
    CIAlpha : float
        alpha for confidence interval. Default is 0.05 which give the 95%
        confidence interval

    Returns
    -------
    squaredDistance : float
        cross-validated estimate of squared distance between class 1 and 2
    euclideanDistance : float
        cross-validated estimate of euclidean distance between class 1 and 2
    CI : ndarray(2,2)
        confidence intervals for squaredDistance (col 0) and euclideanDistance
        (col 1)
    """
    class0 = np.array(class0)
    class1 = np.array(class1)

    assert class0.shape == class1.shape, "Classes must have same shape, different numebrs of trials not implemented yet" #TODO implement different trial numebr for classes

    nTrials, nFeatures = class0.shape
    squaredDistanceEstimates=np.zeros([nTrials,1])

    for x in range(nTrials):
        bigSetIdx = list(range(nTrials))
        smallSetIndex = bigSetIdx.pop(x)

        meanDiff_bigSet = np.mean(class0[bigSetIdx,:] - class1[bigSetIdx,:],axis=0)
        meanDiff_smallSet = class0[smallSetIndex,:] - class1[smallSetIndex,:]
        if subtractMean:
            squaredDistanceEstimates[x] = np.dot(meanDiff_bigSet-np.mean(meanDiff_bigSet)),(meanDiff_smallSet-np.mean(meanDiff_smallSet).transpose())
        else:
            squaredDistanceEstimates[x] = np.dot(meanDiff_bigSet,meanDiff_smallSet.transpose())
    
    squaredDistance = np.mean(squaredDistanceEstimates)
    euclideanDistance = np.sign(squaredDistance)*np.sqrt(np.abs(squaredDistance))
    
    if CIMode == 'jackknife':
        wrapperFun = lambda x,y : cvDistance(x,y,subtractMean=subtractMean)
        [CI, CIDistribution] = cvJackknifeCI([squaredDistance, euclideanDistance], wrapperFun, [class0, class1], CIAlpha)
    elif CIMode == 'none':
        CI = []
        CIDistribution = []
    else:
        raise ValueError(f"CIMode {CIMode} not implemented or is invalid. select from ['jackknife','none']")

    return squaredDistance, euclideanDistance, CI, CIDistribution 

    


# def getFoldedIdx(obsPerClass, nFolds):
#     nClasses = len(obsPerClass)
#     foldIdxPerClass = {}
#     for c in range(nClasses):
#         minPerFold = obsPerClass[c]//nFolds
#         remainder = obsPerClass[c]-minPerFold*nFolds

#         if remainder>0:
#             currIdx = np.arange(minPerFold+1)
#         else:
#             currIdx = np.arange(minPerFold)

#         for x in range(nFolds):
#             foldIdxPerClass
#     pass