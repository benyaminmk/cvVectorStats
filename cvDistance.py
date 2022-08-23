""" Author: Benyamin Meschede-Krasa 
cross validated distance, based on https://github.com/fwillett/cvVectorStats/blob/master/cvDistance.m """
import numpy as np

def cvDistance(class1,class2,subtractMean=False):# ,CIMode='none',CIAlpha=0.05,CIResamples=1000): #TODO implement CI
    assert class1.shape == class2.shape, "Classes must have same shape, different numebr sof trials not implemented yet" #TODO implement different trial numebr for classes

    nTrials, nFeatures = class1.shape
    squaredDistanceEstimates=np.zeros([nTrials,1])

    for x in range(nTrials):
        bigSetIdx = list(range(nTrials))
        smallSetIndex = bigSetIdx.pop(x)

        meanDiff_bigSet = np.mean(class1[bigSetIdx,:] - class2[bigSetIdx,:],axis=0)
        meanDiff_smallSet = class1[smallSetIndex,:] - class2[smallSetIndex,:]
        if subtractMean:
            squaredDistanceEstimates[x] = np.dot(meanDiff_bigSet-np.mean(meanDiff_bigSet)),(meanDiff_smallSet-np.mean(meanDiff_smallSet).transpose()) #check that this results in a single float for dot ptoduct
        else:
            squaredDistanceEstimates[x] = np.dot(meanDiff_bigSet,meanDiff_smallSet.transpose())
    
    squaredDistance = np.mean(squaredDistanceEstimates)
    euclideanDistance = np.sign(squaredDistance)*np.sqrt(np.abs(squaredDistance))
    return squaredDistance, euclideanDistance

    



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