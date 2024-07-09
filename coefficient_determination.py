import numpy as np 


def mean(data):
    # return np.nanmean(data)
    return np.mean(data)


def total_variation(target):
    y_mean = mean(target)
    return np.sum((target - y_mean)**2)



def Explained_variation(y_pred, target):
    y_mean = mean(target)
    return np.sum((y_pred - y_mean)**2)


def Coefficient_determination(Explained_variation, total_variation):
    return Explained_variation/ total_variation



