import numpy as np
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RepeatedEditedNearestNeighbours

def trained_OSS(X_train,Y_train):
    X = np.hstack(X_train)
    print("clear begin!!!")
    renn = OneSidedSelection()
    x_pruned, s_pruned = renn.fit_sample(X, Y_train)
    print("orginal train set's size:", len(X))
    print("after clearing train set's size:", len(x_pruned))
    x_pruned = np.hsplit(x_pruned, np.cumsum([arr.shape[1] for arr in X_train])[:-1])
    print("clear finish!!!")
    return x_pruned, s_pruned