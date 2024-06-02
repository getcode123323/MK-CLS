import copy

import numpy as np
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RepeatedEditedNearestNeighbours

def get_noise(label_new,xall_new):
    label_new1=copy.deepcopy(label_new)
    xall_new1 = copy.deepcopy(xall_new)
    ordered_label_errors=[]
    count_no=0
    for ii in range(len(label_new1)):
        r_pre1 = np.zeros((len(label_new1), 2))
        for jj in range(len(label_new1)):
            for zz in range(14):
                r_pre1[jj][0]=r_pre1[jj][0]+(xall_new1[ii][zz]-xall_new1[jj][zz])**2
            r_pre1[jj][0] = r_pre1[jj][0]**0.5
            r_pre1[jj][1] = label_new1[jj]
        idex = np.lexsort([r_pre1[:, 0]])
        sorted_data = r_pre1[idex, :]
        countc = 0
        for jj in range(6):
            if(label_new1[ii]!=sorted_data[jj][1]):
                countc = countc+1
        the_1 = (countc-1)/5
        if the_1>=0.6:
            ordered_label_errors.append(False)
            count_no=count_no+1
        else:
            ordered_label_errors.append(True)
        if count_no>len(label_new1)*0.01:
            break;
    for ii in range(len(label_new1)-len(ordered_label_errors)):
        ordered_label_errors.append(True)
    #x_mask = ~ordered_label_errors
    x_pruned = xall_new1[ordered_label_errors]
    # print(label_new_2)
    s_pruned = label_new1[ordered_label_errors]
    return x_pruned,s_pruned


def trained_CLNI(X_train,Y_train):
    X = np.hstack(X_train)
    print("clear begin!!!")
    # rus = RandomUnderSampler(random_state=3)
    # X_resampled, y_resampled = rus.fit_sample(X, Y_train)
    x_pruned, s_pruned=get_noise(Y_train,X)
    print("orginal train set's size:", len(X))
    print("after clearing train set's size:", len(x_pruned))
    x_pruned = np.hsplit(x_pruned, np.cumsum([arr.shape[1] for arr in X_train])[:-1])
    print("clear finish!!!")
    return x_pruned,s_pruned
