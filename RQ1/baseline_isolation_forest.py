import numpy as np
from sklearn.ensemble import IsolationForest

#baseline1:孤立森林
def trained_IS(X_train,Y_train):
    X = np.hstack(X_train)
    print("clear begin!!!")
    iso = IsolationForest(random_state=0).fit(X)
    x_array = iso.predict(X)
    count = 0
    de1 = []
    for j in range(len(x_array)):
        if x_array[j] == -1:
            de1.append(j)
            count +=1
    x_pruned = np.delete(X,de1,axis=0)
    s_pruned = np.delete(Y_train,de1,axis=0)
    print("delete number:",count)
    print("orginal train set's size:", len(X))
    print("after clearing train set's size:", len(x_pruned))
    x_pruned = np.hsplit(x_pruned, np.cumsum([arr.shape[1] for arr in X_train])[:-1])
    print("clear finish!!!")
    return x_pruned,s_pruned