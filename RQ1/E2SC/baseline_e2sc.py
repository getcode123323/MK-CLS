import numpy as np
import warnings
from src.main.python.iSel import e2sc
warnings.filterwarnings('ignore')


#baseline1:E2SC

def trained_E2SC(X_train, Y_train):
    X=np.hstack(X_train)
    selector = e2sc.E2SC()
    selector.select_data(X, Y_train)
    idx = selector.sample_indices_
    x_pruned, s_pruned = X[idx], Y_train[idx]
    print("clear begin!!!")
    print("orginal train set's size:", len(X))
    print("after clearing train set's size:", len(x_pruned))
    x_pruned = np.hsplit(x_pruned, np.cumsum([arr.shape[1] for arr in X_train])[:-1])
    print("clear finish!!!")
    return x_pruned, s_pruned

