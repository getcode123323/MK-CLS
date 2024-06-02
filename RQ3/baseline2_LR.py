from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
from imblearn.over_sampling import SVMSMOTE
import copy
from cleanlab.latent_estimation import compute_confident_joint, estimate_latent
from cleanlab.pruning import get_noise_indices
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#clear data
def get_noise(xall_new, label_new, pre_new):
    label_new1=copy.deepcopy(label_new)
    pre_new1 = copy.deepcopy(pre_new)
    xall_new1 = copy.deepcopy(xall_new)
    label_1 = label_new1.ravel()
    y_train2 = label_1.astype(np.int16)
    confident_joint = compute_confident_joint(
        s=y_train2,
        psx=pre_new1,    #P(s = k|x)
        thresholds=None
    )
    py, noise_matrix, inv_noise_matrix = estimate_latent(
            confident_joint=confident_joint,
            s=y_train2,
            py_method='cnt',
            converge_latent_estimates=False
    )
    ordered_label_errors = get_noise_indices(
        s=y_train2,
        psx=pre_new1,
        inverse_noise_matrix=inv_noise_matrix,
        confident_joint=confident_joint,
        prune_method='prune_by_class',
    )
    x_mask = ~ordered_label_errors
    x_pruned = xall_new1[x_mask]
    s_pruned = y_train2[x_mask]
    sample_weight = np.ones(np.shape(s_pruned))
    for k in range(2):
        sample_weight_k = 1.0 / noise_matrix[k][k]
        sample_weight[s_pruned == k] = sample_weight_k
    return x_pruned, s_pruned

#Calculate confidence and perform denoising
def trained_CL(X_train,Y_train):

   #Cross validation to obtain confidence
   sfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
   # Merge four arrays horizontally for cross validation grouping
   X = np.hstack(X_train)
   count=1#Record the number of rounds
   psx = np.zeros((len(Y_train), 2))
   for train_index, test_index in sfolder.split(X, Y_train):
      print("begin!-------",count)
      x_train = X[train_index,:]
      y_train=Y_train[train_index]
      x_test=X[test_index,:]
      #Cross validation obtained the training and testing sets,
      log_reg = LogisticRegression(solver='liblinear')
      log_reg.fit(x_train, y_train)
      psx_cv = log_reg.predict_proba(x_test)
      # Store in the corresponding location in psx[]
      psx[test_index] = psx_cv
      print("finish!-------",count)
      count=count+1
   print("clear begin!!!")
   x_new, y_new = get_noise(X, Y_train, psx)

   #SVMSMOTE
   # SVMSMOTE
   smote = SVMSMOTE(sampling_strategy='auto', svm_estimator=SVC(kernel='poly', degree=4, random_state=3),
                    random_state=3)
   x_new, y_new = smote.fit_resample(x_new, y_new)

   print("orginal train set's size:",len(X))
   print("after clearing train set's size:",len(x_new))
   x_new = np.hsplit(x_new, np.cumsum([arr.shape[1] for arr in X_train])[:-1])
   print("clear finish!!!")
   return x_new,y_new