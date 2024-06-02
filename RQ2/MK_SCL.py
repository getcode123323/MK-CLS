import sys
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
import copy
from imblearn.over_sampling import SVMSMOTE
from cleanlab.latent_estimation import compute_confident_joint, estimate_latent
from cleanlab.pruning import get_noise_indices
from functools import reduce

from sklearn.svm import SVC

from fast_rvm import RVC

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
bestWeights_pinch_FC = [[0.520562327523,0.450165600186,0.517707144896,0.519529102662],
              [0.498936653137,0.523008918762,0.501436042786,0.500075340271],
              [0.469415709442,0.515048689362,0.511442442579,0.516870836018],
              [0.502043418538,0.474704736362,0.499832878419,0.514874453601]]
bestWeights_pinch_HC = [[0.528430415572,0.498754573763,0.488429150967,0.508001702332],
              [0.512488961461,0.512493287813,0.47500664922,0.51249732488],
              [0.5,0.50625,0.5125,0.5]]
bestWeights_maalej_FC = [[0.495767831802,0.532848119736,0.474793148041,0.510634183884],
              [0.5,0.5,0.5125,0.5],
              [0.490633381622,0.50936897358,0.493754561659,0.52187598308],
              [0.504321289062,0.498461914063,0.501391601562,0.50400390625]]
bestWeights_maalej_HC = [[0.489154922733,0.503873537464,0.522352652975,0.521333297772],
              [0.512513205649,0.474998374095,0.512530606898,0.512479026884],
              [0.504959080754,0.382713237438,-0.00119830543425,0.892282387929]]

#Data processing of the original paper
def getGram(dataRep, thetas, dataRep2=None, hasIntercept=False):
    Ks = []
    for x, y in enumerate(dataRep):
        if (dataRep2 is None):
            tem = (np.dot(y, y.transpose()) * thetas[x])
            if hasIntercept:
                Ks.append(np.concatenate((np.ones([tem.shape[0], 1]), tem), 1))
            else:
                Ks.append(tem)
        else:
            tem = (np.dot(y, dataRep2[x].transpose()) * thetas[x])
            if hasIntercept:
                Ks.append(np.concatenate((np.ones([tem.shape[0], 1]), tem), 1))
            else:
                Ks.append(tem)
    # lambda arguments:expression
    return reduce(lambda x, y: x + y, Ks), Ks

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
        prune_method='prune_by_noise_rate',
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
def trained_CL(X_train,Y_train,task,dataset,targetlabel,method):
   if task=="FC":
      targetLabels = ["informative", "has_feature_request", "has_bug_report", "has_user_exp"]
      if targetlabel in targetLabels:
            index = targetLabels.index(targetlabel)
      else:
            print(targetlabel, "not found in the list.")
            sys.exit()
      if dataset == "pinch":
           weights=bestWeights_pinch_FC
      if dataset =="maalej":
           weights=bestWeights_maalej_FC
   else:#task=="HC"
       targetLabels = ["has_feature_request", "has_bug_report", "has_user_exp"]
       if targetlabel in targetLabels:
           index = targetLabels.index(targetlabel)
       else:
           print(targetlabel, "not found in the list.")
           sys.exit()
       if dataset == "pinch":
          weights=bestWeights_pinch_HC
       if dataset == "maalej":
          weights=bestWeights_maalej_HC
   #Cross validation to obtain confidence
   sfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
   # Merge four arrays horizontally for cross validation grouping
   
   X = np.hstack(X_train)
   # SVMSMOTE
   smote = SVMSMOTE(sampling_strategy='auto', svm_estimator=SVC(kernel='poly', degree=4), random_state=3)
   X, Y_train = smote.fit_resample(X, Y_train)

   count=1#Record the number of rounds
   psx = np.zeros((len(Y_train), 2))
   for train_index, test_index in sfolder.split(X, Y_train):
      print("begin!-------",count)
      x_train = X[train_index,:]
      y_train=Y_train[train_index]
      x_test=X[test_index,:]
      #Cross validation obtained the training and testing sets,
      # converting them into a recognizable form for the model
      x_train = np.hsplit(x_train, np.cumsum([arr.shape[1] for arr in X_train])[:-1])
      x_test = np.hsplit(x_test, np.cumsum([arr.shape[1] for arr in X_train])[:-1])
      K_train, Ks_train = getGram(x_train, weights[index])
      model = RVC(kernel=K_train, gamma=1.0, n_iter=300, verbose=False, tol=0.01, fit_intercept=True, given_K=True)
      model.fit(K_train, y_train)
      actives = model.relevant_
      reVecs = [x[actives] for x in x_train]
      test_rvs, kss = getGram(x_test, weights[index], dataRep2=reVecs)
      psx_cx=model.predict_proba(test_rvs,given_K=True)
      # Store in the corresponding location in psx[]
      psx[test_index] = psx_cx
      print("finish!-------",count)
      count=count+1
   print("clear begin!!!")
   x_new, y_new = get_noise(X, Y_train, psx)
   print("orginal train set's size:",len(X))
   print("after clearing train set's size:",len(x_new))
   x_new = np.hsplit(x_new, np.cumsum([arr.shape[1] for arr in X_train])[:-1])
   print("clear finish!!!")
   return x_new,y_new