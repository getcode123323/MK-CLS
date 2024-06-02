import numpy as np
import warnings
from imblearn.over_sampling import BorderlineSMOTE,SVMSMOTE,ADASYN,KMeansSMOTE
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



#Calculate confidence and perform denoising
def trained_CL(X_train,Y_train):
   X = np.hstack(X_train)
   # SVMSMOTE
   smote = SVMSMOTE(sampling_strategy='auto', svm_estimator=SVC(kernel='poly', degree=4,probability=True,random_state=3),
                    random_state=3)
   x_new, y_new = smote.fit_resample(X, Y_train)
   print("clear brgin!!!")
   print("orginal train set's size:",len(X))
   print("after clearing train set's size:",len(x_new))
   x_new = np.hsplit(x_new, np.cumsum([arr.shape[1] for arr in X_train])[:-1])
   print("clear finish!!!")
   return x_new,y_new