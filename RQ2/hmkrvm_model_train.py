#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anonymous
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lib import evaluateFC
from lib import evaluateHC
from lib import calculateMacroF1andRecall
from lib import calculateMicroF1andRecall
from lib import getAUCperClassFlat
from lib import getFCInformativeResult
from lib import runClassifier
from lib import saveObject
from lib import getClassificationMetrics
from lib import get_AUC_Prec_Recall
from lib import get_AUC_ROC
from lib import getKernelsData
from sklearn import preprocessing
#from skbayes.rvm_ard_models import RVC
from functools import reduce
#simplex optimizatino
from scipy.optimize import minimize
from skbayes.rvm_ard_models.fast_rvm import RVC


#Here are all the experimental models of RQ2 (MK-CLS, MK-CL, MK-S, MK-SCL)
#Choose which model to import here, and then apply the model at line 272
from MK-CLS import trained_CL
# from MK-S import trained_CL
# from MK-SCL import trained_CL
# from MK-CL import trained_CL


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
#constructs kernels
def getGram(dataRep,thetas,dataRep2=None,hasIntercept=False):
    Ks=[]
    for x,y in enumerate(dataRep):
        if (dataRep2 is None):
            tem=(np.dot(y,y.transpose())*thetas[x])
            if hasIntercept:
                Ks.append(np.concatenate((np.ones([tem.shape[0],1]),tem),1))
            else:
                Ks.append(tem)
        else:
            tem=(np.dot(y,dataRep2[x].transpose())*thetas[x])
            if hasIntercept:
                Ks.append(np.concatenate((np.ones([tem.shape[0],1]),tem),1))
            else:
                Ks.append(tem)
    #lambda arguments:expression
    return reduce(lambda x,y:x+y,Ks),Ks


#train a model with given thetas
def modelTrain(thetas,X_train,Y_train):
    print("Model Training with weights=", thetas)
    #apply representation weight
    K_train,Ks_train=getGram(X_train,thetas)
    model = RVC( kernel = K_train, gamma = 1.0,n_iter = 300,verbose = False,tol=0.01,fit_intercept=True,given_K=True)
    model.fit(K_train,Y_train)
    return model


#test a trained model with given thetas
def modelPredict(model,thetas,X_train, X_test,Y_test):
    print("Model Predicting with weights=", thetas)
    actives=model.relevant_
    reVecs=[x[actives] for x in X_train]
    test_rvs,kss=getGram(X_test,thetas,dataRep2=reVecs)
    testPredict=model.predict_proba(test_rvs,given_K=True)
    a=np.argmax(testPredict,axis=1)
    acc_test, precision_test, recall_test, f1score_test, cm = getClassificationMetrics(Y_test, a)
    #evaluating result
    auc_roc = get_AUC_ROC(y_true=Y_test,probas_pred= testPredict[:,1],pos_label=1, showPlot = False)
    auc_pr = get_AUC_Prec_Recall(y_true=Y_test,probas_pred= testPredict[:,1],pos_label=1, showPlot = False)
    print("auc_pr=", auc_pr, ", auc_roc=", auc_roc, "f1score_test=", f1score_test, ", recall=",recall_test)
    return auc_pr,auc_roc, f1score_test, recall_test, precision_test, actives, cm, a, testPredict, thetas


#for convience, this function is used to run RVM multiple times, and return best model
#this function is used to overcome randomness that may occur with RVM due to selecting different RVs
def evaluteWeightsOnTest(theta, X_train, X_test, Y_train, Y_test, attempts=10):
    print("\n\n##Working on weights", theta)
    indexOfBest = 0;
    bestScore = 0;
    result = []
    for j in range(0, attempts):
        print("-Attempt", j)
        #traing model
        model = modelTrain(theta,X_train,Y_train)
        #evaluate model
        auc_pr,auc_roc, f1score_test, recall_test, precision_test, actives, cm, predictions, probs, theta= modelPredict(model,theta,X_train, X_test,Y_test)
        temp = [model, auc_pr,auc_roc, f1score_test, recall_test, precision_test, cm, predictions, probs, theta];
        #compare to previous
        if auc_pr>bestScore:
            indexOfBest = j
            bestScore = auc_pr
            
        result.append(temp)
    #save 
    print("sending best with index", indexOfBest)
    return result[indexOfBest]


def runSciptMinimize(ycols, folder, task, method, options,logFolder="Results_bo", initWeights=[0.5,0.5,0.5,0.5]):
    initWeights = np.ndarray(shape=(len(initWeights),), dtype=float, buffer=np.array(initWeights))
    #used for simplex
    def objFun2(thetas): 
        K,Ks=getGram(X_k,thetas)
        model = RVC( kernel = K, gamma = 1.0,n_iter = 100,verbose = False,tol=0.01,fit_intercept=True,given_K=True)
        model.fit(K,T)
        #extracting weights and alpha, and then adding the *learned* intercept colummn
        W=model.coef_.transpose()
        intercept=model.intercept_
        W=np.concatenate((W,[intercept]),0)
        As= model.lambda_[0]
        As[As<0.00001]=0.00001
        A=np.diag(As)
        A_inv = np.linalg.inv(A)
        
        #get model predictions
        actives=model.relevant_
        reVecs = [x[tuple(actives)] for x in X_k]
        #recreate input using relevant vectors and thetas
        train_rvs,kss=getGram(X_k,thetas,dataRep2=reVecs)
        #get predictions
        Y=model.predict_proba(train_rvs,given_K=True)[:,1]
        
        #Calculate likelihood
        K=np.concatenate((np.ones([K.shape[0],1]),K),1)
        B=np.diag((Y*(1-Y))+0.01)
        N = B.shape[0]
        T_hat=np.dot(K,W)+np.dot(np.linalg.inv(B),np.reshape((T-Y),((T-Y).shape[0],1)))
        KAK = np.linalg.multi_dot([K,A_inv,K.transpose()])
        C=B+KAK
        np.fill_diagonal(C, np.diag(C) + 0.00001)
        Cinv = np.linalg.inv(C)
        tct = np.dot(T_hat.transpose(), np.dot(Cinv,T_hat))
        li = (N * np.log(2*3.14)) + np.linalg.slogdet(C)[1] + tct
        li = - 0.5 * li
        #we will use a maximization method, so we need to invert the sign
        li = - li
        return li[0][0]
    
    for targetLabel in ycols:
        print("Dataset", folder)
        print("-Working on label", targetLabel)
        #read data
        X_train, X_test, Y_train, Y_test, _, _, _,_ = getKernelsData(targetLabel,folder, task, logFolder, method)
        #current values  (used inside objFun2)       
        X_k = X_train
        T = Y_train
        result = minimize(objFun2, initWeights, method=method ,options=options)
        bestWeight = result.x
        exitFlag = result.success
        exitMessage = result.message
        bestLikelihood = objFun2(bestWeight)
        print("Best Weights [wordEmbedding, meta, TFIDF, LDA]=",
              "{:.10f}, {:.10f}, {:.10f}, {:.10f}".format(*bestWeight))
        # print("Best Weights [wordEmbedding, meta, TFIDF, LDA]=", bestWeight)
        print("With Likelihood=", bestLikelihood)
        print("With exitFlag=", exitFlag, ", exitMessage=", exitMessage)
       #evaluate model
        result = evaluteWeightsOnTest(bestWeight, X_train, X_test, Y_train, Y_test, attempts=10)
        #save model
        filepath = "../"+logFolder+"/"+folder+"_"+method+"_bestModels_"+task+"_"+targetLabel+".file"
        saveObject(result,filepath)
        #save result
        with open('../'+logFolder+'/result_weights/'+folder+"_"+task+"_"+targetLabel+ "_" + method+".txt", 'a') as the_file:
            # the_file.write("\n-Best Weights= %s,%s,%s,%s"% (bestWeight[0],bestWeight[1], bestWeight[2], bestWeight[3]))
            the_file.write("\n-Best Weights= {:.10f},{:.10f},{:.10f},{:.10f}".format(
                bestWeight[0], bestWeight[1], bestWeight[2], bestWeight[3]))
            the_file.write("\n-exitFlag=%s, exitMessage=%s"%(exitFlag, exitMessage))
            the_file.write("\n-Performance on Testing:")
            the_file.write("\nAUC_PR=%s\n"% result[1])
            the_file.write("\nAUC_ROC=%s\n"% result[2])
            the_file.write("\nF1=%s\n"% result[3])
            the_file.write("\nRecall=%s\n"% result[4])
            the_file.write("\nPrec=%s\n"% result[5]) 

    

       
def runRVCWithLinearKernel(mykernl = "linear"):
    print("runRVCWithLinear2Kernel with kernel=",mykernl)
    return RVC(kernel = mykernl,verbose=True,fit_intercept=True, tol=0.01, n_iter=300)


#This functions is used to train regular RVM (without our extension) on a given dataset
def trainRegularRVM(dataset, saveModel=False):
    print("Results for RVM")
    #uses a single large design matrix that includes
    #1. meta features (word count + rating)
    #2. tf-idf (with 1-3 grams and stemming)
    #3. LDA
    #4. word embedding
    #5. phrase embedding? "experiment_saved_models/rvm_vs_stateofart","Nelder-Mead"
    #reading files
    mindf = 5
    k = 85
    vectLength = 100
    yparent = "informative"
    ycols = ["has_feature_request","has_bug_report","has_user_exp"]
    metacols = ["rating","wordCount"]
    ##READING FILES
    infomat =  pd.read_csv('../datasets/'+dataset+'/dataset.txt')
    meta = infomat[metacols]  
    tfidf = pd.read_csv("../datasets/"+dataset+"/tfidfmat_1gram3_stem_" + str(mindf) + ".txt", header=None, sep=",");
    #LDA
    if dataset == "pinch":
        lda = pd.read_csv("../datasets/"+dataset+"/ldamat_k" + str(k) +"_1gram3_stem_" + str(mindf) + ".txt", header=None);
    else:
        lda = pd.read_csv("../datasets/"+dataset+"/ldamat_1gram3_k" + str(k) +"_stem_" + str(mindf) + ".txt", header=None);
    wordembed = pd.read_csv("../datasets/"+dataset+"/dim" + str(vectLength) + "_review_vectors.txt", header=None)
    #preprocessing
    meta= pd.DataFrame(preprocessing.normalize(infomat[metacols].values));
    meta= pd.DataFrame(preprocessing.scale(infomat[metacols].values));
    tfidf= pd.DataFrame(preprocessing.normalize(tfidf.values));
    tfidf= pd.DataFrame(preprocessing.scale(tfidf.values));
    lda= pd.DataFrame(preprocessing.normalize(lda.values));
    lda= pd.DataFrame(preprocessing.scale(lda.values));
    wordembed= pd.DataFrame(preprocessing.normalize(wordembed.values));
    wordembed= pd.DataFrame(preprocessing.scale(wordembed.values));
    #Phrase embed
    
    #put together
    #meta有2列多行，lda是一个100列多行的矩阵
    X = pd.concat([meta,tfidf], axis=1)
    X = pd.concat([X,lda], axis=1)
    X = pd.concat([X,wordembed], axis=1)
    y = infomat[yparent]
    
    #split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=3,shuffle=True, stratify=y)
    train_indices = y_train.index.values
    test_indices = y_test.index.values

    #train
    rvmLinearResult = []
    for column in ycols:
        y = infomat[column]
        rvmLinearResult.append(runClassifier(runRVCWithLinearKernel,X,y,train_indices,test_indices))

    #add informative
    info_true = infomat["informative"].iloc[test_indices].values#used with getAUCperClass
    rvmLinearResult.append(getFCInformativeResult(rvmLinearResult,infomat,test_indices))

    #evaluate
    print("### RVM Results")
    calculateMacroF1andRecall(rvmLinearResult)
    calculateMicroF1andRecall(rvmLinearResult)
    getAUCperClassFlat(rvmLinearResult, info_true)
    getAUCperClassFlat(rvmLinearResult, info_true,auc_type="roc")
    
    #saving object to disk
    if saveModel:
        filename = "../experiment_saved_models/rvm_vs_stateofart/"+dataset+"_regularrvm_allfeatures.file"
        saveObject(rvmLinearResult,filename)
    return rvmLinearResult


def evaluteModelWithWeights(bestWeight,targetLabel,folder, task, logFolder, method,saveModel=False):
        #read data
        X_train, X_test, Y_train, Y_test, _, _, _,_ = getKernelsData(targetLabel,folder, task, logFolder, method)
        if task=="HC":
            #This is MK-CLS
            X_train, Y_train = trained_CL(X_train, Y_train,task,folder,targetLabel,method)
            #This is MK-S
            # X_train,Y_train=trained_CLNI(X_train, Y_train)
            #This is MK-CL
            # X_train, Y_train = trained_CL(X_train, Y_train,task,folder,targetLabel,method)
            #This is MK-SCL
            # X_train, Y_train = trained_CL(X_train, Y_train,task,folder,targetLabel,method)
        # evaluate model
        result = evaluteWeightsOnTest(bestWeight, X_train, X_test, Y_train, Y_test, attempts=10)
        print("###Weights=",bestWeight)
        print("-Performance on testing:")
        print("-- AUC_PR=", result[1])
        print("-- AUC_ROC=", result[2])
        print("-- F1=", result[3])
        print("-- Recall=", result[4])
        if saveModel:
            #save model
            filepath = "../"+logFolder+"/"+folder+"_"+method+"_bestModels_"+task+"_"+targetLabel+".file"
            saveObject(result,filepath)
            #save result
            filepath = '../'+logFolder+'/result_weights/'+folder+"_"+task+"_"+targetLabel+ "_" + method+".txt"
            with open(filepath, 'a') as the_file:
                the_file.write("\n-Best Weights= %s,%s,%s,%s"% (bestWeight[0],bestWeight[1], bestWeight[2], bestWeight[3]))
                the_file.write("\n-Performance on Testing:")
                the_file.write("\nAUC_PR=%s\n"% result[1])
                the_file.write("\nAUC_ROC=%s\n"% result[2])
                the_file.write("\nF1=%s\n"% result[3])
                the_file.write("\nRecall=%s\n"% result[4])


########################
# To use the simplex method to learn weights for one of the datasets, kindly use (), which will print out the learned weights once optimization is complete
########################

# ##finding thetas
# #select dataset and task
# task = "FC" #FC or HC
# folder = "maalej"#maalej or pinch
# maxiter = 1#determines how fast/slow the optimization will be
# #Task is either FC = Flat classification, or HC = Hierarchical classification
# # if HC is selected, then we assume an informative model representing first level classification was already trained and ready for use
# ycols = ["informative","has_feature_request","has_bug_report","has_user_exp"]
# if task == "HC":
#     ycols = ["has_feature_request","has_bug_report","has_user_exp"]
#
# methods = ['Nelder-Mead']
# logFolders= ["experiment_saved_models/rvm_vs_stateofart/"]
# options = [{'xtol': 1e-8, 'disp': True, 'maxiter':maxiter}]
# for i in range(len(methods)):
#     print("Working on folder=",logFolders[i],". method=", methods[i], "and options=", options[i], "and logFolder=",logFolders[i])
#     runSciptMinimize(ycols, folder, task, methods[i], options[i], logFolders[i])
    

########################
# To train a regular RVM (without our proposed extenstion), then kindly use trainRegularRVM()
# To train an RVM model using a given set of kernel weights, then kindly use modelTrain(), which trains an RVM model using weights, and modelPredict(), which evaluate performance on testing
########################
    
#train regular RVM with no learned kernel weight (uncomment the following line)
# trainRegularRVM("pinch")
# trainRegularRVM("maalej")
    

#train multi-kernel RVM with specific weights
#Un comment the following
#########Pinch
print("##learning weights for pinch")
logFolder = "experiment_saved_models/rvm_vs_stateofart"
method = "Nelder-Mead"
folder = "pinch"
task = "FC"
targetLabels = ["informative","has_feature_request","has_bug_report","has_user_exp"]
bestWeights = [[0.520562327523,0.450165600186,0.517707144896,0.519529102662],
              [0.498936653137,0.523008918762,0.501436042786,0.500075340271],
              [0.469415709442,0.515048689362,0.511442442579,0.516870836018],
              [0.502043418538,0.474704736362,0.499832878419,0.514874453601]]
for i in range(0,len(targetLabels)):
        evaluteModelWithWeights(bestWeights[i],targetLabels[i],folder, task, logFolder,method,True)

task = "HC"
targetLabels = ["has_feature_request","has_bug_report","has_user_exp"]
bestWeights = [[0.528430415572,0.498754573763,0.488429150967,0.508001702332],
              [0.512488961461,0.512493287813,0.47500664922,0.51249732488],
              [0.5,0.50625,0.5125,0.5]]
for i in range(0,len(targetLabels)):
        evaluteModelWithWeights(bestWeights[i], targetLabels[i], folder, task, logFolder,method,True)

print("##Evaluating weights for pinch")
#evaluate weights on test
evaluateFC(folder, logFolder, method)
evaluateHC(folder, logFolder, method)
# # #
# # #
# # #
# # #########Maalej
print("##learning weights for maalej")
folder = "maalej"
task = "FC"
targetLabels = ["informative","has_feature_request","has_bug_report","has_user_exp"]
bestWeights = [[0.495767831802,0.532848119736,0.474793148041,0.510634183884],
              [0.5,0.5,0.5125,0.5],
              [0.490633381622,0.50936897358,0.493754561659,0.52187598308],
              [0.504321289062,0.498461914063,0.501391601562,0.50400390625]]

for i in range(0,len(targetLabels)):
        evaluteModelWithWeights(bestWeights[i], targetLabels[i], folder, task, logFolder,method,True)

task = "HC"
targetLabels = ["has_feature_request","has_bug_report","has_user_exp"]
bestWeights = [[0.489154922733,0.503873537464,0.522352652975,0.521333297772],
                [0.512513205649,0.474998374095,0.512530606898,0.512479026884],
                [0.504959080754,0.382713237438,-0.00119830543425,0.892282387929]]
for i in range(0,len(targetLabels)):
        evaluteModelWithWeights(bestWeights[i], targetLabels[i], folder, task, logFolder,method,True)
print("##Evaluating weights for maalej")
    #evaluate weights on test
evaluateFC(folder, logFolder, method)
evaluateHC(folder, logFolder, method)