#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anonymous
"""
import pickle
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def runClassifier(functionName,features, y, train_indices, test_indices,myargs=[]):
    
    #slice data based on fold
    features_train = features.iloc[train_indices,:]
    features_test =features.iloc[test_indices,:]
    y_train = y.iloc[train_indices,]
    y_test =y.iloc[test_indices,]
    
#    print("features=", features.shape, ", y=", y.shape)
#    print("features.train=", features_train.shape, ", y.train=", y_train.shape, ",features.test=",features_test.shape, "y.test=", y_test.shape)
    #print("features columns:")
    #print(features_test.columns)
    
    #running model
    model = functionName();
#    model = runRVC(*myargs);
    
    #train model
    model.fit(features_train, y_train)
   
    #running model on testin set
    train_pred = pd.DataFrame(index=train_indices, data=model.predict(features_train))
    test_pred = pd.DataFrame(index=test_indices, data=model.predict(features_test))
    
    #get result
    acc_test, precision_test, recall_test, f1score_test, cm = getClassificationMetrics(y_test, test_pred)
    
    #return results
    return model, acc_test, precision_test, recall_test, f1score_test, cm,train_indices, test_indices,train_pred, test_pred, y_test, features_train,features_test


def runSVM(mykern = "linear", randomstate = 3):
    return svm.SVC(kernel = mykern, random_state=randomstate, probability=True)

def runRF(n = 200, randomestimate = 3):
    print("runRF with trees=",200)
    return RandomForestClassifier(random_state=randomestimate,n_estimators=n)

def runLogR(pen="l2", rand=3):
    return linear_model.LogisticRegression(random_state=rand, penalty=pen,solver='liblinear')#l1=lasso, l2=ridge

def runNaiveBayes(alp=1.0):
    return MultinomialNB(alpha=alp);

def getObject(filename):
    #model has randomness in training, it may produce slightly different results with different training sessions
    #thuss loading a saved HMK-RVM model to allow for reproducibility of exact result
    with open(filename, "rb") as inputFile:
        rvmLinearResult = pickle.load(inputFile)
    return rvmLinearResult

def saveObject(modelInfo,filename=None):
     #saving object to disk
    with open(filename, "wb") as outputFile:
        pickle.dump(modelInfo, outputFile, pickle.HIGHEST_PROTOCOL)

def getClassificationMetrics(y_true, y_pred):
    acc_test = metrics.accuracy_score(y_true,y_pred)
    precision_test = metrics.precision_score(y_true,y_pred)
    recall_test = metrics.recall_score(y_true,y_pred);
    f1score_test =  metrics.f1_score(y_true,y_pred);
    cm = confusion_matrix(y_true, y_pred);
    return acc_test, precision_test, recall_test, f1score_test, cm

def calculateMicroF1andRecall(results):
    tp = []
    fp = []
    fn = []
    for result in results:
        tp.append(result[5][1,1])
        fp.append(result[5][0,1])
        fn.append(result[5][1,0])
    
    microrecall = sum(tp)/(sum(tp)+sum(fn))
    microprecision = sum(tp)/(sum(tp)+sum(fp))
    microf1 = 2 * ((microrecall * microprecision)/(microrecall + microprecision))
    print("calculateMicroF1andRecall(): f1=",microf1 ,"recall=",microrecall);
    return microf1, microrecall
    
def calculateMacroF1andRecall(results):
    f1 = []
    recall = []
    precision = []
    for result in results:
        f1.append(result[4])
        recall.append(result[3])
        precision.append(result[2])
    macrof1 = sum(f1)/len(f1)
    macrorecall = sum(recall)/len(recall)
    macrop = sum(precision)/len(precision)
    print("f1=",f1)
    print("recall=",recall)
    print("precision=",precision)
    print("calculateMacroF1andRecall(): f1=",macrof1 ,"recall=",macrorecall, "precision=",macrop);
    return macrof1, macrorecall

def getMicroF1andRecallUsingCM(results):
    tp = []
    fp = []
    fn = []
    #recall = tp/(tp+fn)
    #prec = tp/(tp+fp)
    #matrix = [TN,FP], [FN,TP] WHERE y-axis = True, and X-axis = Predicted
    for cm in results:
        tp.append(cm[1,1])
        fp.append(cm[0,1])
        fn.append(cm[1,0])
    
    microrecall = sum(tp)/(sum(tp)+sum(fn))
    microprecision = sum(tp)/(sum(tp)+sum(fp))
    microf1 = 2 * ((microrecall * microprecision)/(microrecall + microprecision))
    print("calculateMicroF1andRecall(): f1=",microf1 ,"recall=",microrecall);
    
    return microf1, microrecall, microprecision
    
def getMacroF1andRecallUsingDirectVals(f1,recall,precision):
    macrof1 = sum(f1)/len(f1)
    macrorecall = sum(recall)/len(recall)
    macrop = sum(precision)/len(precision)
    print("f1=",f1)
    print("recall=",recall)
    print("precision=",precision)
    print("calculateMacroF1andRecall(): f1=",macrof1 ,"recall=",macrorecall, "precision=",macrop);
    return macrof1, macrorecall, macrop


def getAUCperClass(result, auc_type="pr"):
    auc = []
    #get_AUC_ROC
    #get_AUC_Prec_Recall
    for model in result:
        if hasattr(model[10], 'values'):
            trueY = model[10].values
        else:
            trueY = model[10]
            
        if(auc_type=="roc"):
            auc.append(get_AUC_ROC(y_true=trueY,probas_pred= model[0].predict_proba(model[12])[:, 1],pos_label=1, showPlot = False))
        else:
            auc.append(get_AUC_Prec_Recall(y_true=trueY,probas_pred= model[0].predict_proba(model[12])[:, 1],pos_label=1, showPlot = False))
#        p, r, thresholds = precision_recall_curve(y_true=result[i][10].values, probas_pred= result[i][0].predict_proba(result[i][12])[:, 1])
#        auc.append(metrics.auc(r,p))
#        auc.append(metrics.roc_auc_score(result[i][10].values,result[i][0].predict_proba(result[i][12])[:, 1],))
    print("AUCs -",auc_type," (mean=",np.mean(auc),") =", auc)
    return auc;
        
def getAUCperClassFlat(result, info_true, auc_type="pr"):
    auc = []
    probs = [] #place holder to be used for "informative" class AUC when flat classification is used
    #get_AUC_ROC
    #get_AUC_Prec_Recall
    for i in range(0,len(result)-1):#-1 to exclude the last item, which informative, as in flat, we do not have a classifier for that
        probs.append(result[i][0].predict_proba(result[i][12])[:, 1])
        if(auc_type=="roc"):
            auc.append(get_AUC_ROC(y_true=result[i][10].values,probas_pred= result[i][0].predict_proba(result[i][12])[:, 1],pos_label=1, showPlot = False))
        else:
            auc.append(get_AUC_Prec_Recall(y_true=result[i][10].values,probas_pred= result[i][0].predict_proba(result[i][12])[:, 1],pos_label=1, showPlot = False))
    
    #adding the informative label
    informative_predict_proba = np.average(probs,axis=0)
    if(auc_type=="roc"):
        auc.append(get_AUC_ROC(y_true=info_true, probas_pred=informative_predict_proba,pos_label=1))
    else:
        auc.append(get_AUC_Prec_Recall(y_true=info_true, probas_pred=informative_predict_proba,pos_label=1))
    print("AUCs -",auc_type," (mean=",np.mean(auc),") =", auc)
    return auc;

def get_AUC_ROC(y_true,probas_pred,pos_label=1, showPlot = None):
    #https://stats.stackexchange.com/questions/157012/area-under-precision-recall-curve-auc-of-pr-curve-and-average-precision-ap
    #this Area under the curve (AUC) is calculated from the ROC curve which uses
    #true positive rate and false positive rate 
    #WARNING: THIS AUC SHOULD BE ONLY USED WHEN THE CLASSES ARE WELL BALANCED
    #IF YOU HAVE UNBALANCED CLASSES, CALCULATE AUC USING PRECISION AND RECALL
    prauc = metrics.roc_auc_score(y_true=y_true,y_score=probas_pred)
#    print("auc_roc=",prauc)
    return prauc

def get_AUC_Prec_Recall(y_true,probas_pred,pos_label=1, showPlot = None):
    #https://stats.stackexchange.com/questions/157012/area-under-precision-recall-curve-auc-of-pr-curve-and-average-precision-ap
    #this Area under the curve (AUC) is calculated from the Preicision and Recall curve
    #This is better metric when you have an imbalanced set of classes
    p, r, thresholds = precision_recall_curve(y_true=y_true, probas_pred= probas_pred, pos_label=pos_label)
    auc = metrics.auc(r,p)
#    print("auc_prec_recall=",auc)
    if(showPlot):
        plt.plot(r,p)
#        step_kwargs = ({'step': 'post'}
#               if 'step' in signature(plt.fill_between).parameters
#               else {})
#        plt.step(recall, precision, color='b', alpha=0.2,
#                 where='post')
#        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
                  auc))
        plt.show()
    return auc


def getFCInformativeResult(results,infomat,test_indices):
    trueLabels = infomat["informative"].iloc[test_indices].values
    #get predicted
    #logic = if one class predicted  1, then predicted = 1, otherwise, if all zero, then p=zero
    predLabels = [0] * len(test_indices)#init all zero
    for result in results:
        if -(result[9][0], 'values'):
            classpred = result[9][0].values
        else:
            classpred = result[9][0]
        predLabels = predLabels | classpred
    
    #get result
    acc_test, precision_test, recall_test, f1score_test, cm = getClassificationMetrics(trueLabels, predLabels)
    #matching:# return model, acc_test, precision_test, recall_test, f1score_test, cm,train_indices, test_indices,train_pred, test_pred, y_test, features_train,features_test
    return[0,acc_test, precision_test, recall_test, f1score_test,cm,0,0,0,predLabels,trueLabels,0,0]
    


def getKernelsData(targetLabel,folder, task, logFolder = "Results_bo", method = ""):
    #reading files
    yparent = "informative"
    folder= folder
    mindf = 5
    k = 85
    vectLength = 100
    metacols = ["rating","wordCount"]
    #reading files
    infomat =  pd.read_csv('../datasets/'+folder+'/dataset.txt')
    tfidf = pd.read_csv("../datasets/"+folder+"/tfidfmat_1gram3_stem_" + str(mindf) + ".txt", header=None, sep=",");
    if(folder == "pinch"):
        lda = pd.read_csv("../datasets/"+folder+"/ldamat_k" + str(k) +"_1gram3_stem_" + str(mindf) + ".txt", header=None);
    else:
        lda = pd.read_csv("../datasets/"+folder+"/ldamat_1gram3_k" + str(k) + "_stem_" + str(mindf) + ".txt", header=None);
    wordembed = pd.read_csv("../datasets/"+folder+"/dim" + str(vectLength) + "_review_vectors.txt", header=None)
    
    #preprocessing
    meta= pd.DataFrame(preprocessing.normalize(infomat[metacols].values));
    meta= pd.DataFrame(preprocessing.scale(infomat[metacols].values));
    tfidf= pd.DataFrame(preprocessing.normalize(tfidf.values));
    tfidf= pd.DataFrame(preprocessing.scale(tfidf.values));
    lda= pd.DataFrame(preprocessing.normalize(lda.values));
    lda= pd.DataFrame(preprocessing.scale(lda.values));
    wordembed= pd.DataFrame(preprocessing.normalize(wordembed.values));
    wordembed= pd.DataFrame(preprocessing.scale(wordembed.values));
    
    #preparing x
    X = pd.concat([wordembed, meta], axis=1)
    X = pd.concat([X,tfidf], axis=1)
    X = pd.concat([X,lda], axis=1)
    Xk = [wordembed.values,meta.values,tfidf.values, lda.values]

    #split data based on informative column
    y = infomat[yparent]
    X2_train, X2_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=3,shuffle=True, stratify=y)
    train_indices = y_train.index.values
    test_indices = y_test.index.values

    if(task == "HC"):#flat
        #hierarchical
        #load parent model (best classifier for parent)
#        parentmodel = None
        parentPredictions = None
        mypt = "./"+logFolder+"/"+folder+"_bestModels_FC_"+yparent+".file"
        if(len("method")>0):
            mypt = "./"+logFolder+"/"+folder+"_"+method+"_bestModels_FC_"+yparent+".file"

        with open(mypt, "rb") as inputFile:
            parentobject = pickle.load(inputFile)
#           parentmodel = parentobject[0]
            parentPredictions = parentobject[7]
            #[parentobject[0],0, parentobject[5], parentobject[4], parentobject[3], parentobject[6],parentobject[1],parentobject[2]]
            print("Parent Model with AUC=",parentobject[1])
             #picking new training and testing ind for 2nd level
        #creating ground truth for lvl 2 (for training it must from groud truth, while testing is from predicitions)
        train_indices2 = np.where(infomat["informative"].iloc[train_indices]==1)[0]#getting positions in array
        train_indices2 = train_indices[train_indices2] #getting true ind
        rvm_testinds2 = test_indices[np.where(parentPredictions==1)[0]]
        #update ind
        train_indices = train_indices2

        test_indices = rvm_testinds2
    
    #prepare representations for kernels
    X_train=[]
    X_test=[]
    for rep in Xk:
        X_train.append(rep[train_indices,:])
        X_test.append(rep[test_indices,:])
    labels = infomat[targetLabel].values
    #prepare Y
    Y_train=labels[train_indices]    
    Y_test=labels[test_indices]


    return X_train, X_test, Y_train, Y_test, infomat, train_indices, test_indices, X  

def evaluateFC(dataset, logFolder = "Results_bo", method=""):
    labels = ["has_feature_request","has_bug_report","has_user_exp"]
#    [model,auc_pr,auc_roc, f1score_test, recall_test, precision_test, cm, predictions]
    aucprs = []
    aucrocs = []
    f1s = []
    recalls =[]
    precisions = []
    cms = []
    predicitions = []
    baselineModels = []
    probs = []
    weights = []
    rvs = []
    for i in labels:
        mypt = "./"+logFolder+"/"+dataset+"_bestModels_FC_"+i+".file"
        if(len(method)>0):
            mypt = "./"+logFolder+"/"+dataset+"_"+method+"_bestModels_FC_"+i+".file"
        with open(mypt, "rb") as inputFile:
                myobject = pickle.load(inputFile)
                baselineModels.append(myobject[0])
                rvs.append(len(myobject[0].relevant_[0]))
                aucprs.append(myobject[1])
                aucrocs.append(myobject[2])
                f1s.append(myobject[3])
                recalls.append(myobject[4])
                precisions.append(myobject[5])
                cms.append(myobject[6])
                predicitions.append(myobject[7])
                probs.append(myobject[8][:,1])
                weights.append(myobject[9])
                
    #We have results for three classes, we need informative next
    #To do, we will combine predictions from all three classifiers into a single overall prediction that mimic an informative classifier
    #Next, we compare the overall predictions (using combined knowledge) to the true labels to measure accuracy
    #As for AUC, ROC, since we do not have a classifier for informative, we are taking the average of the class probabilties across the three classifiers to indirectly use them as mimic to an informative classifier
    
    #get predictions
    #logic = if one class predicted  1, then predicted = 1, otherwise, if all zero, then p=zero
    X_train, X_test, Y_train, Y_test,_, _,_,_ = getKernelsData("informative",dataset, "FC")
    predLabels = [0] * len(Y_test)#init all zero
    for classifierPredicition in predicitions:
        predLabels = predLabels | classifierPredicition
    #get result
    acc_test, precision_test, recall_test, f1score_test, cm_test = getClassificationMetrics(Y_test, predLabels)
    f1s.append(f1score_test)
    recalls.append(recall_test)
    precisions.append(precision_test)
    cms.append(cm_test)
    #get AUCs, we need to average probs from all theree classifiers to mimic informative
    informative_predict_proba = np.average(probs,axis=0)
    aucprs.append(get_AUC_Prec_Recall(y_true=Y_test, probas_pred=informative_predict_proba,pos_label=1) )
    aucrocs.append(get_AUC_ROC(y_true=Y_test, probas_pred=informative_predict_proba,pos_label=1))
    #print result
    getMacroF1andRecallUsingDirectVals(f1s,recalls,precisions)
    getMicroF1andRecallUsingCM(cms)
    print("AUCs - pr (mean=",np.mean(aucprs),") =", aucprs)
    print("AUCs - roc (mean=",np.mean(aucrocs),") =", aucrocs)
    print("RVs and weights:")
    for i in range(0,len(labels)):
        print("-",labels[i],": RVs=",rvs[i],", Kernel-Weights= embedding[",weights[i][0],"],meta[",weights[i][1],"],tfidf[",weights[i][2],"],lda[",weights[i][3],"]" )
 

def evaluateHC(dataset, logFolder="Results_bo", method=""):
    labels = ["informative","has_feature_request","has_bug_report","has_user_exp"]
#   [model,auc_pr,auc_roc, f1score_test, recall_test, precision_test, cm, predictions]
    aucprs = []
    aucrocs = []
    f1s = []
    recalls =[]
    precisions = []
    cms = []
    weights = []
    rvs = []
    for i in labels:
        path = "./"+logFolder+"/"+dataset+"_"+method+"_bestModels_HC_"+i+".file";
        if(i=="informative"):
            path = "./"+logFolder+"/"+dataset+"_"+method+"_bestModels_FC_"+i+".file"
        with open(path, "rb") as inputFile:
            myobject = pickle.load(inputFile)
            aucprs.append(myobject[1])
            aucrocs.append(myobject[2])
            f1s.append(myobject[3])
            recalls.append(myobject[4])
            precisions.append(myobject[5])
            cms.append(myobject[6])
            weights.append(myobject[9])
            rvs.append(len(myobject[0].relevant_[0]))
    getMacroF1andRecallUsingDirectVals(f1s,recalls,precisions)
    getMicroF1andRecallUsingCM(cms)
    print("AUCs - pr (mean=",np.mean(aucprs),") =", aucprs)
    print("AUCs - roc (mean=",np.mean(aucrocs),") =", aucrocs)
    print("RVs and weights:")
    for i in range(0,len(labels)):
        print("-",labels[i],": RVs=",rvs[i],", MKWeights= embed[",weights[i][0],"],meta[",weights[i][1],"],tfidf[",weights[i][2],"],lda[",weights[i][3],"]" )
 
    
    