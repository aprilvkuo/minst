#!/usr/bin/env python
# encoding: utf-8


"""
@author: aprilvkuo
@license: Apache Licence
@software: PyCharm Community Edition
@file: minst.py
@time: 2017/8/10 14:40
"""

from xgboost.sklearn import XGBClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression,RidgeClassifier 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import os,traceback,sys

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def getclf():
    clf = []
    knn = KNeighborsClassifier() 
    lr = LogisticRegression()
    svc = svm.SVC(kernel = 'linear')    
    liner = RidgeClassifier()
    rf = RandomForestClassifier()  
    xgb = XGBClassifier()
    clf.append( (knn,'knn'))
    clf.append( (lr,'logistic') )
    clf.append( (svc, 'svm') )
    clf.append( (liner,'liner'))
    clf.append( (rf,'random_forest'))
    clf.append( (xgb, 'xbg'))
    return clf 

def write_to_file():
    pass 
if __name__ == '__main__':
    test_flag = sys.argv[1] 
    df = pd.read_csv(u"./train.csv")
    if test_flag=='1':
        df = df[:50]
    test_df = pd.read_csv('./test.csv')
    train_x,train_y = df.iloc[:,1:],df.iloc[:,0]
    cnt = [i+1 for i in range(len(test_df))]
    target_name = []
    for i in range(10):
        target_name.append('class '+str(i))
    for clf,name in getclf():
        #score = cross_val_score(clf,train_x,train_y)
        try:
            print('clf:',name)
            clf.fit(train_x,train_y)
            result = clf.predict(test_df)
            
            result = pd.DataFrame({'ImageId':cnt,'Label':result})
            result.to_csv(name+".csv",index=False)
            predict_y = clf.predict(test_df)
            report = classification_report(train_y,predict_y,target_names=target_name)
            print(report)
            with open(name+'_report.txt','w',encoding='utf-8') as f:
                f.write(report)
        except Exception as e:
            traceback.print_exc()                
        
        

    



#clf.fit(train_x,train_y)

#result = clf.predict(test_df)
#cnt = [i+1 for i in range(len(result))]
#result = pd.DataFrame({'ImageId':cnt,'Label':result})
#result.to_csv("v_2.csv",index=False)
