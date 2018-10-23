#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import csv
import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd

import csv
import sys
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#print内容打印到指定名字文件上
# class Logger(object):
#     def __init__(self,filename = 'Default.log'):
#         self.terminal = sys.stdout
#         self.log = open(filename,'a')
#     def write(self,message):
#         self.terminal.write(message)
#         self.log.write(message)
#     def flush(self):
#         pass
# sys.stdout = Logger('DTC4.txt')

train = pd.read_csv('train_0.7.csv')  # 训练数据train
X_train = train.values[0:,7:-1]#取样本数据，所有行，除了0,和最后3列的所有列 现在第0列是游戏名
y_train = train.values[0:,-1]

test = pd.read_csv('test_0.3.csv')  # 训练数据test
X_test = test.values[0:,7:-1]#取样本数据，所有行，除了0,和最后1列的所有列 现在第0列是游戏名
y_test = test.values[0:,-1]


gbm0 = GradientBoostingClassifier(n_estimators=5000,learning_rate=0.1,max_depth=7,random_state=10)
#print '7:3,n_estimators=300,learning_rate=0.08,random_state=10'
#1:9 n_estimators=300,learning_rate=0.9, random_state=10
#2:8 n_estimators=70,learning_rate=0.05, random_state=10
#3:7,n_estimators=100,learning_rate=0.09,random_state=10

# weight = []
# for each in y:
#    weight.append(0.05) if each == 0 else weight.append(0.1)
#gbm0.fit(X,y,sample_weight=weight)
gbm0.fit(X_train, y_train)
print(gbm0.feature_importances_)
print(metrics.classification_report(y_train,gbm0.predict(X_train)))
print(metrics.classification_report(y_test,gbm0.predict(X_test)))


# from sklearn.pipeline import Pipeline
# #用信息增益启发式算法建立决策树
# pipeline=Pipeline([('clf',GradientBoostingClassifier(random_state=10))])
#
#
# parameters = {
# 'clf__max_depth': (5,7,9,10,13,15,17,19,21,23, 25, 27),
# 'clf__min_samples_split': (0.2,0.4,0.6,0.8,1.0, 2, 3,4,5,6),
# 'clf__min_samples_leaf': (1, 2, 3,4,5,6)
# }
#
# from sklearn.model_selection import  GridSearchCV
# if __name__ == '__main__':
#     #f1查全率和查准率的调和平均
#     grid_search=GridSearchCV(pipeline,parameters,n_jobs=-1,verbose=1,scoring='f1')
#     grid_search.fit(X_train,y_train)
#     print('最佳效果：%0.3f'%grid_search.best_score_)
#     best_parameters=grid_search.best_estimator_.get_params()
#     print('最优参数'%best_parameters)
#
#     from  sklearn.metrics import classification_report
#     for param_name in sorted(parameters.keys()):
#         print ('\t%s:%r'%(param_name,best_parameters[param_name]))
#     predictions=grid_search.predict(X_test)
#     print(classification_report(y_test,predictions))


