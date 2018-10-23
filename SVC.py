#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import csv
import sys
from itertools import product
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
from sklearn import datasets, svm
from sklearn.datasets import make_moons, make_circles, make_classification
import os
import csv
import sys
import pandas as pd
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

model = SVC(C=300,random_state=10)
model.fit(X_train, y_train)
# weight = []
# for each in y:
#    weight.append(0.05) if each == 0 else weight.append(0.1)
# model.fit(X,y,sample_weight=weight)


print(metrics.classification_report(y_train,model.predict(X_train)))
print(metrics.classification_report(y_test,model.predict(X_test)))


#训练完成后，我们可以用 export_graphviz 将树导出为 Graphviz 格式。
# from sklearn.externals.six import StringIO
# with open("tree_graphy","w") as f:
#     f = tree.export_graphviz(clf,out_file = f)
#
# #安装了pydot 模块，可以直接用Python创建PDF文件
# import  pydot
# dot_data = StringIO()
# tree.export_graphviz(clf,out_file=dot_data)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph[0].write_pdf("tree.pdf")


import  matplotlib.pyplot as plt
#训练模型
# clf.fit(X_train, y_train)
#
# print "测试模型有效性:",clf.score(X_train,y_train)
# print "基础验证的准确率:",clf.score(X_test,y_test)
#
# from sklearn import metrics
# print(metrics.classification_report(y_train,clf.predict(X_train)))
# print(metrics.classification_report(y_test,clf.predict(X_test)))

#
#
#系统每个特征值的影响
# # print "clf.feature_importances_",clf.feature_importances_
#
# from sklearn.pipeline import Pipeline
# #用信息增益启发式算法建立决策树
# pipeline=Pipeline([('clf',tree.DecisionTreeClassifier(random_state=10,criterion='entropy'))])
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
#     print '最佳效果：%0.3f'%grid_search.best_score_
#     best_parameters=grid_search.best_estimator_.get_params()
#     print '最优参数'%best_parameters
#
#     from  sklearn.metrics import classification_report
#     for param_name in sorted(parameters.keys()):
#         print ('\t%s:%r'%(param_name,best_parameters[param_name]))
#     predictions=grid_search.predict(X_test)
#     print classification_report(y_test,predictions)


