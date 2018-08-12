#simple use of naive bayes classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
gnb=GaussianNB()
# create dataset
iris=load_iris()
X=iris.data
#height,weight,age
#X = [[121, 80, 44], [180, 70, 43], [166, 60, 38], [153, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
 #    [174, 71, 40], [159, 52, 37], [171, 76, 42], [183, 85, 43]]
#Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
Y=iris.target # The labels(sertosa-0,versicolor-1 virginica-2)
trained_gnb=gnb.fit(X,Y)
prediction= trained_gnb.predict([[5.0,4.0,1.8,0.8]])
print(prediction)
##########################################################
#              USING IRIS DATA-SET AS A DATAFRAME
##########################################################
#use np.c_ to concatenate the data and lables to one array thus creating a dataframe
iris_dataframe=pd.DataFrame(data=np.c_[iris['data'],iris['target']],
                            columns=iris['feature_names']+['target'])
#print(iris_dataframe.head()) check new dataframe

feature_set=iris_dataframe.drop('target',axis=1)
target_set=iris_dataframe['target']
clf=gnb.fit(feature_set,target_set)
print(clf.predict([[5.0,4.0,1.8,0.8]]))