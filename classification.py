#   K NEAREST NEIGHBOURS
# CHOSING WHICH CLASS A POINT BELONGS TO BY DETERMINING WHICH POINTS ARE CLOSEST
#K SHOULD BE ODD TO AVOID SPLIT VOTES
from sklearn import cross_validation,preprocessing,neighbors,svm
import numpy as np
import pandas as pd

dataframe=pd.read_csv('Bcancer.txt')
dataframe.replace('?',-99999,inplace=True)# This replaces the ? with the value -99999
# thus making the data point an outlier instead of discarding the data point all at once.
dataframe.drop(['id'],1,inplace=True)
X=np.array(dataframe.drop(['class'],1))# all the rest are our features
# their values are use to determine whether a tumor is benign or malignant
y=np.array(dataframe['class']) #our label
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.75)
#use a % of tha data for training
clf=neighbors.KNeighborsClassifier()
#clf=svm.SVC() if using svm
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print(accuracy)
example_measures=np.array([[10,2,1,1,10,5,3,2,1],[1,6,1,1,1,2,3,2,1]])
print(clf.predict(example_measures.reshape(len(example_measures),-1) ) )
