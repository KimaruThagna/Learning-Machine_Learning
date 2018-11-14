from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
digits=load_digits()
data=pd.DataFrame(data=np.c_[digits['data'],digits['target']],
                            columns=digits['feature_names']+['target'])

X=data.drop('target',axis=1)
Y=data['target']
x_train,y_train,x_test,y_test=train_test_split(X,Y,test_size=0.3,random_state=85)
clf=RandomForestClassifier(n_estimators=100)# this defines the number of decision trees used
clf.fit(x_train,y_train)
predictions=clf.predict(x_test)
print(classification_report(y_test,predictions))