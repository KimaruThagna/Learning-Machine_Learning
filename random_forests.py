import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

train_data=load_digits()

print(train_data.target_names)
X=train_data.data
Y=train_data.target
x_train,y_train,x_test,y_test=train_test_split(X,Y,test_size=0.3,random_state=85)
clf=RandomForestClassifier(n_estimators=100)# this defines the number of decision trees used
clf.fit(x_train,y_train)