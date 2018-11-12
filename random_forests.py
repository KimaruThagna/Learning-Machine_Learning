import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier


train_data=pd.read_csv('digits_dataset/train.csv')
test_data=pd.read_csv('digits_dataset/test.csv')
print(train_data.head())

#inspcet single row
val=train_data.iloc[0,1: ].values
val=val.reshape(28,28).astype('uint8')
plt.imshow(val)
plt.show()