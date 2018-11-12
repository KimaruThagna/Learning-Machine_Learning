import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

train_data=pd.read_csv('digits_dataset/train.csv')
test_data=pd.read_csv('digits_dataset/test.csv')
