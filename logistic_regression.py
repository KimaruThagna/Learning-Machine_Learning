import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
#Import data from CSV file
train = pd.read_csv('titanic.csv')

#Analyze the data
print(train.count())
print(train.describe())
#Check the missing data using heatmap The contrast color shows the missing data
sns.heatmap(train.isnull(), yticklabels=False, cbar= False, cmap='summer')
plt.show() #raw dataset


# Create a function to return average age based on passenger class to reduce null data
def avg_age(col):
    Age = col[0]
    Pclass = col[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# Fill values by appling a defined function(should be backed by industry specific knowledge)
train['age'] = train[['age', 'pclass']].apply(avg_age, axis=1)

# Check the heatmap
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show() #after minimalizing the missing vals on age
train.drop(['cabin', 'name', 'ticket'], axis = 1, inplace=True) #inplace = True will not show you the values

#Display columns after dropping
print(train.head().columns)
#pd.get_dummies() converts a set of categorical variables to a series of 1s and 0s
sex = pd.get_dummies(train['sex'], drop_first = True) #drop_fist = True will drop the first column as not relevant

#Create a new variable embark with two column, if both Q and S are zero it means C is 1.
embark = pd.get_dummies(train['embarked'], drop_first = True) #first column is not required
print(embark.head())

#Drop the old column 'Sex' and 'Embarked'
train.drop(['sex', 'embarked'], axis=1, inplace=True)

#Create a new data set with quantitative information
train_new = pd.concat([train, sex, embark], axis=1)
#Display first five values
print(train_new.head())


X = train.drop('survived', axis = 1) #featureset
y = train['survived'] # label(what we want to determine
#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 101)
#classifier
clf = LogisticRegression()
# train
clf.fit(X_train, y_train)
#predict
predictions = clf.predict(X_test)

# Show classification report parameters
print (classification_report(y_test, predictions))