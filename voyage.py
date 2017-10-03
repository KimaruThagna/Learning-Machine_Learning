import pandas as pd
import quandl,math,datetime
from sklearn import cross_validation,preprocessing,svm
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle # python serializer

style.use('ggplot')

df=quandl.get('WIKI/GOOGL') # get dataset
#print(df.head()) #first five elements of the dataset
#Features- Input sets what you feed to your algorithm so that it can learn and make predictions

#Labels-Output set- The predictions that will be made from the features

df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High']- df['Adj. Low'])/df['Adj. Low']*100.0
df['OC_PCT']=(df['Adj. Close']- df['Adj. Open'])/df['Adj. Open']*100.0
#print(df.head()) #first five elements of the dataset
forecast_col='Adj. Close'
df.fillna(-99999,inplace=True) # fill missing data
forecast_out=int(math.ceil(0.01*len(df))) #math.ceil rounds up a decimal number
# This gives 10% of the number of days in the data frame
#if length of df is 100, forcastout will be 10 hence, the prediction will be 10 days into the future
#i.e for a particular row, the label column gives the value of adj. Close from 10 days down the list
df['label']=df[forecast_col].shift(-forecast_out) # shifts the whole column upward
#df.dropna(inplace=True) # drops those that loose value after shifting...the recent 10
#features variables used in model construction
#they will be used to predict the closing price
X=np.array(df.drop(['label'],1)) #this returns a dataframe containing everything BUT label coumn
# #labels the output...in this case....Adj. Close
X=preprocessing.scale(X)# make x values between -1 and 1
X_lately=X[-forecast_out:]
X=X[:-forecast_out]

df.dropna(inplace=True)
y=np.array(df['label'])
#print(len(X),len(y))

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
# use 20% of the data for training and testing

clf=LinearRegression()
clf.fit(X_train,y_train) # training the classifier
# Save the classifier to a file in order to avoid training time
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f) # dump clf to the file f
# To load the file,
pickle_in=open('linearregression.pickle','rb')
clf=pickle.load(pickle_in) # With this saved, no need to define and train the classifier again
accuracy=clf.score(X_test,y_test) # test it...based on training with a different dataset to see accuracy
#The value here is a percentage of how accurate the model, linear regression predicted the values
#point of reference, the test dataset
forecast_set=clf.predict(X_lately)# we have features....but no label(output) hence the predict Function
#will give a prediction of the label based on its training
print(forecast_set,accuracy)
df['Forecast']=np.nan
last_date=df.iloc[-1].name# get the value of the last date in the dataset
# get the index of -1...the last index///dot name gives the value of the last index
last_unix=last_date.timestamp() # convert to a timestamp
one_day=86400 # seconds in a day
next_unix=one_day+last_unix

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)] +[i]# sets the last column, forecast, to the dates
    # and makes the rest nan(Not A Number)
    # df.loc referes to the index...hence create an index of value next_date set all columns to NAN
    #then add i to this index.... i is a single value in the forecast set
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()