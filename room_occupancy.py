import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# load all data
'''
TIME SERIES DATA
'''
data1 = pd.read_csv('occupancy_dataset/datatest.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data2 = pd.read_csv('occupancy_dataset/datatraining.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data3 = pd.read_csv('occupancy_dataset/datatest2.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
# determine the number of features
n_features = data1.values.shape[1]
pyplot.figure()
for i in range(1, n_features):
	# specify the subpout
	pyplot.subplot(n_features, 1, i)
	# plot data from each set
	pyplot.plot(data1.index, data1.values[:, i])
	pyplot.plot(data2.index, data2.values[:, i])
	pyplot.plot(data3.index, data3.values[:, i])
	# add a readable name to the plot
	pyplot.title(data1.columns[i], y=0.5, loc='right')
pyplot.show()
# vertically stack and maintain temporal order
data = pd.concat([data1, data2, data3])
# drop row number
data.drop('no', axis=1, inplace=True)
# save aggregated dataset
data.to_csv('combined.csv')
values=data.values
# split data into inputs and outputs ie, features and labels
X, y = values[:, :-1], values[:, -1]
# split the dataset
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3,random_state=1)

# make a naive prediction. This predicts all of the test data to be one value of the label
# in this case, all the test data will be predicted to be 0 in the 1st round and 1 in round 2
def naive_prediction(testX, value):
	return [value for x in range(len(testX))]


# evaluate skill of predicting each class value
for value in [0, 1]:
	# forecast
	yhat = naive_prediction(testX, value)
	# evaluate
	score = accuracy_score(testy, yhat)
	# summarize
	print('Naive=%d score=%.3f' % (value, score))

#using logistic regression,
model=LogisticRegression()
model.fit(trainX,trainy)
yhat=model.predict(testX)
score=accuracy_score(testy,yhat)
print(score)
#running a logistic regression algorithm on each feature to evaluate feature selection
features = [0, 1, 2, 3, 4]
for f in features:
	# split data into inputs and outputs
	X, y = values[:, f].reshape((len(values), 1)), values[:, -1]
	# split the dataset
	trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=1)
	# define the model
	model = LogisticRegression()
	# fit the model on the training set
	model.fit(trainX, trainy)
	# predict the test set
	yhat = model.predict(testX)
	# evaluate model skill
	score = accuracy_score(testy, yhat)
	print('feature=%d, name=%s, score=%.3f' % (f, data.columns[f], score))
#based on the individual accuracy scores, light is the most important feature followed bytemperature
# this means that in the absence of some of the other columns, light and temperature would still
#yield high accuracy