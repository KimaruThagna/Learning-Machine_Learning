import pandas as pd
import  numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
'''
NOTES
with data that has temporal ordering such as this, immediately adjacent observations 
are required to make good predictions. Thus, in some special cases, shuffling before
train-test spliting is discouraged.
in this case, Because of the high frequency of observations 
(128 per second), the most similar rows will be those adjacent 
in time to the instance being predicted, both in the past and in the future.

'''
# load the dataset
data = pd.read_csv('EEG_Eye_State.csv', header=None)
# retrieve data as numpy array
values = data.values

# create a subplot for each time series
pyplot.figure()
for i in range(values.shape[1]):# for i in columns of the dataset
	pyplot.subplot(values.shape[1], 1, i+1) # create a subplot
	pyplot.plot(values[:, i])# plot the values(all rows of the ith column)
pyplot.show()
too_large=[]
# step over each EEG column
for i in range(values.shape[1] - 1): # deal with all features minus the label
	# calculate column mean and standard deviation
	data_mean, data_std = np.mean(values[:,i]), np.std(values[:,i])
	# define outlier bounds
	cut_off = data_std * 4
	lower, upper = data_mean - cut_off, data_mean + cut_off# outliers will be values below or above 4 stds from the mean
	# remove too small
	too_small = [j for j in range(values.shape[0]) if values[j,i] < lower]
	values = np.delete(values, too_small, 0)
	print('>deleted %d rows' % len(too_small))
	# remove too large
	too_large = [j for j in range(values.shape[0]) if values[j,i] > upper]
	values = np.delete(values, too_large, 0)
	print('>deleted %d rows' % len(too_large))
# save the results to a new file
np.savetxt('EEG_Eye_State_no_outliers.csv', values, delimiter=',')

# load the dataset
data = pd.read_csv('EEG_Eye_State_no_outliers.csv', header=None)
# retrieve data as numpy array
values = data.values
# create a subplot for each time series
pyplot.figure()
for i in range(values.shape[1]):
	pyplot.subplot(values.shape[1], 1, i+1)
	pyplot.plot(values[:, i])
pyplot.show()

scores = list()
kfold = KFold(10, shuffle=True, random_state=1)
for train_ix, test_ix in kfold.split(values):
	# define train/test X/y
	trainX, trainy = values[train_ix, :-1], values[train_ix, -1]
	testX, testy = values[test_ix, :-1], values[test_ix, -1]
	# define model
	model = KNeighborsClassifier(n_neighbors=3)
	# fit model on train set
	model.fit(trainX, trainy)
	# forecast test set
	yhat = model.predict(testX)
	# evaluate predictions
	score = accuracy_score(testy, yhat)
	# store
	scores.append(score)
	print('>%.3f' % score)
# calculate mean score across each run
print('Final Score: %.3f' % (np.mean(scores)))
