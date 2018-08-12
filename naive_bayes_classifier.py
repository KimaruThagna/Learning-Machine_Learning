#simple use of naive bayes classifier
'''
TODO
use real world dataset
'''
from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
# create dataset
#height,weight,age
X = [[121, 80, 44], [180, 70, 43], [166, 60, 38], [153, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [174, 71, 40], [159, 52, 37], [171, 76, 42], [183, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
trained_gnb=gnb.fit(X,Y)
prediction= trained_gnb.predict([[150,50,38]])
print(prediction)