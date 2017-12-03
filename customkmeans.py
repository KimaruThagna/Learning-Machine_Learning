import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')
import pandas as pd
X=np.array([
    [1,2],
    [1.5,1.8],
    [5,8],
    [8,8],
    [1,0.6],
    [9,11]
])
#plt.scatter(X[:,0],X[:,1],s=15,linewidths=5)
#plt.show()
colors=["k","b","r","c","g"]
class K_Means:
    def __init__(self,k=2,tol=0.001,max_iter=300): # tolerance...how much the
        #centroid can move. MAx_Iter--number of times we repreat process to find
#optimum centroid
        self.k=k
        self.tol=tol
        self.max_iter=max_iter
    def fit(self,data):
        self.centroids={}
        for i in range(self.k):
            self.centroids[i]=data[i] #the starting centroids are
            #  the first k elements of the dataset
        for i in range(self.max_iter):
            self.classifications={}# structure keys will be centroid
            # value will be a list of the featuresets closest to the centroid
            for i in range(self.k):
                self.classifications[i]=[]
            for featureset in data:
                distances=[np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                # distance between a particular featureset and the k centroids
                classification=distances.index(min(distances)) # index of minimum value
                self.classifications[classification].append(featureset)

                # at the position of classification, populate the value list with the feature
                prev_centroids=dict(self.centroids)
            for classification in self.classifications:
                # find the mean datapoint of the featuresets of a particular centroid
                self.centroids[classification]=np.average(self.classifications[classification],axis=0)
            optimized=True
            for c in self.centroids:
                original_centroid=prev_centroids[c]
                current_centroid=self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0)>self.tol:
                     # if the centroids have moved by a factor greater than the tolerance,
                    optimized=False
                if optimized:
                    break # dont go upto the max iteration
    def predict(self,data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
clf=K_Means()
clf.fit(X)
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker="*",color="k",s=110)
for classification in clf.classifications:
    color=colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker="x",color=color,s=50)

new_data=np.array([[4,8],
                   [3,7],
                   [5,-4],
                   [1,1],
                   [2,8],
                   [1,4]])
for data in new_data:
    classification=clf.predict(data)
    plt.scatter(data[0],data[1],color=colors[classification],s=50,marker="o")
plt.show()


