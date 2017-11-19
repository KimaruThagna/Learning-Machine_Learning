# The start of unsupervised learning
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
# flat-clustering example algorithm-Kmeans
#hierachical-clustering example algorithm- Meanshift
X=np.array([
    [1,2],
    [1.5,1.8],
    [5,8],
    [8,8],
    [1,0.6],
    [9,11]
])
# plt.scatter(X[:,0],X[:,1],s=150,linewidths=5) #X[:,0] returns the zeroeth elements
# # of the items in the x array X[:,1] returns all the elements in pos 1 of te items in the X array
# plt.show()
clf=KMeans(n_clusters=2) #the number of clusters should be lessthan or
#equal to your datapoints but cannot exceed
clf.fit(X) # train the algorithm

# training the algorithm involves identifying k points(centroids) where k is the number of clusters
#from these points, calculate the distance to all other points and judging by this distance,
#classify these points as to belonging to whichever cluster
# in each grouping, find their mean and find the new center ie, new centroids
#repeat process until centroids dont move. At this point, you have the centroids.
# optimization is achieved if the centroids dont move at all or move very minimally
# Disadvantage---the Kmeans algorithm tries to make clusters of equal number of data points

centroids=clf.cluster_centers_ # coordinates of the centroids
labels=clf.labels_ # the output of the algorithm which is the cluster which a point belongs to
# since the data has n_clusters set to 2, the clusters are 2, 0 and 1
colors=5*["g.","r.","c.","b.","k."]
for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=25)
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150)
plt.show()
