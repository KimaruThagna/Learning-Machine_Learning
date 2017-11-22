# The start of unsupervised learning
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn import preprocessing,cross_validation
import pandas as pd
# # flat clustering example algorithm-Kmeans
# #hierachical clustering example algorithm- Meanshift
# X=np.array([
#     [1,2],
#     [1.5,1.8],
#     [5,8],
#     [8,8],
#     [1,0.6],
#     [9,11]
# ])
# # plt.scatter(X[:,0],X[:,1],s=150,linewidths=5) #X[:,0] returns the zeroeth elements
# # # of the items in the x array X[:,1] returns all the elements in pos 1 of te items in the X array
# # plt.show()
# clf=KMeans(n_clusters=2) #the number of clusters should be lessthan or
# #equal to your datapoints but cannot exceed
# clf.fit(X) # train the algorithm
#
# # training the algorithm involves identifying k points(centroids) where k is the number of clusters
# #from these points, calculate the distance to all other points and judging by this distance,
# #classify these points as to belonging to whichever cluster
# # in each grouping, find their mean and find the new center ie, new centroids
# #repeat process until centroids dont move. At this point, you have the centroids.
# # optimization is achieved if the centroids dont move at all or move very minimally
# # Disadvantage---the Kmeans algorithm tries to make clusters of equal number of data points
#
# centroids=clf.cluster_centers_ # coordinates of the centroids
# labels=clf.labels_ # the output of the algorithm which is the cluster which a point belongs to
# # since the data has n_clusters set to 2, the clusters are 2, 0 and 1
# colors=5*["g.","r.","c.","b.","k."]
# for i in range(len(X)):
#     plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=25)
# plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150)
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
df=pd.read_excel('titanic.xls',sheetname='titanic3')

df.drop(['body','name'],1,inplace=True)
df.convert_objects(convert_numeric=True)# convert df to numeric
df.fillna(0,inplace=True) # file NaN values with 0s
print(df.head())

# non numeric data needs to be mapped to a numeric equivalent for ML algorithms to process
def convert_non_numeric_data(df):
    columns=df.columns.values # obtain coulmn headers
    for col in columns:
        text_digit_vals={} #structure will be {"txt":digit,"txt":digit}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[col].dtype!=np.int64 and df[col].dtype!=np.float64:
            contents=df[col].values.tolist() # convert column contents to list
            unique_elements=set(contents) #this finds non repeating elements
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1
            df[col]=list(map(convert_to_int,df[col]))# maps the dictionary
            #retuned by the function to the column in the data frame
    return df
df=convert_non_numeric_data(df)
#print(df.head())
X=np.array(df.drop(['survived'],1).astype(float))
X=preprocessing.scale(X)# scale the values so that they dont distort the outcome
# the feature set is everything apart from the survived column
y=np.array(df['survived'])# labels
clf=KMeans(n_clusters=2)
clf.fit(X)
correct=0
for i in range(len(X)):
    predict_me=np.array(X[i].astype(float)) # particular record in scaled df
    predict_me=predict_me.reshape(-1,len(predict_me)) #reshape record
    prediction=clf.predict(predict_me) # predict class of record from the trained clf
    if prediction[0]==y[i]: # compare prediction with survival index, 0 or 1
        correct+=1
print(correct/len(X))