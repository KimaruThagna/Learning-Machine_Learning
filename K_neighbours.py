# SUPERVISED ML ALGORITHMS HAVE LABELED DATA AND BROADLY INVOLVE
#REGRESSION AND CLASSIFICATION
#UNSUPERVISED ML ALGORITHMS DONT HAVE LABELED DATA AND THE MAIN
#OBJECTIVE IS TO DISCOVER UNDERLYING PATTERNS EG customer buying behavior and patterns
 #BROADLY INVOLVES CLUSTERING(eg k_means) AND ASSOCIATION ALGORITHMS
 #FOR DATA WITH LARGE DISPARITY, PERFORM NORMALIZATION
 #EG HEIGHT WEIGHT DATA SET(180,70)....180 MIGHT SKEW THE ALGORITHM
import matplotlib.pyplot as plt
import numpy as np
import warnings,random
import pandas as pd
from collections import Counter
from matplotlib import style
style.use('ggplot')

#ACCURACY IS HOW MANY TIMES A MODEL CLASSIFIES DATA CORRECTLY IE
#CORRECT-DATA/TOTAL-DATA
#CONFIDENCE IS HOW WELL THE MODEL HAS CLASSIFIED A PARTICULAR DATAPOINT
#IE, FOR KNN, ITS HOW MANY POINTS OUT OF THE TOTAL NEAREST NEIGHBOURS(K) BELONG TO THE MOST POPULAR CLASS\
# OR HOW CONFIDENT THE MODEL FEELS IT HAS CLASSIFIED A DATA POINT
def KNN(data,prediction,k=3):
    if len(data)>=k:
        warnings.warn("Number of classes exceeds value of k")
    distances=[]
    for group in data:
        for features in data[group]:
            euclidian_distance=np.linalg.norm(np.array(features)-np.array(prediction))
            #np.sqrt( np.sum( np.array(features)-np.array(prediction)**2 ) )
            distances.append([euclidian_distance,group])
    votes=[i[1] for i in sorted(distances)[:k]]
    # output the first 3(k=3) groups with the least distances
    vote_result=Counter(votes).most_common(1)[0][0] # return one item that is most common
    #confidence-> the % that the classifier gives in which it shows
    #how many out of the total k neighbours are in the most popular group
    confidence=Counter(votes).most_common(1)[0][1]/k
    print(confidence)
    #print(votes)
    #print(Counter(votes).most_common(1))
    # most common returns a list of tuples so pick the first tuple and the 1st element in the tuple
    return vote_result


dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
# a dataset consisting of the labels k and r(the classes)
#which have 2D labels as the figures
new_feature=[5,7]
df=pd.read_csv('Bcancer.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'],1, inplace=True)
full_data=df.astype(float).values.tolist()
random.shuffle(full_data)# shuffle data for training and testing
# print(full_data[:5])
test_size=0.25
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]
[ train_set[ i[-1] ].append(i[:-1]) for i in train_data ] #i[-1] denotes last element in list
[ test_set[ i[-1] ].append(i[:-1]) for i in test_data ]

correct=0
total=0
# test the algorithm with the data
for group in test_set: # for each class, 2 or 4
    for data in test_set[group]: # for each data list in 2...or 4
        vote=KNN(train_set,data,k=5) # pass the whole train_set and use ech occurence of data
        # as a means to gauge whether the algorithm works
        if vote==group:
            correct+=1
        total+=1
print(correct/total)
# [[plt.scatter(ii[0],ii[1],color=i) for ii in dataset[i]] for i in dataset]
# plt.show()

#print(KNN(dataset,new_feature))