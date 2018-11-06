# Recursive Feature Elimination
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
# load the iris datasets
dataset = datasets.load_iris()
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(dataset.data, dataset.target)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

# displaying relative feature importance
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(dataset.data, dataset.target)
# display the relative importance of each attribute
print(model.feature_importances_)

'''
LINKS FOR FURTHER READING
https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0
https://towardsdatascience.com/why-how-and-when-to-apply-feature-selection-e9c69adfabf2
https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization
https://www.kaggle.com/residentmario/automated-feature-selection-with-sklearn
https://www.kaggle.com/mnoori/feature-selection-for-mlr-with-python
'''