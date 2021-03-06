To best evaluate a model's performance in terms of accuracy, one must always evaluate
based on a baseline skill.
For Classification problems, the baseline is the naive score
For regrssion problems, the mean value is the naive score
In context of classification, a naive model is one that predicts all test data to be one
of the target labels and hence, the number of naive predictions depends on the number of labels in the dataset
#######
ACCURACY PARADOX
#######
An occurence where the training data has a large class imbalance such that the majority of the training set is of one output label
In this case, any other predictive model would be beaten by the naive model and as such,
there should be other methods of evaluating the best algorithm for such a situation.
These would inlcude:

PRECISION
A measure of a classifiers exactness. A low value suggests a high number of false positives

RECALL/True Positive Rate
Can be thought of a measure of classifier completeness.  low recall may indicate many false negatives

F1-SCORE

Coonveys the balance between the precision and recall.


+++++++++++++++++++++++++++
The above descriptions are just explanations/interpretations and not the actual
technical definnitions. The definitions and formulas can be found online
+++++++++++++++++++++++++++


Matthews Correlation Coeffecient(MCC)
Used as a ameasure of the strength of quality of binary classifications.
Return Values range from -1 to +1
-1---Total disagreement between predicted and actual val
0-- indicates that the model is no better than random guessing
+1- indicates perfect agreement between predicted and actual value

# FURTHER READING ON MCC#
https://en.wikipedia.org/wiki/Matthews_correlation_coefficient