import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score