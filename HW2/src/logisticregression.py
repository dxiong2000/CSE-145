import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('../data/Customer_Churn_processed.csv')
X = data.drop(columns=['LEAVE'])
y = data['LEAVE']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=True)