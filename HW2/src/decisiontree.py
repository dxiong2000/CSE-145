import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../data/Customer_Churn_processed.csv')
data = data.to_numpy()

