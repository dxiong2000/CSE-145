import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
from IPython.display import display

data = pd.read_csv('../data/Customer_Churn_processed.csv')

X = data.drop(columns=['LEAVE'])
y = data['LEAVE']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=True)

model = DecisionTreeClassifier(max_depth=10)
model = model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(accuracy_score(y_test, y_predict))

# EXPORTING TREE AS PNG
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
#
# dot_data = tree.export_graphviz(model, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.format = 'png'
# graph.render("tree")

randforest = RandomForestClassifier(max_depth=10)
randforest = randforest.fit(X_train, y_train)
y_predict = randforest.predict(X_test)

print(accuracy_score(y_test, y_predict))