import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_excel(r'E:/School/2020 Spring/CS145/HW2/data/Customer_Churn.xlsx')
print('Label encoding the string values in the data table...')
'''
LABEL ENCODINGS

REPORTED_SATISFACTION
very_unsat: 0
unsat:      1
avg:        2
sat:        3
very_sat:   4

REPORTED_USAGE_LEVEL
very_little:0
little:     1
avg:        2
high:       3
very_high:  4

CONSIDERING_CHANGE_OF_PLAN
never_thought:              0
no:                         1
perhaps:                    2
considering:                3
actively_looking_into_it:   4

LEAVE
STAY:  0
LEAVE: 1

COLLEGE
zero: 0
one:  1
'''
reported_satisfaction = {'very_unsat': 0, 'unsat': 1, 'avg': 2, 'sat': 3, 'very_sat': 4}
reported_usage_level = {'very_little': 0, 'little': 1, 'avg': 2, 'high': 3, 'very_high': 4}
considering_change_of_plan = {'never_thought': 0, 'no': 1, 'perhaps': 2, 'considering': 3,
                              'actively_looking_into_it': 4}
leave = {'STAY': 0, 'LEAVE': 1}
college = {'zero': 0, 'one': 1}
for i, (v1, v2, v3, v4, v5) in enumerate(
        zip(data['REPORTED_SATISFACTION'], data['REPORTED_USAGE_LEVEL'], data['CONSIDERING_CHANGE_OF_PLAN'],
            data['LEAVE'], data['COLLEGE'])):
    data.loc[i, 'REPORTED_SATISFACTION'] = reported_satisfaction[v1]
    data.loc[i, 'REPORTED_USAGE_LEVEL'] = reported_usage_level[v2]
    data.loc[i, 'CONSIDERING_CHANGE_OF_PLAN'] = considering_change_of_plan[v3]
    data.loc[i, 'LEAVE'] = leave[v4]
    data.loc[i, 'COLLEGE'] = college[v5]
data.to_csv('../data/Customer_Churn_processed.csv', index=False)
print('Done label encoding.')