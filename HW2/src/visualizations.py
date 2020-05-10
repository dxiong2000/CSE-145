import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import Artist as ax
from textwrap import wrap
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer

def income_vs_churn():
    data = pd.read_csv('../data/Customer_Churn_processed.csv')
    df = pd.DataFrame(data, columns=['INCOME','LEAVE'])
    '''
    range1: 20-55k
    range2: 55-90k
    range3: 90-125k
    range4: 125-160k
    [#_STAY, #_LEAVE]
    '''
    range1 = [0, 0]
    range2 = [0, 0]
    range3 = [0, 0]
    range4 = [0, 0]
    for income, leave in zip(df['INCOME'], df['LEAVE']):
        if income <= 55000:
            if leave == 0:
                range1[0] += 1
            else:
                range1[1] += 1
            continue
        elif income <= 90000:
            if leave == 0:
                range2[0] += 1
            else:
                range2[1] += 1
            continue
        elif income <= 125000:
            if leave == 0:
                range3[0] += 1
            else:
                range3[1] += 1
            continue
        else:
            if leave == 0:
                range4[0] += 1
            else:
                range4[1] += 1

    # set width of bar
    barWidth = 0.2

    # Set position of bar on X axis
    r1 = np.arange(len(range1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    # Make the plot
    plt.bar(r1, range1, color='#FF5733', width=barWidth, edgecolor='white', label='$20k-55k')
    plt.bar(r2, range2, color='#C70039', width=barWidth, edgecolor='white', label='$55k-90k')
    plt.bar(r3, range3, color='#900C3F', width=barWidth, edgecolor='white', label='$90k-125k')
    plt.bar(r4, range4, color='#581845', width=barWidth, edgecolor='white', label='$125k-160k')

    # Add xticks on the middle of the group bars
    plt.xlabel('Churn', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.title('Churn based on Income Ranges')
    plt.xticks([r + barWidth for r in range(len(range1))], ["STAY", "LEAVE"])

    plt.legend()
    plt.show()

def information_gain():
    data = pd.read_csv('../data/Customer_Churn_processed.csv')
    df_leave = data['LEAVE']
    data.drop(columns=['LEAVE'], inplace=True)

    # gets information gain values for each attribute
    res = list(zip(list(data.columns), mutual_info_classif(data, df_leave, discrete_features=True)))
    # sort IG values in decreasing order
    res = sorted(res, key=lambda x: x[1], reverse=True)
    # unzip lists
    (labels, info_gain) = zip(*res)

    # plot information gain bar chart
    # ['HOUSE','INCOME', 'OVERAGE', 'HANDSET', 'OVER_15MIN', 'LEFTOVER', "AVG_CALL", "REPORT_SATIS", "COLLEGE", "CHANGE_PLAN", "REP_USAGE"]
    plt.bar(['A', 'B', 'C', 'D', 'E', 'F', "G", "H", "I", "J", "K"], list(info_gain))
    for i, v in enumerate(list(info_gain)):
        plt.text(i - 0.45, v + 0.01, str('%.3f'%(v)))
    plt.xlabel('Attributes', fontweight='bold')
    plt.ylabel('Information Gain', fontweight='bold')
    plt.title('Attributes ranked by Information Gain')
    plt.show()

income_vs_churn()
information_gain()

