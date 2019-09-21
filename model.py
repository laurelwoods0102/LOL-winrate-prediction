import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# Setting name
myName = "hide-on-bush"
'''
df = pd.read_csv('./dataset/dataset_{}.csv'.format(myName))
column_name = [i for i in range(145)]
column_name.append("result")
column_name.append("myPick")
df.columns = column_name

df.plot.bar()
plt.show()
#print(df.sum(axis=0))



Y = df[145]
X = df[:144]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

log_clf = LogisticRegression()
log_clf.fit(X_train, Y_train)
print(log_clf.score(X_test, Y_test))
'''
df = pd.read_csv('./result/raw_dataset_hide-on-bush.csv')
column = ['result', 'myPick', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
df.columns = column

df = df.drop(df.columns[[0, 1]], axis=1)
df.plot.bar()
plt.show()