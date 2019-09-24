import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.linear_model as skl_lm

import statsmodels.api as sm
import statsmodels.formula.api as smf

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

# Plot Settings
fig = plt.figure(figsize=(12,5))
gs = mpl.gridspec.GridSpec(1, 4)
ax1 = plt.subplot(gs[0,:-2])
ax2 = plt.subplot(gs[0,-2])
ax3 = plt.subplot(gs[0,-1])


df = pd.read_csv('./dataset/dataset_hide-on-bush.csv')
column = list()
champ_col = ['c' + str(i) for i in range(145)]
#column = ['champ_' + str(i) for i in range(145)]
column.extend(champ_col)
column.append('myPick')
column.append('result')
df.columns = column

#df = df.drop(df.columns[[i for i in range(60)]], axis=1)

#predictors = df[:145]

#X_train = predictors.values.reshape(-1, 1)
#Y = df[146]

X_train = df[:1000]     
X_test = df[1000:]

#X_train_predictors = X_train[[i for i in range(145)]]
'''
X_train_predictors = X_train[[0]].values.reshape(-1, 1)   # Data : Leblanc
X_train_response = X_train.result

X_test_predictors = X_test[[0]].values.reshape(-1, 1)   # Data : Leblanc
X_test_response = X_test.result


clf = skl_lm.LogisticRegression(solver='newton-cg')
clf.fit(X_train_predictors, X_train_response)
prob = clf.predict_proba(X_test_predictors)

print(clf)
print('classes: ',clf.classes_)
print('coefficients: ',clf.coef_)
print('intercept :', clf.intercept_)

'''
'''
result = df.result


df_e = pd.DataFrame({'x':df.c1, 'y':result})

est = smf.logit('y~x', df_e).fit()
print(est.summary())
'''
'''
y = np.ravel(df.result)
X = sm.add_constant(df[champ_col])

est = sm.Logit(df.result.ravel(), X).fit()
print(est.summary())
'''
formula = "result ~ c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15 + c16 + c17 + c18 + c19 + c20 + c21 + c22 + c23 + c24 + c25 + c26 + c27 + c28 + c29 + c30 + c31 + c32 + c33 + c34 + c35 + c36 + c37 + c38 + c39 + c40 + c41 + c42 + c43 + c44 + c45 + c46 + c47 + c48 + c49 + c50 + c51 + c52 + c53 + c54 + c55 + c56 + c57 + c58 + c59 + c60 + c61 + c62 + c63 + c64 + c65 + c66 + c67 + c68 + c69 + c70 + c71 + c72 + c73 + c74 + c75 + c76 + c77 + c78 + c79 + c80 + c81 + c82 + c83 + c84 + c85 + c86 + c87 + c88 + c89 + c90 + c91 + c92 + c93 + c94 + c95 + c96 + c97 + c98 + c99 + c100 + c101 + c102 + c103 + c104 + c105 + c106 + c107 + c108 + c109 + c110 + c111 + c112 + c113 + c114 + c115 + c116 + c117 + c118 + c119 + c120 + c121 + c122 + c123 + c124 + c125 + c126 + c127 + c128 + c129 + c130 + c131 + c132 + c133 + c134 + c135 + c136 + c137 + c138 + c139 + c140 + c141 + c142 + c143 + c144"
est = smf.logit("result ~ c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15 + c16 + c17 + c18 + c19 + c20 + c21 + c22 + c23 + c24 + c25 + c26 + c27 + c28 + c29 + c30 + c31 + c32 + c33 + c34 + c35 + c36 + c37 + c38 + c39 + c40 + c41 + c42 + c43 + c44 + c45 + c46 + c47 + c48 + c49 + c50", data=df).fit()
print(est.summary())