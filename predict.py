import xgboost as xgb
import pandas as pd # analysis
import numpy as np # calculation
from pandas import Series, DataFrame
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import seaborn as sns

# read data set
data = pd.read_csv('data.csv') # read training data
data = data.iloc[:,1:]

# pick features that are most relevant
k = 10
cols = data.corr().nlargest(k, '日收益率_Dret')['日收益率_Dret'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar = True, annot=True, square=True, fmt='.2f',annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)

data = data[['滞后一期收益', '市盈率_PE', '市净率_PB', '市销率_PS', '每股收益(摊薄)(元/股)_EPS', '每股营业利润(元/股)_OpPrfPS', '每股净资产(元/股)_NAPS', '每股营业收入_IncomePS']]

# split into training set and test set
# data.rename("")
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.33)

params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2,
          'learning_rate': 0.001, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE: %.6f" % mse)

from sklearn.metrics import mean_absolute_error
me = mean_absolute_error(y_test, y_pred)
print('me: ', me)

# d = {'y': y_test, 'y_pred': y_pred, 'diff': abs(y_test - y_pred)}
# df = pd.DataFrame(data=d)
# temp = df[df['diff'] < me]
# print('accuracy', (len(temp)/ len(y_test)) * 100)
