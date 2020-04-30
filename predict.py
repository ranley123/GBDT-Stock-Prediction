import pandas as pd # analysis
import numpy as np # calculation
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import ensemble

# read data set
data = pd.read_csv('data.csv') # read training data

# record the unique 12 stock codes into a list
stocks = []
stock_codes = list(data.iloc[:, 0].unique())

# for each stock code, extract corresponding data instances
for code in stock_codes:
    df = data[data['股票代码_Stkcd'] == code]
    stocks.append(df)

# for each stock predict
for i in range(len(stocks)):
    data = stocks[i]
    # delete the stock code
    data = data.iloc[:,1:]
    # split to x and y
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    # split into training, testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    
    # build a model
    params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2,
            'learning_rate': 0.0001, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    me = mean_absolute_error(y_test, y_pred)
    
    # print information
    print('-----------------------------------')
    print('Stock Code: ', stock_codes[i])
    print("MSE: %.6f" % mse)
    print('me: ', me)

    '''
    -----------------------------------
    Stock Code:  672
    MSE: 0.000864
    me:  0.022254277293874802
    -----------------------------------
    Stock Code:  63
    MSE: 0.002332
    me:  0.035109195328277346
    -----------------------------------
    Stock Code:  2302
    MSE: 0.000830
    me:  0.019881888097589197
    -----------------------------------
    Stock Code:  2307
    MSE: 0.001288
    me:  0.02550267491394151
    -----------------------------------
    Stock Code:  2773
    MSE: 0.000897
    me:  0.02380382597393028
    -----------------------------------
    Stock Code:  2792
    MSE: 0.000489
    me:  0.016330879865239956
    -----------------------------------
    Stock Code:  2821
    MSE: 0.001109
    me:  0.02607645004735473
    -----------------------------------
    Stock Code:  600487
    MSE: 0.000935
    me:  0.02323362063247149
    -----------------------------------
    Stock Code:  600535
    MSE: 0.000809
    me:  0.022143157406598807
    -----------------------------------
    Stock Code:  600545
    MSE: 0.000132
    me:  0.008671967475127924
    -----------------------------------
    Stock Code:  600585
    MSE: 0.000580
    me:  0.018715383698890888
    -----------------------------------
    Stock Code:  600801
    MSE: 0.000806
    me:  0.02026521755367372
    '''