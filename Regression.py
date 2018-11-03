import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
all = pd.read_csv('D:/kaggle/House Price/all.csv')
# Training data
df_train = all.iloc[:1460, :]
# Test data
df_test = all.iloc[1460:, :]
df_test = df_test.drop('SalePrice', axis=1)
# Split data
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# ------------------------------------------------------------------------------------
# Regression
linreg = LinearRegression()
y_pred = linreg.fit(X_train, y_train).predict(X_test)
accuary_regression = linreg.score(X, y)
r_square_regression = metrics.explained_variance_score(y_test, y_pred)

# ------------------------------------------------------------------------------------
# Lasso
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
accuary_lasso = lasso.score(X, y)
r_square_lasso = metrics.explained_variance_score(y_test, y_pred_lasso)

# ------------------------------------------------------------------------------------
# ElasticNet
from sklearn.linear_model import ElasticNet
elasticn = ElasticNet(alpha=0.1, l1_ratio=0.7)
y_pred_ela = elasticn.fit(X_train, y_train).predict(X_test)
accuary_elas = elasticn.score(X, y)
r_square_ela = metrics.explained_variance_score(y_test, y_pred_ela)

print('Regression：', accuary_regression, r_square_regression)
print('Lasso：', accuary_lasso, r_square_lasso)
print('ElasticNet：', accuary_elas, r_square_ela)