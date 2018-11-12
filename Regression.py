import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
# --------------------------------------------------------Input data----------------------------------------------------
train = pd.read_csv('D:/kaggle/House Price/processed_train.csv')
test = pd.read_csv('D:/kaggle/House Price/processed_test.csv')
# Split data
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2)
# ---------------------------------------------------------Regression---------------------------------------------------
linreg = LinearRegression()
y_pred = linreg.fit(X_train, y_train).predict(X_test)
accuary_regression = linreg.score(X_train, y_train)
r_square_regression = metrics.explained_variance_score(y_test, y_pred)
mrse_regression = mean_squared_error(y_test, y_pred)
# -----------------------------------------------------------Lasso------------------------------------------------------
from sklearn.linear_model import Lasso, LassoCV
# Select the best alpha
lasso_cv = LassoCV(alphas=[0.00009, 0.000091, 0.000099, 0.0001, 0.00002, 0.00015, 0.0002])
lasso_cv.fit(X_train, y_train)
# View alpha
lasso_cv.alpha_
#
lasso = Lasso(alpha=0.00002, max_iter=50000)
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
accuary_lasso = lasso.score(X, y)
r_square_lasso = metrics.explained_variance_score(y_test, y_pred_lasso)
mrse_lasso = mean_squared_error(y_test, y_pred_lasso)
# -------------------------------------------------------------Ridge----------------------------------------------------
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=500)
y_pred_ridge = ridge.fit(X_train, y_train).predict(X_test)
accuary_ridge = lasso.score(X, y)
r_square_ridge = metrics.explained_variance_score(y_test, y_pred_ridge)
mrse_ridge = mean_squared_error(y_test, y_pred_ridge)
# -------------------------------------------------------------ElasticNet-----------------------------------------------
from sklearn.linear_model import ElasticNet, ElasticNetCV
# Select the best alpha
ElasticNetCV_cv = ElasticNetCV(alphas=[0.00009, 0.000091, 0.000099, 0.0001, 0.00002, 0.00015, 0.0002],
                               l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.003, 0.6])
ElasticNetCV_cv.fit(X_train, y_train)
# View result
ElasticNetCV_cv.alpha_
ElasticNetCV_cv.l1_ratio_
# Use best alpha and l1_ratio
elasticn = ElasticNet(alpha=0.0002, l1_ratio=0.2, max_iter=50000)
y_pred_ela = elasticn.fit(X_train, y_train).predict(X_test)
accuary_elas = elasticn.score(X, y)
r_square_ela = metrics.explained_variance_score(y_test, y_pred_ela)
mrse_ela = mean_squared_error(y_test, y_pred_ela)
# ----------------------------------------------------------------Output------------------------------------------------
from prettytable import PrettyTable
result = PrettyTable(['model', 'R^2', 'Variance Score'])
result.add_row(['Regression', accuary_regression, r_square_regression])
result.add_row(['Lasso', accuary_lasso, r_square_lasso])
result.add_row(['Ridge', accuary_ridge, r_square_ridge])
result.add_row(['ElasticNet', accuary_elas, r_square_ela])
print(result)

