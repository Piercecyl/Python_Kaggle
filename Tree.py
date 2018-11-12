from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
# -----------------------------------------------------Input data-------------------------------------------------------
train = pd.read_csv('D:/kaggle/House Price/processed_train.csv')
test = pd.read_csv('D:/kaggle/House Price/processed_test.csv')
# Split data
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2)
# ----------------------------------------------------RandomForest------------------------------------------------------
# Select parameter
from sklearn.ensemble import RandomForestRegressor
# parameters = {
#     "n_estimators": [100,  200,  300, 400, 500] # Test out various amounts of trees in the forest
# }
# forest = RandomForestRegressor()
# grid = GridSearchCV(forest, parameters, n_jobs=-1)
# grid.fit(X_train, y_train)
# grid.score(X_test, y_test)
# grid.best_params_
# Run Random forest
ran_forest = RandomForestRegressor(n_estimators=400, n_jobs=-1)
y_pre_randomforest = ran_forest.fit(X_train, y_train).predict(X_test)
random_score = ran_forest.score(X, y)
random_r2 = r2_score(y_test, y_pre_randomforest)
random_mrse = mean_squared_error(y_test, y_pre_randomforest)
# ----------------------------------------------------XgBoost-----------------------------------------------------------
# Parameter Adjustment
# from xgboost import XGBRegressor
# cv_params = {'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.7]}
# other_params = {'learning_rate': 0.1, 'n_estimators': 120, 'max_depth': 3, 'min_child_weight': 2, 'seed': 0,
#                     'subsample': 0.6, 'colsample_bytree': 0.7, 'gamma': 0.1, 'reg_alpha': 0.05, 'reg_lambda': 1}
#
# model = XGBRegressor(**other_params)
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
# optimized_GBM.fit(X_train, y_train)
# evalute_result = optimized_GBM.cv_results_
# print('樹數量：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


xg = XGBRegressor(learning_rate = 0.1, n_estimators = 120, max_depth = 3, min_child_weight = 2, seed = 0,
                    subsample = 0.6, colsample_bytree = 0.7, gamma = 0.1, reg_alpha = 0.05, reg_lambda = 1)
y_pre_xg = xg.fit(X_train, y_train).predict(X_test)
xg_score = xg.score(X, y)
xg_r2 = r2_score(y_test, y_pre_xg)
xg_mrse = mean_squared_error(y_test, y_pre_xg)
# --------------------------------------------GradientBoostingRegressor-------------------------------------------------
from sklearn.ensemble import GradientBoostingRegressor
# cv_params = {'n_estimators': [450, 500, 550]}
# other_params = {'n_estimators': 400, 'max_depth': 10, 'min_samples_split': 2}
#
# model = GradientBoostingRegressor(**other_params)
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=-1)
# optimized_GBM.fit(X_train, y_train)
# evalute_result = optimized_GBM.cv_results_
# print('樹數量：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

gbr = GradientBoostingRegressor()
y_pre_gbr = gbr.fit(X_train, y_train).predict(X_test)
xg_score = xg.score(X, y)
gbr_r2 = r2_score(y_test, y_pre_gbr)
gbr_mrse = mean_squared_error(y_test, y_pre_gbr)
# -------------------------------------------------Output---------------------------------------------------------------
from prettytable import PrettyTable
result = PrettyTable(['model', 'accuracy', 'R^2'])
result.add_row(['RandomForest', random_score, random_r2])
result.add_row(['XgBoost', xg_score, xg_r2])
result.add_row(['GradientBoostingRegressor', xg_score, gbr_r2])
print(result)
