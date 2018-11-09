from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# -----------------------------------------------------Input data-------------------------------------------------------
train = pd.read_csv('D:/kaggle/House Price/processed_train.csv')
test = pd.read_csv('D:/kaggle/House Price/processed_test.csv')
# Split data
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2)
# ---------------------------------------------------------SVR----------------------------------------------------------
# Select parameter
cv_params = {'epsilon': [0.02, 0.03, 0.04, 0.05, 0.06]}
other_params = {'gamma': 0.00015, 'C': 16, 'epsilon': 0.02}

model = SVR(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(X_train, y_train)
evalute_result = optimized_GBM.cv_results_
print('樹數量：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
# Run model
svr = SVR(gamma = 0.00015, C = 16, epsilon = 0.02)
y_pred_SVR = svr.fit(X_train, y_train).predict(X_test)
SVR_score = svr.score(X, y)
SVR_r2 = r2_score(y_test, y_pred_SVR)
# ---------------------------------------------------LinearSVR----------------------------------------------------------
# Select parameter
cv_params = {'C': [0.1, 0.01, 0.2, 0.3, 0.001, 0.0001, 0.00001]}
other_params = {'C': 16, 'max_iter': 100000}

model = LinearSVR(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=-1)
optimized_GBM.fit(X_train, y_train)
evalute_result = optimized_GBM.cv_results_
print('樹數量：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

lin_SVR = LinearSVR(C=0.1, max_iter=100000)
y_pred_lin_SVR = lin_SVR.fit(X_train, y_train).predict(X_test)
lin_SVR_score = lin_SVR.score(X, y)
lin_SVR_r2 = r2_score(y_test, y_pred_lin_SVR)
# ----------------------------------------------------------------Output------------------------------------------------
from prettytable import PrettyTable
result = PrettyTable(['model', 'R^2_x', 'R^2'])
result.add_row(['SVR', SVR_score, SVR_r2])
result.add_row(['LinearSVR', lin_SVR_score, lin_SVR_r2])
print(result)

























