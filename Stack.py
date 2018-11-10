from mlxtend.regressor import StackingRegressor
import numpy as np
import pandas as pd
sta = StackingRegressor(regressors=[linreg, lasso, elasticn], meta_regressor=elasticn)
fin_pred = sta.fit(X, y).predict(test)
y_pred = np.exp(fin_pred)
y_pred = pd.DataFrame(fin_pred)
y_pred.to_csv('D:/kaggle/House Price/stack.csv')

print("Mean Squared Error: %.4f" % np.mean((fin_pred - y) ** 2))