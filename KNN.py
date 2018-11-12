from sklearn.neighbors import KNeighborsRegressor
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
# --------------------------------------------------------KNN-----------------------------------------------------------
# Select parameter
parameters = {
    "n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 19, 20] # Test out various amounts of trees in the forest
}
KNN = KNeighborsRegressor()
grid = GridSearchCV(KNN, parameters, n_jobs=-1)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)
grid.best_params_

knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
y_pred_knn = knn.fit(X_train, y_train).predict(X_test)
knn_score = knn.score(X_test, y_test)
knn_r2 = r2_score(y_test, y_pred_knn)
knn_mrse = mean_squared_error(y_test, y_pred_knn)
# -------------------------------------------------Output---------------------------------------------------------------
from prettytable import PrettyTable
result = PrettyTable(['model', 'accuracy', 'R^2', 'MRSE'])
result.add_row(['KNN', knn_score, knn_r2, knn_mrse])
print(result)

