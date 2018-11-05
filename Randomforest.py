from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('D:/kaggle/House Price/all_random.csv')
# Select columns with a correlation > 0.5
corr = df.corr()
rel_vars = corr.SalePrice[(corr.SalePrice > 0.5)]
rel_cols = list(rel_vars.index.values)

# Training data
df_train = df[rel_cols]
df_train = df_train.iloc[:1460, :]
# Test data
df_test = df[rel_cols]
df_test = df_test.iloc[1460:, :]
df_test = df_test.drop('SalePrice', axis=1)

# Split X, y
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

# Create random forest regression predictor
parameters = {
    "n_estimators": [100,  200,  300], # Test out various amounts of trees in the forest
    "max_features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] # Test amount of features
}
forest = RandomForestClassifier()
grid = GridSearchCV(forest, parameters, n_jobs=-1)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)
grid.best_params_


ran_forest = RandomForestClassifier( n_estimators=200, random_state=3, n_jobs=-1, max_features=6)
ran_forest.fit(X_train, y_train)
score = ran_forest.score(X_train, y_train)
y_pred = ran_forest.predict(df_test)
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('D:/kaggle/House Price/regression.csv')
















