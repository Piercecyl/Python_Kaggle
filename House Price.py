import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from datetime import datetime
from sklearn.linear_model import Ridge, RidgeCV
from mlxtend.regressor import StackingRegressor
from sklearn.metrics import mean_squared_error
from scipy.special import boxcox1p
import warnings
warnings.filterwarnings('ignore')
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
# ----------------------------------------------Input Data--------------------------------------------------------------
start_time = timer(None)
train = pd.read_csv('D:/kaggle/House Price/train.csv', index_col='Id')
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
sale_price = train['SalePrice']
train = train.drop('SalePrice', axis=1)
test = pd.read_csv('D:/kaggle/House Price/test.csv', index_col='Id')
df_all = pd.concat([train, test], axis=0, sort=True)
def feature_engineer(df_all, sale_price):
    # -----------------------------------Dealing with NA values---------------------------------------------------------
    # col_na = df_all.isnull().sum()
    # col_na = col_na[col_na > 0]
    # print(col_na.sort_values(ascending = False))
    # Meaningful NA Values(Some Na is meaningful)
    cols_fillna = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu',
                   'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType',
                   'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']
    for cols in cols_fillna:
        df_all[cols].fillna(value='None', inplace=True)
    # GarageYrBlt NA: no garage. Fill with property YearBuilt
    df_all.loc[df_all.GarageYrBlt.isnull(), 'GarageYrBlt'] = df_all.loc[df_all.GarageYrBlt.isnull(), 'YearBuilt']
    # No masonry veneer - fill area with 0
    df_all['MasVnrArea'].fillna(value=0, inplace=True)
    # No basement - fill areas/counts with 0
    df_all.BsmtFullBath.fillna(0, inplace=True)
    df_all.BsmtHalfBath.fillna(0, inplace=True)
    df_all.BsmtFinSF1.fillna(0, inplace=True)
    df_all.BsmtFinSF2.fillna(0, inplace=True)
    df_all.BsmtUnfSF.fillna(0, inplace=True)
    df_all.TotalBsmtSF.fillna(0, inplace=True)
    df_all.GarageArea.fillna(0, inplace=True)
    df_all.GarageCars.fillna(0, inplace=True)
    # Colmns LotFrontage：Used Ridge regression to find the best fit values
    def scale_standardization(col):
        return (col-col.mean())/(col.std())
    # def scale_minmax(col):
    #     return (col-col.min())/(col.max() - col.min())
    df_lotf = pd.get_dummies(df_all)
    for col in df_lotf.drop('LotFrontage', axis=1).columns:
        df_lotf[col] = scale_standardization(df_lotf[col])
    lf_train = df_lotf.dropna()
    lf_train_y = lf_train.LotFrontage
    lf_train_X = lf_train.drop('LotFrontage', axis=1)
    lasso_cv = RidgeCV(alphas=np.logspace(-3, 3, 100))
    lasso_cv.fit(lf_train_X, lf_train_y)
    lr = Ridge(alpha=lasso_cv.alpha_, max_iter=1000)
    lr.fit(lf_train_X, lf_train_y)
    lf_test = df_lotf.LotFrontage.isnull()
    X = df_lotf[lf_test].drop('LotFrontage', axis=1)
    y = lr.predict(X)
    df_all.loc[lf_test, 'LotFrontage'] = y
    # Remaining NA
    col_na = df_all.isnull().sum()
    col_na = col_na[col_na > 0]
    for col in col_na.index:
        df_all[col].fillna(df_all[col].mode()[0], inplace=True)
    # -----------------------------------------Transfer discrete into category-------------------------------Categorical
    cols_ExGd = ['ExterQual','ExterCond','BsmtQual','BsmtCond',
                 'HeatingQC','KitchenQual','FireplaceQu','GarageQual',
                'GarageCond','PoolQC']
    dict_ExGd = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0}
    for col in cols_ExGd:
        df_all[col].replace(dict_ExGd, inplace=True)
    # Remaining columns
    df_all['BsmtExposure'].replace({'Gd':4,'Av':3,'Mn':2,'No':1,'None':0}, inplace=True)
    df_all['CentralAir'].replace({'Y':1,'N':0}, inplace=True)
    df_all['Functional'].replace({'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0}, inplace=True)
    df_all['GarageFinish'].replace({'Fin':3,'RFn':2,'Unf':1,'None':0}, inplace=True)
    df_all['LotShape'].replace({'Reg':3,'IR1':2,'IR2':1,'IR3':0}, inplace=True)
    df_all['Utilities'].replace({'AllPub':3,'NoSewr':2,'NoSeWa':1,'ELO':0}, inplace=True)
    df_all['LandSlope'].replace({'Gtl':2,'Mod':1,'Sev':0}, inplace=True)
    # -----------------------------------------------Adjust Skew-----------------------------------------------Numerical
    # The closer to 0, the better
    df_num = df_all.dtypes[df_all.dtypes != object].index
    skew_feature = np.abs(df_all[df_num][:2917].skew()).sort_values(ascending=False)
    skew_feature = skew_feature[skew_feature > 0.2].index
    # Box Cox
    for skew in skew_feature:
        df_all[skew] = boxcox1p(df_all[skew], 0.15)
    # Log1p
    # for skew in skew_feature:
    #     df_all[skew] = np.log1p(df_all[skew])
    # ---------------------------------------------Data scale normalization------------------------------------Numerical
    # Standardization numeric data
    df_all[df_num] = df_all[df_num].apply(scale_standardization, axis=0)
    # ----------------------------------------------------one-hot encode-------------------------------------Categorical
    # processing data with one-hot encode：一個方法讓各屬性距離原點是相同距離(沒有階級)
    # select which features to use (all for now)
    model_cols = df_all.columns
    # encode categoricals
    df_model = pd.get_dummies(df_all)
    # Rather than including Condition1 and Condition2, or Exterior1st and Exterior2nd,
    # combine the dummy variables (allowing 2 true values per property)
    if ('Condition1' in model_cols) and ('Condition2' in model_cols):
        cond_suffix = ['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNn']
        for suffix in cond_suffix:
            col_cond1 = 'Condition1_' + suffix
            col_cond2 = 'Condition2_' + suffix
            df_model[col_cond1] = df_model[col_cond1] | df_model[col_cond2]
            df_model.drop(col_cond2, axis=1, inplace=True)
    if ('Exterior1st' in model_cols) and ('Exterior2nd' in model_cols):
        # some different strings in Exterior1st and Exterior2nd for same type - rename columns to correct
        df_model.rename(columns={'Exterior2nd_Wd Shng': 'Exterior2nd_WdShing',
                                 'Exterior2nd_Brk Cmn': 'Exterior2nd_BrkComm',
                                 'Exterior2nd_CmentBd': 'Exterior2nd_CemntBd'}, inplace=True)
        ext_suffix = ['AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd',
                      'HdBoard', 'ImStucc', 'MetalSd', 'Plywood', 'Stone',
                      'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing', 'AsbShng']
        for suffix in ext_suffix:
            col_cond1 = 'Exterior1st_' + suffix
            col_cond2 = 'Exterior2nd_' + suffix
            df_model[col_cond1] = df_model[col_cond1] | df_model[col_cond2]
            df_model.drop(col_cond2, axis=1, inplace=True)
    # ------------------------------------------------Feature Selection-------------------------------------------------
    # Identify Correlated Feature and remove them(Co-linearity)
    train = df_model
    train = train.iloc[:1458, :]
    threshold = 0.9
    corr_matrix = train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_model = df_model.drop(columns=to_drop)
    # Feature Selection through Feature importance
    X = df_model.iloc[:1458, :]
    y = sale_price.astype('int')
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=1000, n_jobs=-1), threshold='1.25*median')
    embeded_rf_selector.fit(X, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:, embeded_rf_support].columns.tolist()
    df_model = df_model[embeded_rf_feature]
    # Log 'SalePrice'
    sale_price = np.log(sale_price)
    # --------------------------------------------------Drop outlier Data-----------------------------------------------
    # 預測結果後，超過三倍標準差為Outliers
    def find_outliers(model, X, y, sigma=3):
        # predict y values using model
        try:
            y_pred = pd.Series(model.predict(X), index=y.index)
        # if predicting fails, try fitting the model first
        except:
            model.fit(X, y)
            y_pred = pd.Series(model.predict(X), index=y.index)
        # calculate residuals between the model prediction and true y values
        resid = y - y_pred
        mean_resid = resid.mean()
        std_resid = resid.std()
        # calculate z statistic, define outliers to be where |z|>sigma
        z = (resid - mean_resid) / std_resid
        outliers = z[abs(z) > sigma].index
        # print and plot the results
        # print('R2=', model.score(X, y))
        # print('rmse=', mean_squared_error(y, y_pred))
        # print('---------------------------------------')
        # print(len(outliers), 'outliers:')
        # print(outliers.tolist())
        return outliers
    outliers = find_outliers(Ridge(), X, y)
    df_model = df_model.drop(outliers)
    sale_price = sale_price.drop(outliers)
    processed_train = df_model.iloc[:1438, :]
    processed_test = df_model.iloc[1438:, :]
    return processed_train, processed_test, sale_price
    # -----------------------------------------------PCA----------------------------------------------------------------
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=80)
    # np_SalePrice = df_model.SalePrice.iloc[:1440]
    # processed_train = df_model
    # processed_train = processed_train.iloc[:1440, :]
    # processed_train = processed_train.drop('SalePrice', axis=1)
    # processed_test = df_model
    # processed_test = processed_test.iloc[1440:, :]
    # processed_test = processed_test.drop('SalePrice', axis=1)
    # processed_train = pca.fit_transform(processed_train)
    # processed_test = pca.fit_transform(processed_test)
    # processed_train = pd.DataFrame(processed_train)
    # processed_test = pd.DataFrame(processed_test)
    # return processed_train, processed_test, np_SalePrice
train, test, sale_price = feature_engineer(df_all, sale_price)
# ----------------------------------------------Input processed data----------------------------------------------------
# Split data
X = train
y = sale_price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2)
# -------------------------------------------------Regression-----------------------------------------------------------
from sklearn.linear_model import LinearRegression
models = []
linreg = LinearRegression()
y_pred = linreg.fit(X_train, y_train).predict(X_test)
models.append(('linreg', LinearRegression()))
# ----------------------------------------------------Lasso-------------------------------------------------------------
from sklearn.linear_model import Lasso, LassoCV
# Select the best alpha
lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 100))
lasso_cv.fit(X_train, y_train)
# run Lasso
lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=1000)
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
models.append(('lasso', Lasso(alpha=lasso_cv.alpha_, max_iter=1000)))
# ---------------------------------------------------Ridge--------------------------------------------------------------
lasso_cv = RidgeCV(alphas=np.logspace(-3, 3, 100))
lasso_cv.fit(X_train, y_train)
# View alpha
ridge = Ridge(alpha=lasso_cv.alpha_, max_iter=1000)
y_pred_ridge = ridge.fit(X_train, y_train).predict(X_test)
models.append(('ridge', Ridge(alpha=lasso_cv.alpha_, max_iter=1000)))
# ------------------------------------------------ElasticNet------------------------------------------------------------
from sklearn.linear_model import ElasticNet, ElasticNetCV
# Select the best alpha
ElasticNetCV_cv = ElasticNetCV(alphas=np.logspace(-3, 3, 100), l1_ratio=np.logspace(-3, 3, 100))
ElasticNetCV_cv.fit(X_train, y_train)
# Use best alpha and l1_ratio
elasticn = ElasticNet(alpha=ElasticNetCV_cv.alpha_, l1_ratio=ElasticNetCV_cv.l1_ratio_, max_iter=1000)
y_pred_ela = elasticn.fit(X_train, y_train).predict(X_test)
models.append(('elasticn', ElasticNet(alpha=ElasticNetCV_cv.alpha_, l1_ratio=ElasticNetCV_cv.l1_ratio_, max_iter=1000)))
# ------------------------------------------------RandomForest----------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
parameters = {
    "n_estimators": range(100, 501, 50),
}
forest = RandomForestRegressor()
optimized_RF = GridSearchCV(forest, parameters, n_jobs=-1)
optimized_RF.fit(X_train, y_train)
# Run Random forest
ran_forest = optimized_RF.best_estimator_
y_pre_randomforest = ran_forest.fit(X_train, y_train).predict(X_test)
models.append(('ran_forest', optimized_RF.best_estimator_))
# ---------------------------------------------------XgBoost------------------------------------------------------------
from xgboost import XGBRegressor
parameters = {
        'n_estimators': range(100, 1001, 100),
        'max_depth': range(1, 15, 1)
        }
model = XGBRegressor()
optimized_XGB = GridSearchCV(model, parameters, scoring='r2', cv=5, n_jobs=-1)
optimized_XGB.fit(X_train, y_train)

xg = optimized_XGB.best_estimator_
y_pre_xg = xg.fit(X_train, y_train).predict(X_test)
models.append(('xgboost', optimized_XGB.best_estimator_))
# --------------------------------------------GradientBoostingRegressor-------------------------------------------------
from sklearn.ensemble import GradientBoostingRegressor
parameters = {
    'n_estimators': range(100, 1001, 100),
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': range(2, 6, 1)
    }
model = GradientBoostingRegressor()
optimized_GBR = GridSearchCV(model, parameters, scoring='r2', cv=5, n_jobs=-1)
optimized_GBR.fit(X_train, y_train)
gbr = optimized_GBR.best_estimator_
y_pre_gbr = gbr.fit(X_train, y_train).predict(X_test)
models.append(('GradientBoostingRegressor', optimized_GBR.best_estimator_))
# ---------------------------------------------------SVR----------------------------------------------------------------
parameters = {
    'C': range(1, 21, 2),
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'gamma': ['auto']
    }
model = SVR()
optimized_SVR = GridSearchCV(model, parameters, scoring='r2', cv=5, verbose=1, n_jobs=-1)
optimized_SVR.fit(X_train, y_train)
# Run model
svr = optimized_SVR.best_estimator_
y_pred_SVR = svr.fit(X_train, y_train).predict(X_test)
models.append(('svr', optimized_SVR.best_estimator_))
# ---------------------------------------------------LinearSVR----------------------------------------------------------
parameters = {
    'C': np.arange(0.1, 1, 0.1),
    'max_iter': [5000]
}
model = LinearSVR()
optimized_LSVR = GridSearchCV(model, parameters, scoring='r2', cv=5, verbose=1, n_jobs=-1)
optimized_LSVR.fit(X_train, y_train)

lin_SVR = optimized_LSVR.best_estimator_
y_pred_lin_SVR = lin_SVR.fit(X_train, y_train).predict(X_test)
models.append(('lin_SVR', optimized_LSVR.best_estimator_))
# ---------------------------------------------------KNN----------------------------------------------------------------
parameters = {
    "n_neighbors": np.arange(3,11,1)
}
KNN = KNeighborsRegressor()
grid_KNN = GridSearchCV(KNN, parameters, n_jobs=-1)
grid_KNN.fit(X_train, y_train)
knn = grid_KNN.best_estimator_
y_pred_knn = knn.fit(X_train, y_train).predict(X_test)
models.append(('knn', grid_KNN.best_estimator_))
# --------------------------------------------------Output--------------------------------------------------------------
def BasedLines(X_train, y_train, models):
    results = []
    names = []
    for name, model in models:
        kfold = RepeatedKFold(n_splits=10, n_repeats=5)
        cv_result = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
        results.append(cv_result)
        names.append(name)
    return names, results
def Score(names, results):
    def floatingDecimals(f_val, dec=3):
        prc = "{:." + str(dec) + "f}"
        return float(prc.format(f_val))
    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(), 6))
    scoreDataFrame = pd.DataFrame({'Model': names, 'RMSE': scores})
    return print(scoreDataFrame.sort_values('RMSE'))
names, results = BasedLines(X, y, models)
Score(names, results)
timer(start_time)
# --------------------------------------------------Stack--------------------------------------------------------------
# Test
# sta = StackingRegressor(regressors=[lasso, ridge, elasticn, linreg, xg], meta_regressor=ridge)
# fin_pred = sta.fit(X_train, y_train).predict(X_test)
# print("Mean Squared Error:", mean_squared_error(y_test, fin_pred))
# # Fin results
# fin_pred = sta.fit(X, y).predict(test)
# y_pred = np.exp(fin_pred)
# y_pred = pd.DataFrame({'Id': range(1461, 2920, 1), 'SalePrice': y_pred})
# y_pred.to_csv('D:/kaggle/House Price/sample_submission1.csv', index=False)


























