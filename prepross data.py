import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
# --------------------------------------------------Input Data----------------------------------------------------------
train = pd.read_csv('D:/kaggle/House Price/train.csv', index_col='Id')
train = train[train['GrLivArea'] < 4000]
test = pd.read_csv('D:/kaggle/House Price/test.csv', index_col='Id')
df_all = pd.concat([train, test], axis=0, sort=True)
# ----------------------------------------------Dealing with NA values--------------------------------------------------
col_na = df_all.isnull().sum()
col_na = col_na[col_na > 0]
# print(col_na.sort_values(ascending = False))
# Meaningful NA Values(Some Na is meaningful)
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2']
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
# No garage - fill areas/counts with 0
df_all.GarageArea.fillna(0, inplace=True)
df_all.GarageCars.fillna(0, inplace=True)
# Colmns LotFrontage：Used Ridge regression to find the best fit values
def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())
df_lotf = pd.get_dummies(df_all.drop('SalePrice', axis=1))
for col in df_lotf.drop('LotFrontage', axis=1).columns:
    df_lotf[col] = scale_minmax(df_lotf[col])
lf_train = df_lotf.dropna()
lf_train_y = lf_train.LotFrontage
lf_train_X = lf_train.drop('LotFrontage', axis=1)
lr = Ridge()
lr.fit(lf_train_X, lf_train_y)
lf_test = df_lotf.LotFrontage.isnull()
X = df_lotf[lf_test].drop('LotFrontage', axis=1)
y = lr.predict(X)
df_all.loc[lf_test, 'LotFrontage'] = y
# Remaining NA
df_last = df_all.drop('SalePrice', axis=1)
col_na = df_last.isnull().sum()
col_na = col_na[col_na > 0]
for col in col_na.index:
    df_all[col].fillna(df_all[col].mode()[0], inplace=True)
# -----------------------------------------Transfer discrete into category----------------------------------------------
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
# ---------------------------------------------Data scale normalization-------------------------------------------------
# Log 'SalePrice'
df_all.SalePrice = np.log(df_all.SalePrice)
# Standardization numeric data
def scale_standare(col):
    return (col-col.mean())/(np.std(col))
df_num = df_all.dtypes[df_all.dtypes != object].index
df_num = df_num.drop('SalePrice')
df_all[df_num] = df_all[df_num].apply(scale_standare, axis=0)
# ----------------------------------------------------one-hot encode----------------------------------------------------
# processing data with one-hot encode：一個方法讓各屬性距離原點是相同距離
# select which features to use (all for now)
model_cols = df_all.columns
# encode categoricals
df_model = pd.get_dummies(df_all[model_cols])
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
# --------------------------------------------------Drop outlier Data---------------------------------------------------
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
    print('R2=', model.score(X, y))
    print('rmse=', mean_squared_error(y, y_pred))
    print('---------------------------------------')
    print(len(outliers), 'outliers:')
    print(outliers.tolist())
    return outliers
train = df_model
train = train.iloc[:1456, :]
X = train.drop('SalePrice', axis=1)
y = train.SalePrice
outliers = find_outliers(Ridge(), X, y)
df_model = df_model.drop(outliers)
# ------------------------------------------------Feature Selection-----------------------------------------------------
# Identify Correlated Feature and remove them
train = df_model
train = train.iloc[:1437, :]
threshold = 0.9
corr_matrix = train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
df_model = df_model.drop(columns=to_drop)
# Feature Selection through Feature importance
X = df_model.drop('SalePrice', axis=1).iloc[:1437, :]
y = np.exp(df_model.SalePrice.iloc[:1437]).astype('int')
embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=1000), threshold='1.25*median')
embeded_rf_selector.fit(X, y)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:, embeded_rf_support].columns.tolist()
embeded_rf_feature.append('SalePrice')
df_model = df_model[embeded_rf_feature]
# # -----------------------------------------------PCA------------------------------------------------------------------
# from sklearn.decomposition import PCA
# pca = PCA(n_components=80)
# processed_train = df_model
# processed_train = processed_train.iloc[:1437, :]
# processed_train = processed_train.drop('SalePrice', axis=1)
# processed_test = df_model
# processed_test = processed_test.iloc[1437:, :]
# processed_test = processed_test.drop('SalePrice', axis=1)
# processed_train = pca.fit_transform(processed_train)
# processed_test = pca.fit_transform(processed_test)
# ---------------------------------------Generate Training data and test data-------------------------------------------
processed_train = df_model
processed_train = processed_train.iloc[:1437, :]
processed_test = df_model
processed_test = processed_test.iloc[1437:, :]
processed_train = pd.DataFrame(processed_train)
processed_test = pd.DataFrame(processed_test)
processed_train.to_csv('D:/kaggle/House Price/processed_train.csv')
processed_test.to_csv('D:/kaggle/House Price/processed_test.csv')
