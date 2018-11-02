import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
train = pd.read_csv('D:/kaggle/House Price/train.csv', index_col='Id')
test = pd.read_csv('D:/kaggle/House Price/test.csv', index_col='Id')
df_all = pd.concat([train, test], axis=0, sort=True)
# Columns with NA values
# col_na = df_all.isnull().sum()
# # col_na = col_na[col_na > 0]
# # print(col_na.sort_values(ascending = False))
# Meaningful NA Values(Some Na is meaningful)
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2']
for cols in cols_fillna:
    df_all[cols].fillna(value='None', inplace=True)
# GarageYrBlt NA: no garage. Fill with property YearBuilt.YearBuilt
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
# 歸一化
# process LotFrontage
def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())
# Colmns LotFrontage
df_lotf = pd.get_dummies(df_all.drop('SalePrice', axis=1))
for col in df_lotf.drop('LotFrontage', axis=1).columns:
    df_lotf[col] = scale_minmax(df_lotf[col])
lf_train = df_lotf.dropna()
lf_train_y = lf_train.LotFrontage
lf_train_X = lf_train.drop('LotFrontage', axis=1)
lr = Ridge()
lr.fit(lf_train_X, lf_train_y)
print('R^2:\n', lr.score(lf_train_X, lf_train_y))
lf_test = df_lotf.LotFrontage.isnull()
X = df_lotf[lf_test].drop('LotFrontage',axis=1)
y = lr.predict(X)
df_all.loc[lf_test,'LotFrontage'] = y

# Remaining NA
df_last = df_all.drop('SalePrice', axis=1)

col_na = df_last.isnull().sum()
col_na = col_na[col_na > 0]
for col in col_na.index:
    df_all[col].fillna(df_all[col].mode()[0], inplace=True)


# Transfer discrete into
df_all.dtypes[df_all.dtypes == object].index





# def training_data(trd):
#
#     # --Remove the feature which include most na data
#     trd = trd.drop(columns=['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence'])
#     dtypes = trd.dtypes
#     cols_numeric = dtypes[dtypes != object].index.tolist()
#     cols_numeric.remove('MSSubClass')
#
#     # --Processing missing data at feature 'MasVnrArea'
#     # plt.hist(train_data['MasVnrArea'])
#     # plt.show()
#     trd['SalePrice'] = np.log(trd['SalePrice'])
#     # --Processing missing data at feature 'MasVnrArea'
#     trd['MasVnrArea'] = trd['MasVnrArea'].fillna(value = trd['MasVnrArea'].mean())
#
#     # --Processing missing data at feature 'BsmtQual'
#     trd['BsmtQual'] = trd['BsmtQual'].fillna(value = 'No')
#
#     # --Processing missing data at feature 'BsmtCond'
#     trd['BsmtCond'] = trd['BsmtCond'].fillna(value = 'No')
#
#     # --Processing missing data at feature 'BsmtFinType1'
#     trd['BsmtFinType1'] = trd['BsmtFinType1'].fillna(value = 'No')
#
#     # --Processing missing data at feature 'BsmtExposure'
#     trd['BsmtExposure'] = trd['BsmtExposure'].fillna(value = 'None')
#
#     # --Processing missing data at feature 'BsmtFinType2'
#     trd['BsmtFinType2'] = trd['BsmtFinType2'].fillna(value = 'No')
#
#     # --Processing missing data at feature 'GarageYrBlt' 、 'GarageType' 、 'GarageFinish' 、 'GarageQual' 、 'GarageCond'
#     trd['GarageYrBlt'] = trd['GarageYrBlt'].fillna(value=round(trd['GarageYrBlt'].mean(), 0))
#     trd['GarageType'] = trd['GarageType'].fillna(value='Attchd')
#     trd['GarageFinish'] = trd['GarageFinish'].fillna(value='None')
#     trd['GarageQual'] = trd['GarageQual'].fillna(value='None')
#     trd['GarageCond'] = trd['GarageCond'].fillna(value='None')
#
#     # --Processing missing data at feature 'LotFrontage'
#     trd['LotFrontage'] = trd['LotFrontage'].fillna(trd['LotFrontage'].median())
#
#
#     # --Processing missing data at feature 'FireplaceQu'
#     trd['FireplaceQu'] = trd['FireplaceQu'].fillna(value = 'No')
#
#
#     # --Processing missing data at feature 'Electrical'
#     trd['Electrical'] = trd['Electrical'].fillna('SBrkr')
#
#     # --Processing missing data at feature 'MasVnrType'
#     trd['MasVnrType'] = trd['MasVnrType'].fillna('None')
#     # print(train_data['MasVnrType'].value_counts())
#
#     # Encode ordinal data
#     trd['LotShape'] = trd['LotShape'].map({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3})
#     trd['LandContour'] = trd['LandContour'].map({'Low': 0, 'HLS': 1, 'Bnk': 2, 'Lvl': 3})
#     trd['Utilities'] = trd['Utilities'].map({'NoSeWa': 0, 'NoSeWa': 1, 'AllPub': 2})
#     trd['BldgType'] = trd['BldgType'].map({'Twnhs': 0, 'TwnhsE': 1, 'Duplex': 2, '2fmCon': 3, '1Fam': 4})
#     trd['HouseStyle'] = trd['HouseStyle'].map(
#         {'1Story': 0, '1.5Fin': 1, '1.5Unf': 2, '2Story': 3, '2.5Fin': 4, '2.5Unf': 5, 'SFoyer': 6, 'SLvl': 7})
#     trd['BsmtFinType1'] = trd['BsmtFinType1'].map(
#         {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
#     trd['BsmtFinType2'] = trd['BsmtFinType2'].map(
#         {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
#     trd['LandSlope'] = trd['LandSlope'].map({'Gtl': 0, 'Mod': 1, 'Sev': 2})
#     trd['Street'] = trd['Street'].map({'Grvl': 0, 'Pave': 1})
#     trd['MasVnrType'] = trd['MasVnrType'].map(
#         {'None': 0, 'BrkCmn': 1, 'BrkFace': 2, 'CBlock': 3, 'Stone': 4})
#     trd['CentralAir'] = trd['CentralAir'].map({'N': 0, 'Y': 1})
#     trd['GarageFinish'] = trd['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
#     trd['PavedDrive'] = trd['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2})
#     trd['BsmtExposure'] = trd['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
#     trd['ExterQual'] = trd['ExterQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
#     trd['ExterCond'] = trd['ExterCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
#     trd['BsmtCond'] = trd['BsmtCond'].map({'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
#     trd['BsmtQual'] = trd['BsmtQual'].map({'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
#     trd['HeatingQC'] = trd['HeatingQC'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
#     trd['KitchenQual'] = trd['KitchenQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
#     trd['FireplaceQu'] = trd['FireplaceQu'].map({'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
#     trd['GarageQual'] = trd['GarageQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
#     trd['GarageCond'] = trd['GarageCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
#
#     # Encode Categorical Variables
#     trd['Foundation'] = trd['Foundation'].map(
#         {'BrkTil': 0, 'CBlock': 1, 'PConc': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5})
#     trd['Heating'] = trd['Heating'].map(
#         {'Floor': 0, 'GasA': 1, 'GasW': 2, 'Grav': 3, 'OthW': 4, 'Wall': 5})
#     trd['Electrical'] = trd['Electrical'].map({'SBrkr': 0, 'FuseA': 1, 'FuseF': 2, 'FuseP': 3, 'Mix': 4})
#     trd['Functional'] = trd['Functional'].map(
#         {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7})
#     trd['GarageType'] = trd['GarageType'].map(
#         {'None': 0, 'Detchd': 1, 'CarPort': 2, 'BuiltIn': 3, 'Basment': 4, 'Attchd': 5, '2Types': 6})
#     trd['SaleType'] = trd['SaleType'].map(
#         {'Oth': 0, 'ConLD': 1, 'ConLI': 2, 'ConLw': 3, 'Con': 4, 'COD': 5, 'New': 6, 'VWD': 7, 'CWD': 8, 'WD': 9})
#     trd['SaleCondition'] = trd['SaleCondition'].map(
#         {'Partial': 0, 'Family': 1, 'Alloca': 2, 'AdjLand': 3, 'Abnorml': 4, 'Normal': 5})
#     trd['MSZoning'] = trd['MSZoning'].map(
#         {'A': 0, 'FV': 1, 'RL': 2, 'RP': 3, 'RM': 4, 'RH': 5, 'C (all)': 6, 'I': 7})
#     trd['LotConfig'] = trd['LotConfig'].map({'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4})
#     trd['Neighborhood'] = trd['Neighborhood'].map(
#         {'Blmngtn': 0, 'Blueste': 1, 'BrDale': 2, 'BrkSide': 3, 'ClearCr': 4, 'CollgCr': 5, 'Crawfor': 6, 'Edwards': 7,
#          'Gilbert': 8,
#          'IDOTRR': 9, 'MeadowV': 10, 'Mitchel': 11, 'NAmes': 12, 'NoRidge': 13, 'NPkVill': 14, 'NridgHt': 15,
#          'NWAmes': 16,
#          'OldTown': 17, 'SWISU': 18, 'Sawyer': 19, 'SawyerW': 20, 'Somerst': 21, 'StoneBr': 22, 'Timber': 23,
#          'Veenker': 24})
#     trd['Condition1'] = trd['Condition1'].map(
#         {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 'PosA': 6, 'RRNe': 7, 'RRAe': 8})
#     trd['Condition2'] = trd['Condition2'].map(
#         {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 'PosA': 6, 'RRNe': 7, 'RRAe': 8})
#     trd['RoofStyle'] = trd['RoofStyle'].map(
#         {'Flat': 0, 'Gable': 1, 'Gambrel': 2, 'Hip': 3, 'Mansard': 4, 'Shed': 5})
#     trd['RoofMatl'] = trd['RoofMatl'].map(
#         {'ClyTile': 0, 'CompShg': 1, 'Membran': 2, 'Metal': 3, 'Roll': 4, 'Tar&Grv': 5, 'WdShake': 6, 'WdShngl': 7})
#     trd['Exterior1st'] = trd['Exterior1st'].map(
#         {'AsbShng': 0, 'AsphShn': 1, 'BrkComm': 2, 'BrkFace': 3, 'CBlock': 4, 'CemntBd': 5, 'HdBoard': 6, 'ImStucc': 7,
#          'MetalSd': 8,
#          'Other': 9, 'Plywood': 10, 'PreCast': 11, 'Stone': 12, 'Stucco': 13, 'VinylSd': 14, 'Wd Sdng': 15,
#          'WdShing': 16})
#     trd['Exterior2nd'] = trd['Exterior2nd'].map(
#         {'AsbShng': 0, 'AsphShn': 1, 'Brk Cmn': 2, 'BrkFace': 3, 'CBlock': 4, 'CmentBd': 5, 'HdBoard': 6, 'ImStucc': 7,
#          'MetalSd': 8,
#          'Other': 9, 'Plywood': 10, 'PreCast': 11, 'Stone': 12, 'Stucco': 13, 'VinylSd': 14, 'Wd Sdng': 15,
#          'Wd Shng': 16})
#     #normalize numeric columns
#     def scale_minmax(col):
#         return (col - col.min()) / (col.max() - col.min())
#     scale_cols = [col for col in cols_numeric if col != 'SalePrice']
#     trd[scale_cols] = trd[scale_cols].apply(scale_minmax, axis=0)
#
#     #Box-Cox Transform Suitable Variables
#     cols_notransform = ['2ndFlrSF', '1stFlrFrac', '2ndFlrFrac', 'StorageAreaSF',
#                         'EnclosedPorch', 'LowQualFinSF', 'MasVnrArea',
#                         'MiscVal', 'ScreenPorch', 'OpenPorchSF', 'WoodDeckSF', 'SalePrice',
#                         'BsmtGLQSF', 'BsmtALQSF', 'BsmtBLQSF', 'BsmtRecSF', 'BsmtLwQSF', 'BsmtUnfSF',
#                         'BsmtGLQFrac', 'BsmtALQFrac', 'BsmtBLQFrac', 'BsmtRecFrac', 'BsmtLwQFrac', 'BsmtUnfFrac']
#     col_nunique = dict()
#     for col in cols_numeric:
#         col_nunique[col] = trd[col].nunique()
#
#     col_nunique = pd.Series(col_nunique)
#     # cols_discrete = col_nunique[col_nunique < 13].index.tolist()
#     cols_continuous = col_nunique[col_nunique >= 13].index.tolist()
#
#     cols_transform = [col for col in cols_continuous if col not in cols_notransform]
#     from scipy import stats
#     for col in cols_transform:
#         # transform column
#         trd.loc[:, col], _ = stats.boxcox(trd.loc[:, col] + 1)
#
#         # renormalise column
#         trd.loc[:, col] = scale_minmax(trd.loc[:, col])
#     return trd