import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# 讀取data
train_data = pd.read_csv('D:/kaggle/House Price/train.csv')
test_data = pd.read_csv('D:/kaggle/House Price/test.csv')

# 尋找缺失值
def show_missing():
    missing = train_data.columns[train_data.isnull().any()].tolist()
    return missing
# --Missing data counts and percentage
# print('Missing Data Count')
# print(test_data[show_missing()].isnull().sum().sort_values(ascending = False))

# 資料前處理
def training_data(trd):
    # --Remove the feature which include most na data
    trd = trd.drop(columns=['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence'])
    # print(train_data.shape)
    # --Processing missing data at feature 'MasVnrArea'
    # plt.hist(train_data['MasVnrArea'])
    # plt.show()
    trd['MasVnrArea'] = trd['MasVnrArea'].fillna(value = trd['MasVnrArea'].mean())

    # --Processing missing data at feature 'BsmtQual'
    trd['BsmtQual'] = trd['BsmtQual'].fillna(value = 'No')

    # --Processing missing data at feature 'BsmtCond'
    trd['BsmtCond'] = trd['BsmtCond'].fillna(value = 'No')

    # --Processing missing data at feature 'BsmtFinType1'
    trd['BsmtFinType1'] = trd['BsmtFinType1'].fillna(value = 'No')

    # --Processing missing data at feature 'BsmtExposure'
    trd['BsmtExposure'] = trd['BsmtExposure'].fillna(value = 'None')

    # --Processing missing data at feature 'BsmtFinType2'
    trd['BsmtFinType2'] = trd['BsmtFinType2'].fillna(value = 'No')

    # --Processing missing data at feature 'GarageYrBlt' 、 'GarageType' 、 'GarageFinish' 、 'GarageQual' 、 'GarageCond'
    trd['GarageYrBlt'] = trd['GarageYrBlt'].fillna(value=round(trd['GarageYrBlt'].mean(), 0))
    trd['GarageType'] = trd['GarageType'].fillna(value='Attchd')
    trd['GarageFinish'] = trd['GarageFinish'].fillna(value='None')
    trd['GarageQual'] = trd['GarageQual'].fillna(value='None')
    trd['GarageCond'] = trd['GarageCond'].fillna(value='None')

    # --Processing missing data at feature 'LotFrontage'
    trd['LotFrontage'] = trd['LotFrontage'].fillna(trd['LotFrontage'].median())


    # --Processing missing data at feature 'FireplaceQu'
    trd['FireplaceQu'] = trd['FireplaceQu'].fillna(value = 'No')


    # --Processing missing data at feature 'Electrical'
    trd['Electrical'] = trd['Electrical'].fillna('SBrkr')

    # --Processing missing data at feature 'MasVnrType'
    trd['MasVnrType'] = trd['MasVnrType'].fillna('None')
    # print(train_data['MasVnrType'].value_counts())

    # Encode ordinal data
    trd['LotShape'] = trd['LotShape'].map({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3})
    trd['LandContour'] = trd['LandContour'].map({'Low': 0, 'HLS': 1, 'Bnk': 2, 'Lvl': 3})
    trd['Utilities'] = trd['Utilities'].map({'NoSeWa': 0, 'NoSeWa': 1, 'AllPub': 2})
    trd['BldgType'] = trd['BldgType'].map({'Twnhs': 0, 'TwnhsE': 1, 'Duplex': 2, '2fmCon': 3, '1Fam': 4})
    trd['HouseStyle'] = trd['HouseStyle'].map(
        {'1Story': 0, '1.5Fin': 1, '1.5Unf': 2, '2Story': 3, '2.5Fin': 4, '2.5Unf': 5, 'SFoyer': 6, 'SLvl': 7})
    trd['BsmtFinType1'] = trd['BsmtFinType1'].map(
        {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
    trd['BsmtFinType2'] = trd['BsmtFinType2'].map(
        {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
    trd['LandSlope'] = trd['LandSlope'].map({'Gtl': 0, 'Mod': 1, 'Sev': 2})
    trd['Street'] = trd['Street'].map({'Grvl': 0, 'Pave': 1})
    trd['MasVnrType'] = trd['MasVnrType'].map(
        {'None': 0, 'BrkCmn': 1, 'BrkFace': 2, 'CBlock': 3, 'Stone': 4})
    trd['CentralAir'] = trd['CentralAir'].map({'N': 0, 'Y': 1})
    trd['GarageFinish'] = trd['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
    trd['PavedDrive'] = trd['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2})
    trd['BsmtExposure'] = trd['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
    trd['ExterQual'] = trd['ExterQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['ExterCond'] = trd['ExterCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['BsmtCond'] = trd['BsmtCond'].map({'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['BsmtQual'] = trd['BsmtQual'].map({'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['HeatingQC'] = trd['HeatingQC'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['KitchenQual'] = trd['KitchenQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['FireplaceQu'] = trd['FireplaceQu'].map({'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['GarageQual'] = trd['GarageQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['GarageCond'] = trd['GarageCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    # Encode Categorical Variables
    trd['Foundation'] = trd['Foundation'].map(
        {'BrkTil': 0, 'CBlock': 1, 'PConc': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5})
    trd['Heating'] = trd['Heating'].map(
        {'Floor': 0, 'GasA': 1, 'GasW': 2, 'Grav': 3, 'OthW': 4, 'Wall': 5})
    trd['Electrical'] = trd['Electrical'].map({'SBrkr': 0, 'FuseA': 1, 'FuseF': 2, 'FuseP': 3, 'Mix': 4})
    trd['Functional'] = trd['Functional'].map(
        {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7})
    trd['GarageType'] = trd['GarageType'].map(
        {'None': 0, 'Detchd': 1, 'CarPort': 2, 'BuiltIn': 3, 'Basment': 4, 'Attchd': 5, '2Types': 6})
    trd['SaleType'] = trd['SaleType'].map(
        {'Oth': 0, 'ConLD': 1, 'ConLI': 2, 'ConLw': 3, 'Con': 4, 'COD': 5, 'New': 6, 'VWD': 7, 'CWD': 8, 'WD': 9})
    trd['SaleCondition'] = trd['SaleCondition'].map(
        {'Partial': 0, 'Family': 1, 'Alloca': 2, 'AdjLand': 3, 'Abnorml': 4, 'Normal': 5})
    trd['MSZoning'] = trd['MSZoning'].map(
        {'A': 0, 'FV': 1, 'RL': 2, 'RP': 3, 'RM': 4, 'RH': 5, 'C (all)': 6, 'I': 7})
    trd['LotConfig'] = trd['LotConfig'].map({'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4})
    trd['Neighborhood'] = trd['Neighborhood'].map(
        {'Blmngtn': 0, 'Blueste': 1, 'BrDale': 2, 'BrkSide': 3, 'ClearCr': 4, 'CollgCr': 5, 'Crawfor': 6, 'Edwards': 7,
         'Gilbert': 8,
         'IDOTRR': 9, 'MeadowV': 10, 'Mitchel': 11, 'NAmes': 12, 'NoRidge': 13, 'NPkVill': 14, 'NridgHt': 15,
         'NWAmes': 16,
         'OldTown': 17, 'SWISU': 18, 'Sawyer': 19, 'SawyerW': 20, 'Somerst': 21, 'StoneBr': 22, 'Timber': 23,
         'Veenker': 24})
    trd['Condition1'] = trd['Condition1'].map(
        {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 'PosA': 6, 'RRNe': 7, 'RRAe': 8})
    trd['Condition2'] = trd['Condition2'].map(
        {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 'PosA': 6, 'RRNe': 7, 'RRAe': 8})
    trd['RoofStyle'] = trd['RoofStyle'].map(
        {'Flat': 0, 'Gable': 1, 'Gambrel': 2, 'Hip': 3, 'Mansard': 4, 'Shed': 5})
    trd['RoofMatl'] = trd['RoofMatl'].map(
        {'ClyTile': 0, 'CompShg': 1, 'Membran': 2, 'Metal': 3, 'Roll': 4, 'Tar&Grv': 5, 'WdShake': 6, 'WdShngl': 7})
    trd['Exterior1st'] = trd['Exterior1st'].map(
        {'AsbShng': 0, 'AsphShn': 1, 'BrkComm': 2, 'BrkFace': 3, 'CBlock': 4, 'CemntBd': 5, 'HdBoard': 6, 'ImStucc': 7,
         'MetalSd': 8,
         'Other': 9, 'Plywood': 10, 'PreCast': 11, 'Stone': 12, 'Stucco': 13, 'VinylSd': 14, 'Wd Sdng': 15,
         'WdShing': 16})
    trd['Exterior2nd'] = trd['Exterior2nd'].map(
        {'AsbShng': 0, 'AsphShn': 1, 'Brk Cmn': 2, 'BrkFace': 3, 'CBlock': 4, 'CmentBd': 5, 'HdBoard': 6, 'ImStucc': 7,
         'MetalSd': 8,
         'Other': 9, 'Plywood': 10, 'PreCast': 11, 'Stone': 12, 'Stucco': 13, 'VinylSd': 14, 'Wd Sdng': 15,
         'Wd Shng': 16})
    return trd
def testing_data(trd):
    # --Remove the feature which include most na data
    trd = trd.drop(columns=['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence'])
    # --Processing missing data at feature 'MasVnrArea'
    # plt.hist(train_data['MasVnrArea'])
    # plt.show()
    trd['MasVnrArea'] = trd['MasVnrArea'].fillna(value = trd['MasVnrArea'].mean())
    # --Processing missing data at feature 'BsmtQual'
    trd['BsmtQual'] = trd['BsmtQual'].fillna(value = 'No')
    # --Processing missing data at feature 'BsmtCond'
    trd['BsmtCond'] = trd['BsmtCond'].fillna(value = 'No')
    # --Processing missing data at feature 'BsmtFinType1'
    trd['BsmtFinType1'] = trd['BsmtFinType1'].fillna(value = 'No')
    # --Processing missing data at feature 'BsmtExposure'
    trd['BsmtExposure'] = trd['BsmtExposure'].fillna(value = 'None')
    # --Processing missing data at feature 'BsmtFinType2'
    trd['BsmtFinType2'] = trd['BsmtFinType2'].fillna(value = 'No')
    # --Processing missing data at feature 'GarageYrBlt' 、 'GarageType' 、 'GarageFinish' 、 'GarageQual' 、 'GarageCond'
    trd['GarageYrBlt'] = trd['GarageYrBlt'].fillna(value=round(trd['GarageYrBlt'].mean(), 0))
    trd['GarageType'] = trd['GarageType'].fillna(value='Attchd')
    trd['GarageFinish'] = trd['GarageFinish'].fillna(value='None')
    trd['GarageQual'] = trd['GarageQual'].fillna(value='None')
    trd['GarageCond'] = trd['GarageCond'].fillna(value='None')
    # --Processing missing data at feature 'LotFrontage'
    trd['LotFrontage'] = trd['LotFrontage'].fillna(trd['LotFrontage'].median())
    # --Processing missing data at feature 'FireplaceQu'
    trd['FireplaceQu'] = trd['FireplaceQu'].fillna(value = 'No')
    # --Processing missing data at feature 'MasVnrType'
    trd['MasVnrType'] = trd['MasVnrType'].fillna('None')
    # --Processing missing data at feature 'MSZoning'
    trd['MSZoning'] = trd['MSZoning'].fillna('RL')
    # --Processing missing data at feature 'BsmtFullBath'
    trd['BsmtFullBath'] = trd['BsmtFullBath'].fillna(0)
    # --Processing missing data at feature 'BsmtHalfBath'
    trd['BsmtHalfBath'] = trd['BsmtHalfBath'].fillna(0)
    # --Processing missing data at feature 'Utilities'
    trd['Utilities'] = trd['Utilities'].fillna('AllPub')
    # --Processing missing data at feature 'Functional'
    trd['Functional'] = trd['Functional'].fillna(random.choice(test_data['Functional']))
    # --Processing missing data at feature 'Exterior2nd'
    trd['Exterior2nd'] = trd['Exterior2nd'].fillna('Other')
    # --Processing missing data at feature 'Exterior1st'
    trd['Exterior1st'] = trd['Exterior1st'].fillna('Other')
    # --Processing missing data at feature 'SaleType'
    trd['SaleType'] = trd['SaleType'].fillna('Oth')
    # --Processing missing data at feature 'BsmtFinSF1'
    trd['BsmtFinSF1'] = trd['BsmtFinSF1'].fillna(0)
    # --Processing missing data at feature 'BsmtFinSF2'
    trd['BsmtFinSF2'] = trd['BsmtFinSF2'].fillna(0)
    # --Processing missing data at feature 'BsmtUnfSF'
    trd['BsmtUnfSF'] = trd['BsmtUnfSF'].fillna(0)
    # --Processing missing data at feature 'KitchenQual'
    trd['KitchenQual'] = trd['KitchenQual'].fillna('TA')
    # --Processing missing data at feature 'GarageCars'
    trd['GarageCars'] = trd['GarageCars'].fillna(0)
    # --Processing missing data at feature 'GarageArea'
    trd['GarageArea'] = trd['GarageArea'].fillna(0)
    # --Processing missing data at feature 'TotalBsmtSF'
    trd['TotalBsmtSF'] = trd['TotalBsmtSF'].fillna(0)

    # Encode ordinal data
    trd['LotShape'] = trd['LotShape'].map({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3})
    trd['LandContour'] = trd['LandContour'].map({'Low': 0, 'HLS': 1, 'Bnk': 2, 'Lvl': 3})
    trd['Utilities'] = trd['Utilities'].map({'NoSeWa': 0, 'NoSeWa': 1, 'AllPub': 2})
    trd['BldgType'] = trd['BldgType'].map({'Twnhs': 0, 'TwnhsE': 1, 'Duplex': 2, '2fmCon': 3, '1Fam': 4})
    trd['HouseStyle'] = trd['HouseStyle'].map(
        {'1Story': 0, '1.5Fin': 1, '1.5Unf': 2, '2Story': 3, '2.5Fin': 4, '2.5Unf': 5, 'SFoyer': 6, 'SLvl': 7})
    trd['BsmtFinType1'] = trd['BsmtFinType1'].map(
        {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
    trd['BsmtFinType2'] = trd['BsmtFinType2'].map(
        {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
    trd['LandSlope'] = trd['LandSlope'].map({'Gtl': 0, 'Mod': 1, 'Sev': 2})
    trd['Street'] = trd['Street'].map({'Grvl': 0, 'Pave': 1})
    trd['MasVnrType'] = trd['MasVnrType'].map(
        {'None': 0, 'BrkCmn': 1, 'BrkFace': 2, 'CBlock': 3, 'Stone': 4})
    trd['CentralAir'] = trd['CentralAir'].map({'N': 0, 'Y': 1})
    trd['GarageFinish'] = trd['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
    trd['PavedDrive'] = trd['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2})
    trd['BsmtExposure'] = trd['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
    trd['ExterQual'] = trd['ExterQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['ExterCond'] = trd['ExterCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['BsmtCond'] = trd['BsmtCond'].map({'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['BsmtQual'] = trd['BsmtQual'].map({'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['HeatingQC'] = trd['HeatingQC'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['KitchenQual'] = trd['KitchenQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['FireplaceQu'] = trd['FireplaceQu'].map({'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['GarageQual'] = trd['GarageQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    trd['GarageCond'] = trd['GarageCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    # Encode Categorical Variables
    trd['Foundation'] = trd['Foundation'].map(
        {'BrkTil': 0, 'CBlock': 1, 'PConc': 2, 'Slab': 3, 'Stone': 4, 'Wood': 5})
    trd['Heating'] = trd['Heating'].map(
        {'Floor': 0, 'GasA': 1, 'GasW': 2, 'Grav': 3, 'OthW': 4, 'Wall': 5})
    trd['Electrical'] = trd['Electrical'].map({'SBrkr': 0, 'FuseA': 1, 'FuseF': 2, 'FuseP': 3, 'Mix': 4})
    trd['Functional'] = trd['Functional'].map(
        {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7})
    trd['GarageType'] = trd['GarageType'].map(
        {'None': 0, 'Detchd': 1, 'CarPort': 2, 'BuiltIn': 3, 'Basment': 4, 'Attchd': 5, '2Types': 6})
    trd['SaleType'] = trd['SaleType'].map(
        {'Oth': 0, 'ConLD': 1, 'ConLI': 2, 'ConLw': 3, 'Con': 4, 'COD': 5, 'New': 6, 'VWD': 7, 'CWD': 8, 'WD': 9})
    trd['SaleCondition'] = trd['SaleCondition'].map(
        {'Partial': 0, 'Family': 1, 'Alloca': 2, 'AdjLand': 3, 'Abnorml': 4, 'Normal': 5})
    trd['MSZoning'] = trd['MSZoning'].map(
        {'A': 0, 'FV': 1, 'RL': 2, 'RP': 3, 'RM': 4, 'RH': 5, 'C (all)': 6, 'I': 7})
    trd['LotConfig'] = trd['LotConfig'].map({'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4})
    trd['Neighborhood'] = trd['Neighborhood'].map(
        {'Blmngtn': 0, 'Blueste': 1, 'BrDale': 2, 'BrkSide': 3, 'ClearCr': 4, 'CollgCr': 5, 'Crawfor': 6, 'Edwards': 7,
         'Gilbert': 8,
         'IDOTRR': 9, 'MeadowV': 10, 'Mitchel': 11, 'NAmes': 12, 'NoRidge': 13, 'NPkVill': 14, 'NridgHt': 15,
         'NWAmes': 16,
         'OldTown': 17, 'SWISU': 18, 'Sawyer': 19, 'SawyerW': 20, 'Somerst': 21, 'StoneBr': 22, 'Timber': 23,
         'Veenker': 24})
    trd['Condition1'] = trd['Condition1'].map(
        {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 'PosA': 6, 'RRNe': 7, 'RRAe': 8})
    trd['Condition2'] = trd['Condition2'].map(
        {'Artery': 0, 'Feedr': 1, 'Norm': 2, 'RRNn': 3, 'RRAn': 4, 'PosN': 5, 'PosA': 6, 'RRNe': 7, 'RRAe': 8})
    trd['RoofStyle'] = trd['RoofStyle'].map(
        {'Flat': 0, 'Gable': 1, 'Gambrel': 2, 'Hip': 3, 'Mansard': 4, 'Shed': 5})
    trd['RoofMatl'] = trd['RoofMatl'].map(
        {'ClyTile': 0, 'CompShg': 1, 'Membran': 2, 'Metal': 3, 'Roll': 4, 'Tar&Grv': 5, 'WdShake': 6, 'WdShngl': 7})
    trd['Exterior1st'] = trd['Exterior1st'].map(
        {'AsbShng': 0, 'AsphShn': 1, 'BrkComm': 2, 'BrkFace': 3, 'CBlock': 4, 'CemntBd': 5, 'HdBoard': 6, 'ImStucc': 7,
         'MetalSd': 8,
         'Other': 9, 'Plywood': 10, 'PreCast': 11, 'Stone': 12, 'Stucco': 13, 'VinylSd': 14, 'Wd Sdng': 15,
         'WdShing': 16})
    trd['Exterior2nd'] = trd['Exterior2nd'].map(
        {'AsbShng': 0, 'AsphShn': 1, 'Brk Cmn': 2, 'BrkFace': 3, 'CBlock': 4, 'CmentBd': 5, 'HdBoard': 6, 'ImStucc': 7,
         'MetalSd': 8,
         'Other': 9, 'Plywood': 10, 'PreCast': 11, 'Stone': 12, 'Stucco': 13, 'VinylSd': 14, 'Wd Sdng': 15,
         'Wd Shng': 16})
    return trd
train_data = training_data(train_data)
test_data = testing_data(test_data)
x = train_data[train_data.select_dtypes(exclude=['object']).columns]
x = x.drop(columns=['SalePrice'])
y = train_data['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20)

# Regression_instantiate
# linreg = LinearRegression()
# linreg.fit(x_train, y_train)
# confidence = linreg.score(x, y)
# y_pred = linreg.predict(test_data)
# y_pred = pd.DataFrame(y_pred)
# y_pred.to_csv("D:/kaggle/House Price/regression_predict.csv")
# print(confidence)


#RandomForest
from sklearn.ensemble import RandomForestClassifier
ran_forest = RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state=3, n_jobs=2)
ran_forest.fit(x_train, y_train)
test_y_predicted = ran_forest.predict(test_data)

#Select_fit feature
feat_labels = x.columns[1:]
importances = ran_forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))



# New random forest with only the elevent most important variables
X_best = train_data[['BsmtFullBath', 'GarageQual', 'Street', 'Heating', 'YearRemodAdd', '2ndFlrSF',
                     'TotalBsmtSF', 'GarageFinish', 'BsmtFinType2', 'RoofStyle', 'LotArea', 'OverallCond']]
y = train_data['SalePrice']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_best,y, test_size = .20,random_state = 101)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
rforest = RandomForestRegressor(n_estimators=1000, random_state=3, n_jobs=2)
rforest.fit(X_best, y)
y_pred_rforest = rforest.predict(X_test)

#input test data
test_best_feature = test_data[['BsmtFullBath', 'GarageQual', 'Street', 'Heating', 'YearRemodAdd', '2ndFlrSF',
                     'TotalBsmtSF', 'GarageFinish', 'BsmtFinType2', 'RoofStyle', 'LotArea', 'OverallCond']]
y_pred_rforest = rforest.predict(test_best_feature)
y_pred_rforest = pd.DataFrame(y_pred_rforest)
y_pred_rforest.to_csv("D:/kaggle/House Price/randomforest_predict.csv")





# Use best fit feature at Regression
X_best = train_data[['BsmtFullBath', 'GarageQual', 'Street', 'Heating', 'YearRemodAdd', '2ndFlrSF',
                     'TotalBsmtSF', 'GarageFinish', 'BsmtFinType2', 'RoofStyle', 'LotArea', 'OverallCond']]
y = train_data['SalePrice']

test_best_feature = test_data[['BsmtFullBath', 'GarageQual', 'Street', 'Heating', 'YearRemodAdd', '2ndFlrSF',
                     'TotalBsmtSF', 'GarageFinish', 'BsmtFinType2', 'RoofStyle', 'LotArea', 'OverallCond']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size = .20,random_state = 101)
import pandas as pd
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(test_best_feature)
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv("D:/kaggle/House Price/regression_predict.csv")




























