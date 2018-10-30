import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_data = pd.read_csv('D:/kaggle/House Price/train.csv')
def show_missing():
    missing = train_data.columns[train_data.isnull().any()].tolist()
    return missing

# --Missing data counts and percentage
# print('Missing Data Count')
# print(train_data[show_missing()].isnull().sum().sort_values(ascending = False))
# print('--'*40)
# print('Missing Data Percentage')
# print(round(train_data[show_missing()].isnull().sum().sort_values(ascending = False)/len(train_data)*100,2))


# --Remove the feature which include most na data
train_data = train_data.drop(columns=['PoolQC', 'MiscFeature','Alley','Fence'])
# print(train_data.shape)


# --Processing missing data at feature 'MasVnrArea'
plt.hist(train_data['MasVnrArea'])
plt.show()
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(value = train_data['MasVnrArea'].mean())

# --Processing missing data at feature 'BsmtQual'
train_data['BsmtQual'] = train_data['BsmtQual'].fillna(value = 'No')

# --Processing missing data at feature 'BsmtCond'
train_data['BsmtCond'] = train_data['BsmtCond'].fillna(value = 'No')

# --Processing missing data at feature 'BsmtFinType1'
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].fillna(value = 'No')

# --Processing missing data at feature 'BsmtExposure'
train_data['BsmtExposure'] = train_data['BsmtExposure'].fillna(value = 0)

# --Processing missing data at feature 'BsmtFinType2'
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].fillna(value = 'No')

# --Processing missing data at feature 'GarageYrBlt' 、 'GarageType' 、 'GarageFinish' 、 'GarageQual' 、 'GarageCond'
garage_null = ['GarageYrBlt' , 'GarageType' , 'GarageFinish' , 'GarageQual' , 'GarageCond']
def feat_impute(column, value):
    train_data.loc[train_data[column].isnull()] = value


for cols in garage_null:
   if train_data[cols].dtype == np.object:
         feat_impute(cols, 'None')
   else:
         feat_impute(cols, 0)

# print(train_data['GarageYrBlt'].isnull().any())


# --Processing missing data at feature 'LotFrontage'
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())


# --Processing missing data at feature 'FireplaceQu'
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna(value = 'No')



































































