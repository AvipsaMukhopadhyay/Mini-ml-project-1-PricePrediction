import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
housing = pd.read_csv("Data1.csv")


def check_non_numeric(data, columns):
    non_numeric_values = {}
    for column in columns:
        non_numeric = data[pd.to_numeric(data[column], errors='coerce').isna()][column].unique()
        if len(non_numeric) > 0:
            non_numeric_values[column] = non_numeric
    return non_numeric_values

cols= ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude','Y house price of unit area']

non_numeric_values = check_non_numeric(housing, cols)
print("Non-numeric values found:", non_numeric_values)

convhous = housing.copy()
convhous[cols] = convhous[cols].apply(pd.to_numeric, errors='coerce')
convhous.dropna(subset=cols, inplace=True)

# print(housing.head())
# print(housing.info())
# print(housing["X5 latitude"].value_counts)
# print(housing.describe())
# fk = ["X1 transaction date","X2 house age","X3 distance to the nearest MRT station","X4 number of convenience stores","X5 latitude","X5 latitude"]
# for i in fk:
#     plt.scatter(data=housing,x=i,y="Y house price of unit area")
#     plt.show()

# def train_test_split(Data,test_ratio):
#     np.random.seed(42)
#     sh = np.random.permutation(len(Data))
#     test_size=int(len(Data)*test_ratio)
#     test = sh[:test_size]
#     train = sh[test_size:]
#     return Data.iloc[test],Data.iloc[train]
# testset,trainset = train_test_split(housing,0.2)
# print(len(trainset),len(testset))

train_set, test_set = train_test_split(convhous,test_size=0.2,random_state=42)
print(len(train_set),len(test_set))

#for stratified splitting
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(housing, housing['X5 latitude']):
#     strat_train_set = housing.loc[train_index]
#     strat_test_set = housing.loc[test_index]
# print(strat_train_set)

housing['X2 house age'] = pd.to_numeric(housing['X2 house age'], errors='coerce')
housing['X4 number of convenience stores'] = pd.to_numeric(housing['X4 number of convenience stores'], errors='coerce')
housing['X1 transaction date'] = pd.to_numeric(housing['X1 transaction date'], errors='coerce')
housing['X3 distance to the nearest MRT station'] = pd.to_numeric(housing['X3 distance to the nearest MRT station'], errors='coerce')
housing['X5 latitude'] = pd.to_numeric(housing['X5 latitude'], errors='coerce')
housing['X6 longitude'] = pd.to_numeric(housing['X6 longitude'], errors='coerce')
housing['Y house price of unit area'] = pd.to_numeric(housing['Y house price of unit area'], errors='coerce')


housing['newc'] = housing['X2 house age'] * housing['X4 number of convenience stores']
print(housing['newc'].head())
print(housing.head())

# corr = housing.corr()
# print(corr['X2 house age'].sort_values(ascending=False))

# def check_non_numeric(data, columns):
#     for column in columns:
#         non_numeric = data[pd.to_numeric(data[column], errors='coerce').isna()][column].unique()
#         if len(non_numeric) > 0:
#             print(f"Non-numeric values in column '{column}': {non_numeric}")

# numericc = [
#     'X1 transaction date', 
#     'X2 house age', 
#     'X3 distance to the nearest MRT station', 
#     'X4 number of convenience stores', 
#     'X5 latitude', 
#     'X6 longitude', 
#     'Y house price of unit area'
# ]
# check_non_numeric(housing, numericc)

# for column in numericc:
#     housing[column] = pd.to_numeric(housing[column], errors='coerce')

# check_non_numeric(housing,numericc)

housing['checkc'] = housing['X5 latitude']*housing['Y house price of unit area']
print(housing['checkc'].head)

hous = train_set.drop("Y house price of unit area",axis = 1)
houslab = train_set['Y house price of unit area'].copy()

# print(hous)
# print(houslab)


imputer = SimpleImputer(strategy="median")
imputer.fit(convhous)

print(imputer.statistics_)
X = imputer.transform(convhous)
houstr = pd.DataFrame(X, columns=convhous.columns)
print(houstr.describe())

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

housingtr = my_pipeline.fit_transform(hous)

# print(housingtr.shape)




# model = LinearRegression()
# model.fit(hous,houslab)

# # print("Intercept:", model.intercept_)
# # print("Coefficients:", model.coef_)

# sda = hous.iloc[:5]
# sla = houslab.iloc[:5]
# pdsa = my_pipeline.transform(sda)
# print(model.predict(pdsa))


# housepr = model.predict(hous)
# mmse = mean_squared_error(houslab,housepr)
# rrmse = np.sqrt(mmse)
# print(mmse)
# print(rrmse)


# mod = DecisionTreeRegressor()
# mod.fit(hous,houslab)

# # print("Intercept:", model.intercept_)
# # print("Coefficients:", model.coef_)

# sd = hous.iloc[:5]
# sl = houslab.iloc[:5]
# pds = my_pipeline.transform(sd)
# print(mod.predict(pds))



# houspr = mod.predict(hous)
# mse = mean_squared_error(houslab,houspr)
# rmse = np.sqrt(mse)
# print(mse)
# print(rmse)

modu = RandomForestRegressor()
modu.fit(hous,houslab)

sdc = hous.iloc[:5]
slc = houslab.iloc[:5]
pdsc = my_pipeline.transform(sdc)
print(modu.predict(pdsc))

housprs = modu.predict(hous)
cmse = mean_squared_error(houslab,housprs)
crmse = np.sqrt(cmse)
print(cmse)
print(crmse)


# sc = cross_val_score(mod, hous, houslab, scoring="neg_mean_squared_error", cv=10)
# sca= cross_val_score(model, hous, houslab, scoring="neg_mean_squared_error",cv=10)
scs= cross_val_score(modu, hous, houslab, scoring="neg_mean_squared_error",cv=10)
# rsc = np.sqrt(-sc)
# rsca = np.sqrt(-sca)
rscs = np.sqrt(-scs)
# print(rsc)
# print(rsca)
print(rscs)

def printsc(scs):
    print("scores",scs)
    print("mean",scs.mean())
    print("standard deviation",scs.std())
printsc(rscs)

testf = test_set.drop("Y house price of unit area", axis=1)
testlab = test_set['Y house price of unit area'].copy()
testfp = my_pipeline.transform(testf)

fpred = modu.predict(testf)
fmse = mean_squared_error(testlab,fpred)
frmse = np.sqrt(fmse)

print(frmse)
