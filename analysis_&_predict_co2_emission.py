# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""# **DATA PROCESSING**"""

data = pd.read_csv("https://raw.githubusercontent.com/KingBambara/CO2_Emission_CAR/main/my1995-2024-fuel-consumption-ratings.csv")
data_desc = pd.read_csv("https://raw.githubusercontent.com/KingBambara/CO2_Emission_CAR/main/Data%20Description.csv")

data.describe()

data.info()

data['Vehicle class'].unique()

data.head()

data.isnull().sum()

data_desc

"""# **DATA VIZUALISATION & ANALYSIS**


"""

# Heatmap pour voir les corrélations entre les valeurs numériques des colonnes
numeric_columns = data.select_dtypes(exclude=['object'])
correlation_matrix = numeric_columns.corr()
plt.figure(figsize = (12, 12))
sns.heatmap(correlation_matrix, cmap = 'viridis', annot = True)
plt.title('Correlation Heatmap')
plt.show()

#plt.figure(figsize=(18,6))
#sns.pairplot(data, hue='CO2 emissions (g/km)')

# box plot pour les émissions de CO2
sns.boxplot(x = data['CO2 emissions (g/km)'])
plt.show()

#histplot pour la distribution normale des valeurs de CO2
sns.histplot(x = data['CO2 emissions (g/km)'])
plt.title('Répartition des émissions de CO2 (g/km)')
plt.show()

sns.histplot(data, x = "Combined (L/100 km)", hue = "Fuel type", element = "poly")
plt.title('Types de carburant et leur consommation')
plt.show()

plt.figure(figsize = (10, 6))
sns.boxplot(x = 'Fuel type', y = 'Combined (L/100 km)', data = data)
plt.title('Consommation de carburant par type de carburant')
plt.xlabel('Type de carburant')
plt.ylabel('Consommation de carburant')
plt.show()

plt.figure(figsize = (12, 6))
mean_emissions_by_make = data.groupby('Make')['CO2 emissions (g/km)'].mean().sort_values(ascending=False)
plt.bar(mean_emissions_by_make.index, mean_emissions_by_make)
plt.title("Émissions moyennes de CO2 par marque de voiture")
plt.xlabel("marque de voiture")
plt.ylabel("Émissions moyennes de CO2")
plt.xticks(rotation = 90)
plt.show()

emissions_make = data['Make'].value_counts()


plt.figure(figsize=(12, 6))
plt.bar(emissions_make.index, emissions_make, color = 'skyblue')

plt.title('nombre de véhicule par compagnie')
plt.xlabel('companies')
plt.ylabel('nombre de véhicle')
plt.xticks(rotation = 90)
plt.show()

"""# **DATA CLEANING**"""

col = ['Make', 'Engine size (L)', 'Cylinders', 'Fuel type', 'Combined (L/100 km)', 'CO2 emissions (g/km)' ]

data_col = data[col]

data_col.head()

numeric_columns = data_col.select_dtypes(exclude=['object'])
correlation_matrix = numeric_columns.corr()
plt.figure(figsize = (10, 10))
sns.heatmap(correlation_matrix, cmap = 'crest', annot = True, fmt=".1f")
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize = (10, 6))
mean_emissions_by_make = data_col.groupby('Make')['CO2 emissions (g/km)'].mean().sort_values(ascending=False)
plt.bar(mean_emissions_by_make.index, mean_emissions_by_make)
plt.title("Émissions moyennes de CO2 par marque de voiture")
plt.xlabel("marque de voiture")
plt.ylabel("Émissions moyennes de CO2")
plt.xticks(rotation = 90)
plt.show()

#removal of natural gas lines
idex = data_col[data_col['Fuel type'] == 'N' ].index
data_col = data_col.drop(idex)

"""### Transform column Fuel type into characteristic"""

dums = pd.get_dummies(data_col['Fuel type'],prefix="Fuel_Type", dtype= int)
frames = [data_col, dums]
result = pd.concat(frames,axis=1)
result.rename(columns={'Fuel_Type_D': 'diesel', 'Fuel_Type_E': 'ethanol (E85)',
                       'Fuel_Type_X': 'regular gasoline', 'Fuel_Type_Z': 'premium gasoline'}, inplace = True)
result.head()

#removal of Fuel type and Make
result.drop(['Fuel type'],inplace=True,axis=1)
result.drop(['Make'],inplace=True,axis=1)
result.head()

X = result.drop(['CO2 emissions (g/km)'], axis= 1)
X = X.values
y = result["CO2 emissions (g/km)"].values

X.shape

"""### Splitting the dataset into the training set and test set

---
80% for training set

20% for test set





"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("X_train", X_train.shape)

print("y_train",y_train.shape)

print("X_test",X_test.shape)

print("y_test",y_test.shape)

#training set and test set for Decision Tree Regression & Random Forest Regression
X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.2, random_state=0)

X_train

"""### **Feature Scaling**"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, :3] = sc.fit_transform(X_train[:, :3])
X_test[:, :3] = sc.transform(X_test[:, :3])

X_train

X_train_

"""# **Prediction**

### **Linear Regression**
"""

from sklearn.metrics import accuracy_score, mean_absolute_error as mae, r2_score, mean_squared_error as mse

np.random.seed(0)

from sklearn.linear_model import LinearRegression
model_lr= LinearRegression()
model_lr.fit(X_train, y_train)

y_pred= model_lr.predict(X_test)

model_lr_score = model_lr.score(X_test, y_test)
MSE = round(np.sqrt(mse(y_test,y_pred)),4)

lr_R2 = r2_score(y_test,y_pred)
lr_RMSE = np.sqrt(MSE)
lr_mae = mae(y_test, y_pred)

print('test Score : ', model_lr_score)
print('Mean Squared Error (MSE): ',MSE)
print('R- Squared (R-square): ',lr_R2)
print('Root Mean Squared Error (RMSE): ',lr_RMSE)
print('Mean Absolute error (MAE): ',lr_mae)

"""### **Ridge Regression**"""

from sklearn.linear_model import Ridge

model_ridge_reg = Ridge(random_state = 0)
model_ridge_reg.fit(X_train,y_train)
y_pred= model_ridge_reg.predict(X_test)

model_ridge_reg_score = model_ridge_reg.score(X_test, y_test)
MSE = round(np.sqrt(mse(y_test,y_pred)),4)
ridge_reg_R2 = r2_score(y_test,y_pred)
ridge_reg_RMSE = np.sqrt(MSE)
ridge_reg_mae = mae(y_test,y_pred)
print('test Score : ', model_ridge_reg_score)
print('Mean Squared Error (MSE): ',MSE)
print('R- Squared (R-square): ',ridge_reg_R2)
print('Root Mean Squared Error (RMSE): ',ridge_reg_RMSE)
print('Mean Absolute error (MAE): ', ridge_reg_mae)

"""### **SVR**"""

from sklearn.svm import SVR
np.random.seed(0)

model_svr = SVR(kernel = 'linear', C=1.0, epsilon=0.2)
model_svr.fit(X_train,y_train)
y_pred= model_svr.predict(X_test)

model_svr_score = model_svr.score(X_test, y_test)
MSE = round(np.sqrt(mse(y_test,y_pred)),4)
svr_R2 = r2_score(y_test,y_pred)
svr_RMSE = np.sqrt(MSE)
svr_mae = mae(y_test,y_pred)
print('test Score : ', model_svr_score)
print('Mean Squared Error (MSE): ',MSE)
print('R- Squared (R-square): ',svr_R2)
print('Root Mean Squared Error (RMSE): ',svr_RMSE)
print('Mean Absolute error (MAE): ', svr_mae)

"""### **Lasso Regression**"""

from sklearn.linear_model import Lasso

model_la_reg = Lasso(random_state = 0)
model_la_reg.fit(X_train,y_train)
y_pred= model_la_reg.predict(X_test)

model_la_reg_score = model_la_reg.score(X_test, y_test)
MSE = round(np.sqrt(mse(y_test,y_pred)),4)
la_reg_R2 = r2_score(y_test,y_pred)
la_reg_RMSE = np.sqrt(MSE)
la_reg_mae = mae(y_test,y_pred)
print('test Score : ', model_la_reg_score)
print('Mean Squared Error (MSE): ',MSE)
print('R- Squared (R-square): ',la_reg_R2)
print('Root Mean Squared Error (RMSE): ',la_reg_RMSE)
print('Mean Absolute error (MAE): ', la_reg_mae)

"""
### **Decision Tree Regression**"""

from sklearn.tree import DecisionTreeRegressor

model_dt_reg = DecisionTreeRegressor(criterion = 'squared_error', splitter = 'best', random_state = 0, max_depth = 12)
model_dt_reg.fit(X_train_,y_train_)
y_pred_dtr= model_dt_reg.predict(X_test_)

model_dt_reg_score = model_dt_reg.score(X_test_, y_test_)
MSE = round(np.sqrt(mse(y_test_,y_pred_dtr)),4)
dt_reg_R2 = r2_score(y_test_,y_pred_dtr)
dt_reg_RMSE = np.sqrt(MSE)
dt_reg_mae = mae(y_test_, y_pred_dtr)
print('test Score : ', model_dt_reg_score)
print('Mean Squared Error (MSE): ',MSE)
print('R- Squared (R-square): ',dt_reg_R2)
print('Root Mean Squared Error (RMSE): ',dt_reg_RMSE)
print('Mean Absolute error (MAE): ',dt_reg_mae)

from sklearn import tree
plt.figure(figsize=(20, 15))
tree.plot_tree(model_dt_reg, rounded = True, filled=True)
plt.show()

"""### **Random Forest Regression**"""

from sklearn.ensemble import RandomForestRegressor

model_rf_reg = RandomForestRegressor(n_estimators = 17, criterion = 'poisson', max_depth = 9, random_state = 0)
model_rf_reg.fit(X_train_, y_train_)
y_pred_rf= model_rf_reg.predict(X_test_)
model_rf_reg_score = model_rf_reg.score(X_test_, y_test_)
MSE = round(np.sqrt(mse(y_test_,y_pred_rf)),4)
rf_reg_R2 = r2_score(y_test_,y_pred)
rf_reg_RMSE = np.sqrt(MSE)
rf_reg_mae = mae(y_test_,y_pred_rf)
print('test Score : ', model_rf_reg_score)
print('Mean Squared Error (MSE): ',MSE)
print('R- Squared (R-square): ',rf_reg_R2)
print('Root Mean Squared Error (RMSE): ',rf_reg_RMSE)
print('Mean Absolute error (MAE): ', rf_reg_mae)

"""### **XGBoost**"""

from xgboost import XGBRegressor

XGB = XGBRegressor()
XGB.fit(X_train_,y_train_)
y_pred_XGB= XGB.predict(X_test_)

XGB_score = XGB.score(X_test_, y_test_)
MSE = round(np.sqrt(mse(y_test_,y_pred_XGB)),4)
XGB_R2 = r2_score(y_test_,y_pred_XGB)
XGB_RMSE = np.sqrt(MSE)
XGB_mae = mae(y_test_,y_pred_XGB)
print('train Score : ', XGB.score(X_train_, y_train_))
print('test Score : ', XGB_score)
print('Mean Squared Error (MSE): ',MSE)
print('R- Squared (R-square): ',XGB_R2)
print('Root Mean Squared Error (RMSE): ',XGB_RMSE)
print('Mean Absolute error (MAE): ', XGB_mae)

"""### **Best algorithm**"""

best_score_rmse = pd.DataFrame({
    'Name' : ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor',
              'Ridge Regression', 'SVR', 'Lasso Regression', 'XGBoost'],
    'R2 Score' : [lr_R2, dt_reg_R2, rf_reg_R2, ridge_reg_R2, svr_R2, la_reg_R2, XGB_R2],
    'RMSE' : [lr_RMSE, dt_reg_RMSE, rf_reg_RMSE, ridge_reg_RMSE, svr_RMSE, la_reg_RMSE, XGB_RMSE],
    'MAE' : [lr_mae, dt_reg_mae, rf_reg_mae, ridge_reg_mae, svr_mae, la_reg_mae, XGB_mae]
})
best_score_rmse

plt.title('R2 score')
plt.bar(best_score_rmse['Name'], best_score_rmse['R2 Score'],
        color = ['red','green','blue','magenta', 'black', 'cyan','purple'])
plt.xlabel('Model')
plt.ylabel('R2 score')
plt.ylim(0, 1)
plt.xticks(rotation = 90)

plt.title('rmse')
plt.bar(best_score_rmse['Name'], best_score_rmse['RMSE'],
        color = ['red','green','blue','magenta', 'black', 'cyan','purple'])
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.xticks(rotation = 90)

"""### **comparison between predict and actual values**"""

frames = [y_pred_XGB, y_test]
result_pred = pd.DataFrame(data=frames)
result_pred=result_pred.T

result_pred_rf=result_pred.rename(columns={0:'Predict',1:'Real'})
result_pred_rf["Predict"]=result_pred_rf["Predict"].map(lambda x:round(x,2))
result_pred_rf["Diff"]=abs(result_pred_rf["Predict"]-result_pred_rf["Real"])
result_pred_rf["Diff"]=result_pred_rf["Diff"]
print("Mean Diff: ",abs(result_pred_rf["Diff"]).mean())
result_pred_rf.head(20)

sns.scatterplot(x = 'Predict', y = 'Real', data=result_pred_rf)

