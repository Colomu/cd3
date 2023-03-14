import os
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import json
import pycountry
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib
from math import sqrt

# To change scientific numbers to float
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Increases the size of sns plots
sns.set(rc={'figure.figsize':(5,5)})

# view all the dataframe
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

# remove warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



#1.LOADING RAW DATA

# Loading the data
raw_data = pd.read_csv('C:/Users/olomuc/OneDrive - MHRA/COVID_ML_MODELS/covid_fat_deaths.csv')

# print the shape
print(raw_data.shape)

for column in raw_data:
    unique_vals = np.unique(raw_data[column])
    nr_values = len(unique_vals)
    if nr_values < 10:
        print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
    else:
        print('The number of values for feature {} :{}'.format(column, nr_values))


#3. check for null values

raw_data.isnull().sum()

# manually creating null values
raw_data['Pulses 2'] = raw_data['Pulses']
raw_data.loc[0, 'Pulses 2'] = np.nan
raw_data.loc[1, 'Pulses 2'] = np.nan
raw_data.loc[2, 'Pulses 2'] = np.nan

# drop null values
raw_data['Pulses 2'][raw_data['Pulses 2'].isna()] = raw_data['Pulses 2'].mean()

# dropping the column
del raw_data['Pulses 2']

# drop rows with NaN values in the 'Deaths' column
raw_data.dropna(subset=['Deaths'], inplace=True)

# select only the required features
X = raw_data[['Milk - Excluding Butter', 'Obesity', 'Animal products total']].values
X_columns = ['Milk - Excluding Butter', 'Obesity', 'Animal products total']
y = raw_data['Animal fats'].astype(int)


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size = 0.2, random_state=15)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Replace NaN values with the mean of the corresponding feature
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Training the Regression

lm = LinearRegression(fit_intercept = True)
lm.fit(X_train, y_train)

y_pred = lm.predict(X_train)

# Normalizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Performing Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Printing the coefficients and intercept
print('Coefficients:', ridge.coef_)
print('Intercept:', ridge.intercept_)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Lasso regularization object with alpha value
lasso = Lasso(alpha=0.01)

# Fit the model to the training data
lasso.fit(X_train_scaled, y_train)

# Evaluate the model on the training and testing data
print('Training R^2:', lasso.score(X_train_scaled, y_train))
print('Testing R^2:', lasso.score(X_test_scaled, y_test))
print('Training RMSE:', mean_squared_error(y_train, lasso.predict(X_train_scaled), squared=False))
print('Testing RMSE:', mean_squared_error(y_test, lasso.predict(X_test_scaled), squared=False))
print('Training MAE:', mean_absolute_error(y_train, lasso.predict(X_train_scaled)))
print('Testing MAE:', mean_absolute_error(y_test, lasso.predict(X_test_scaled)))

# Print the coefficients and intercept
print('Coefficients:', lasso.coef_)
print('Intercept:', lasso.intercept_)

# Create decision tree regressor object
dt = DecisionTreeRegressor()

# Fit the model to the training data
dt.fit(X_train, y_train)

# Evaluate the model on the training and testing data
print('Training R^2:', dt.score(X_train, y_train))
print('Testing R^2:', dt.score(X_test, y_test))
print('Training RMSE:', mean_squared_error(y_train, dt.predict(X_train), squared=False))
print('Testing RMSE:', mean_squared_error(y_test, dt.predict(X_test), squared=False))
print('Training MAE:', mean_absolute_error(y_train, dt.predict(X_train)))
print('Testing MAE:', mean_absolute_error(y_test, dt.predict(X_test)))

# Create random forest regressor object
rf = RandomForestRegressor()

# Fit the model to the training data
rf.fit(X_train, y_train)

# Evaluate the model on the training and testing data
print('Training R^2:', rf.score(X_train, y_train))
print('Testing R^2:', rf.score(X_test, y_test))
print('Training RMSE:', mean_squared_error(y_train, rf.predict(X_train), squared=False))
print('Testing RMSE:', mean_squared_error(y_test, rf.predict(X_test), squared=False))
print('Training MAE:', mean_absolute_error(y_train, rf.predict(X_train)))
print('Testing MAE:', mean_absolute_error(y_test, rf.predict(X_test)))

# Create gradient boosting regressor object
gb = GradientBoostingRegressor()

# Fit the model to the training data
gb.fit(X_train, y_train)

# Evaluate the model on the training and testing data
print('Training R^2:', gb.score(X_train, y_train))
print('Testing R^2:', gb.score(X_test, y_test))
print('Training RMSE:', mean_squared_error(y_train, gb.predict(X_train), squared=False))
print('Testing RMSE:', mean_squared_error(y_test, gb.predict(X_test), squared=False))
print('Training MAE:', mean_absolute_error(y_train, gb.predict(X_train)))
print('Testing MAE:', mean_absolute_error(y_test, gb.predict(X_test)))

# Create XGBoost regressor object
xgb = XGBRegressor()

# Fit the model to the training data
xgb.fit(X_train, y_train)

# Evaluate the model on the training and testing data
print('Training R^2:', xgb.score(X_train, y_train))
print('Testing R^2:', xgb.score(X_test, y_test))
print('Training RMSE:', mean_squared_error(y_train, xgb.predict(X_train), squared=False))
print('Testing RMSE:', mean_squared_error(y_test, xgb.predict(X_test), squared=False))
print('Training MAE:', mean_absolute_error(y_train, gb.predict(X_train)))
print('Testing MAE:', mean_absolute_error(y_test, gb.predict(X_test)))

# Print the coefficients and intercept for decision tree regression
print('Decision Tree Coefficients:', dt.feature_importances_)

# Print the coefficients and intercept for random forest regression
print('Random Forest Coefficients:', rf.feature_importances_)

# Print the coefficients and intercept for gradient boosting regression
print('Gradient Boosting Coefficients:', gb.feature_importances_)

# Print the coefficients and intercept for XGBoost regression
print('XGBoost Coefficients:', xgb.feature_importances_)

# Saving the best model using joblib
filename = 'best_model.joblib'
joblib.dump(rf, filename)

# Loading the best model
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)
