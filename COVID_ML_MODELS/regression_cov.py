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

"""
    This code trains and evaluates several regression models on a COVID-19 dataset, using features such as animal fats, fruits, and obesity to predict the number of recoveries in different countries. It performs the following steps:

    1. Sets options for printing and warning suppression.
    2. Loads the raw data from a CSV file and prints out the number of values for each feature.
    3. Checks for null values in the data and replaces any null values with the mean of the non-null values. Drops any rows with null values in the 'Recovered' column.
    4. Selects the relevant features and splits the data into training and testing sets.
    5. Defines several regression models to be trained and their hyperparameters, and creates hyperparameter grids for the Ridge and Random Forest Regressor models.
    6. Fits and evaluates each model with hyperparameter tuning, using RMSE and R2 as evaluation metrics.
    7. Stores the results in a dictionary and saves them to a JSON file.

    Usage: 
        - Load the 'covid_fat_recov.csv' dataset into the working directory and run the code.

    Output:
        - Prints the RMSE and R2 scores for each model.
        - Saves the results to a 'results.json' file.
    """

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
raw_data = pd.read_csv('C:/Users/olomuc/OneDrive - MHRA/COVID_ML_MODELS/covid_fat_recov.csv')

# print the shape
print(raw_data.shape)

for column in raw_data.select_dtypes(include=np.number):
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
raw_data.dropna(subset=['Recovered'], inplace=True)

# select only the required features
X = raw_data[['Animal fats', 'Fish Seafood', 'Fruits - Excluding Wine', 'Milk - Excluding Butter', 'Vegetables', 'Obesity', 'Animal products total']].values
X_columns = ['Animal fats', 'Fish Seafood', 'Fruits - Excluding Wine', 'Milk - Excluding Butter', 'Vegetables', 'Obesity', 'Animal products total']
y = raw_data['Recovered'].astype(int)


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size = 0.2, random_state=15)

# define the models to be trained and their hyperparameters
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(),
    'XGBoost Regressor': XGBRegressor()
}

# define the hyperparameters to be tuned for the Ridge and Random Forest Regressor models
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}

# create a dictionary of hyperparameter grids for each model
param_grids = {
    'Ridge Regression': ridge_params,
    'Random Forest Regressor': rf_params
}

# create a dictionary to store the results
results = {}

# fit and evaluate the models with hyperparameter tuning
for name, model in models.items():
    print(f"Training {name}...")
    if name in param_grids:
        # perform grid search with cross-validation to find best hyperparameters
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - RMSE: {rmse:.4f}, R^2: {r2:.4f}")
    # add results to dictionary
    results[name] = {'RMSE': rmse, 'R2': r2}

# save results to JSON file
with open('results.json', 'w') as f:
    json.dump(results, f)