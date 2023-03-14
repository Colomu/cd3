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


# 1. LOADING RAW DATA

# Loading the data
raw_data = pd.read_csv('C:/Users/olomuc/OneDrive - MHRA/COVID_ML_MODELS/covid_fat_deaths.csv')

# select only the required features
features = raw_data.drop(['Deaths'], axis=1)


# print the shape
print(features.shape)


for column in raw_data:
    unique_vals = np.unique(raw_data[column])
    nr_values = len(unique_vals)
    if nr_values < 10:
        print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
    else:
        print('The number of values for feature {} :{}'.format(column, nr_values))


# 2. DATA CLEANING AND PREPARATION

# check for null values
print(raw_data.isnull().sum())

# manually creating null values
raw_data['Pulses 2'] = raw_data['Pulses']
raw_data.loc[0, 'Pulses 2'] = np.nan
raw_data.loc[1, 'Pulses 2'] = np.nan
raw_data.loc[2, 'Pulses 2'] = np.nan

# impute null values
imp_mean = SimpleImputer(strategy='mean')
raw_data['Pulses 2'] = imp_mean.fit_transform(raw_data[['Pulses 2']])

# dropping the column
raw_data.drop(columns=['Pulses 2'], inplace=True)

# drop rows with NaN values in the 'Deaths' column
raw_data.dropna(subset=['Deaths'], inplace=True)

# select only the required features
#X = raw_data[['Animal fats', 'Fish Seafood', 'Fruits - Excluding Wine', 'Milk - Excluding Butter', 'Vegetables', 'Obesity', 'Animal products total']].values
#X_columns = ['Animal fats', 'Fish Seafood', 'Fruits - Excluding Wine', 'Milk - Excluding Butter', 'Vegetables', 'Obesity', 'Animal products total']
#y = raw_data['Deaths'].astype(int)



df = raw_data
labels = df['Deaths']
features = raw_data.drop(['Deaths'], axis=1)

def split_data_sets(features, labels):
    
    labels = np.ravel(labels)


    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    return (X_train, y_train, X_val, y_val, X_test, y_test)

print(features.shape)
print(labels.shape)

def calculate_regression_metrics(X_train, y_train, X_val, y_val, X_test, y_test):
    model = SGDRegressor(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
    
    y_val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = sqrt(mean_squared_error(y_val, y_val_pred))
    
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("R^2 score of training set:", train_r2)
    print("Root Mean Squared Error of training set:", train_rmse)
    print("R^2 score of validation set:", val_r2)
    print("Root Mean Squared Error of validation set:", val_rmse)
    print("R^2 score of testing set:", test_r2)
    print("Root Mean Squared Error of testing set:", test_rmse)

    metrics = {
    'train': {
        'r2': train_r2,
        'rmse': train_rmse,
        
    },
    'val': {
        'r2': val_r2,
        'rmse': val_rmse,
       
    },
    'test': {
        'r2': test_r2,
        'rmse': test_rmse,
        
}}
    return metrics

    

def tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, param_grid):
    
    if param_grid is None:
        raise ValueError(f"No param_grid specified for {model_class.__name__}")
    # check if the parameter "C" exist in the grid and remove it
    if 'C' in param_grid:
        param_grid.pop("C")
    model = model_class()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    y_val_pred = best_model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = sqrt(mean_squared_error(y_val, y_val_pred))
    y_test_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    performance_metrics = {"validation_r2": val_r2,"test_r2":test_r2, "validation_rmes": val_rmse, "test_rmes": test_rmse}

    
    return best_model, best_params, performance_metrics


def evaluate_all_models(X_train, y_train, X_val, y_val, X_test, y_test):
    models = {
              "DecisionTreeRegressor": DecisionTreeRegressor(), 
              "RandomForestRegressor": RandomForestRegressor(), 
              "GradientBoostingRegressor": GradientBoostingRegressor(),
             "SGDRegressor": SGDRegressor() }
    
    param_grids = {"DecisionTreeRegressor": {"max_depth": [5, 10, 15]}, 
                   "RandomForestRegressor": {"max_depth": [5, 10, 15]}, 
                   "GradientBoostingRegressor": {"learning_rate": [0.1, 0.05, 0.01], "max_depth": [5, 10, 15]},
                   "SGDRegressor": {"max_iter": [1000, 5000, 10000], "tol": [1e-3, 1e-4, 1e-5]}}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
        y_val_pred = model.predict(X_val)
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = sqrt(mean_squared_error(y_val, y_val_pred))
        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
        print(f"{name} train R^2: {train_r2:.4f}")
        print(f"{name} train RMSE: {train_rmse:.4f}")
        print(f"{name} validation R^2: {val_r2:.4f}")
        print(f"{name} validation RMSE: {val_rmse:.4f}")
        print(f"{name} test R^2: {test_r2:.4f}")
        print(f"{name} test RMSE: {test_rmse:.4f}")
        best_params = tune_regression_model_hyperparameters(model.__class__, X_train, y_train, X_val, y_val, X_test, y_test, param_grids.get(name))
        print(f"Best parameters for {name}: {best_params}")
        

def find_best_model(models, param_grid_list, X_train, y_train, X_val, y_val, X_test, y_test):
    best_models = {}
    best_params = {}
    best_metrics = {}
    best_val_acc = 0
    best_model_name = ""
    model_folder = 'regression_models'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    for i, (model, param_grid) in enumerate(zip(models, param_grid_list)):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params[type(best_model).__name__] = grid_search.best_params_
        best_models[type(best_model).__name__] = best_model
        
        y_val_pred = best_model.predict(X_val)
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = sqrt(mean_squared_error(y_val, y_val_pred))
        
        
        best_metrics[type(best_model).__name__] = {
            'r2': val_r2,
            'rmse': val_rmse,
            
        }
        best_val_r2 = -1  # initialize the variable with a default value
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_name = type(best_model).__name__
        
        print(f"{type(best_model).__name__} - Validation r2: {val_r2:.3f}, Validation rmse: {val_rmse:.3f}")

        model_name = type(best_model).__name__
        model_path = f"{model_folder}/{model_name}/model.joblib"
        if not os.path.exists(f"{model_folder}/{model_name}"):
            os.makedirs(f"{model_folder}/{model_name}")
        joblib.dump(best_model, model_path)
        param_path = f"{model_folder}/{model_name}/hyperparameters.json"
        with open(param_path, "w") as f:
            json.dump(best_params, f)
        metric_path = f"{model_folder}/{model_name}/metrics.json"
        with open(metric_path, "w") as f:
            json.dump(best_metrics, f)   
        
       
    return best_models, best_params, best_metrics, best_val_acc, best_model_name


models = [SGDRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor()]
param_grid_list = [
{'alpha': [0.0001, 0.001, 0.01, 0.1], 'l1_ratio': [0, 0.1, 0.5, 0.9], 'penalty': ['l1', 'l2', 'elasticnet']},
{'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
{'n_estimators': [50, 100, 200, 500], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
{'n_estimators': [50, 100, 200, 500], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
{'n_estimators': [50, 100, 200, 500], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
]
model_names = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
# load data into 'features' and 'labels' variables
labels = ['Deaths']
features = raw_data.drop(['Deaths'], axis=1)

# check if the variables are defined
if features is not None and labels is not None:
    # pass the variables as arguments to the function
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_sets(features, labels)
    for idx, model in enumerate(models):
        grid = GridSearchCV(model, param_grid_list[idx], cv = 5)
        grid.fit(X_train, y_train)
else:
    print("Features and labels are not defined. Please make sure to load the data correctly.")






        

if __name__ == "__main__":
    labels = ['Deaths']
    features = raw_data.drop(['Deaths'], axis=1)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data_sets(features, labels)
    calculate_regression_metrics(X_train, y_train, X_val, y_val, X_test, y_test)
    metrics = evaluate_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
    find_best_model(models, param_grid_list, X_train, y_train, X_val, y_val, X_test, y_test)
    