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
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib
from math import sqrt

"""
    Trains and evaluates several regression models on a COVID-19 dataset, using features such as animal fats, fruits, and obesity to predict the number of deaths in different countries.

    Performs the following steps:

    1. Sets options for printing and warning suppression.
    2. Loads the raw data from a CSV file and prints out the number of values for each feature.
    3. Checks for null values in the data and replaces any null values with the mean of the non-null values. Drops any rows with null values in the 'Deaths' column.
    4. Selects the relevant features and splits the data into training and testing sets.
    5. Creates scatterplot matrix, scatterplot matrix with regression lines, and checks for missing or invalid values.
    6. Defines several regression models to be trained and their hyperparameters, and creates hyperparameter grids for the Ridge and Random Forest Regressor models.
    7. Fits and evaluates each model with hyperparameter tuning, using RMSE and R2 as evaluation metrics.
    8. Stores the results in a dictionary and saves them to a JSON file.
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

#runs the first 5 rows
print(raw_data.head())

#2. DATA PROCESSING

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


print(raw_data.head(10))

#4 NUMERIC VALUES SCATTER PLOT

# Create the scatterplot matrix and save to a new file
sns.pairplot(raw_data)
plt.savefig('scatterplot_matrix.png')
print('Scatterplot matrix saved to scatterplot_matrix.png')

# Create the scatterplot matrix with regression lines and R-squared values
g = sns.PairGrid(raw_data[['Deaths', 'Animal fats', 'Fish Seafood', 'Fruits - Excluding Wine', 'Milk - Excluding Butter', 'Vegetables', 'Obesity', 'Animal products total']])
g.map_upper(sns.regplot, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
g.map_upper(lambda x, y, **kwargs: plt.text(x.max(), y.min(), f"r$^2$={linregress(x, y).rvalue**2:.2f}", ha='right', va='bottom'))
g.map_lower(sns.kdeplot, cmap='Blues_d')
g.map_diag(sns.histplot, kde=False, bins=10)
plt.savefig('scatterplot_matrix2.png')
print('Scatterplot matrix with regression lines and R-squared values saved to scatterplot_matrix2.png')

# Check for missing or invalid values
if raw_data.isnull().values.any():
    print('There are missing or invalid values in the data')
    print(raw_data.isnull().sum())
else:
    print('There are no missing or invalid values in the data')

# Check for categorical variables
cat_cols = []
for col in raw_data.columns:
    if raw_data[col].dtype == 'object':
        cat_cols.append(col)
if cat_cols:
    print('The following columns contain categorical variables:')
    print(cat_cols)
else:
    print('There are no categorical variables in the data')
    
    
# Create the scatterplot with regression line and R-squared value for Deaths and Animal fats

#g = sns.regplot(x='Animal fats', y='Deaths', data=raw_data, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
#plt.text(g.get_xlim()[1], g.get_ylim()[0], f"r$^2$={linregress(raw_data['Animal fats'], raw_data['Deaths']).rvalue**2:.2f}", ha='right', va='bottom')
#plt.savefig('scatterplot_matrix3.png')
#print('Scatterplot with regression line and R-squared value saved to scatterplot_matrix3.png')

#data = pd.read_csv('covid_fat_deaths2.csv')

# Extract the 'deaths' and 'Animal fat' columns
#Deaths = data['Deaths']
#Animal_fat = data['Animal fats']



#ax = sns.scatterplot(x="Animal fats", y="Deaths", data=raw_data, s=90)

# Create a scatter plot with a regression line
#sns.regplot(x=Animal_fat, y=Deaths, scatter=True)

# Calculate the R2 value
#X = Animal_fat.values.reshape(-1, 1)
#y = Deaths.values.reshape(-1, 1)
#model = LinearRegression().fit(X, y)
#r2 = r2_score(y, model.predict(X))
#plt.annotate(f"R2={r2:.2f}", xy=(0.05, 0.95), xycoords='axes fraction')

# Save the plot to a file
#plt.savefig('scatterplot.png')

# Split the data into X & y

X = raw_data.drop(['Deaths'], axis = 1).values
X_columns = raw_data.drop(['Deaths'], axis = 1)
y = raw_data['Deaths'].astype(int)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size = 0.2, random_state=15)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
