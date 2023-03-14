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
from geopy.geocoders import Nominatim


# read the existing CSV file into a pandas dataframe
df = pd.read_csv("covid_fat_death2.csv")

# normalize the country names in the 'Country' column
df['Country'] = df['Country'].str.strip().str.title().str.replace('[^\w\s]', '')

# create a geolocator object using the Nominatim service
geolocator = Nominatim(user_agent="my_app")



geolocator = Nominatim(user_agent="my_app")

def get_continent(country):
    try:
        location = geolocator.geocode(country, exactly_one=True, addressdetails=True)
        country_code = location.raw['address']['country_code']
        continent_code = pycountry.countries.get(alpha_2=country_code).continent
        continent_name = continent_code.name
        return continent_name
    except:
        return "Unknown"



# add a new 'Continent' column to the dataframe
df['Continent'] = df['Country'].apply(get_continent)

# save the updated dataframe to the CSV file
df.to_csv("covid_fat_death2.csv", index=False)

print('hello world')
