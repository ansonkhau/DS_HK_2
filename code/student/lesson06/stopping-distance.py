import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

# Store data in a consistent place
DATA_DIR = '/users/ansonau/DS_HK_2/data/'
from sklearn import linear_model

cars = pd.read_csv(DATA_DIR + 'cars1920.csv')

lm = linear_model.LinearRegression()
log_lm = linear_model.LinearRegression()

#Polynomial Regression 
cars['speed_squared'] = cars['speed']**2
dist = cars['dist']
cars_squared = [ [x, y] for x,y in zip(cars['speed'].values, cars['speed_squared'].values)]

ridge = linear_model.Ridge()
ridge.fit(cars_squared,dist)

print ridge.coef_[1]
print ridge.coef_[0]


response = ((ridge.coef_[1] * cars['speed'])**2) + ((ridge.coef_[0] * cars['speed'])) + ridge.intercept_

print ridge.score(cars[['speed','speed_squared']].values, dist)