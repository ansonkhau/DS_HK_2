import pandas as pd 
import numpy as np 
from sklearn import linear_model, feature_selection
from sklearn.feature_selection.univariate_selection import f_regression

# Store data in a consistent place
DATA_DIR = '/users/ansonau/DS_HK_2/data/'

cars = pd.read_csv(DATA_DIR + 'cars93.csv')

X = cars[['EngineSize','Horsepower','RPM','Rev.per.mile','Fuel.tank.capacity','Passengers','Length','Wheelbase','Width','Turn.circle','Weight']]
f,p = f_regression(X, cars['MPG.city'].values)  
print [ [x, y] for x,y in zip(f, p)]


# Regression on Weight

#Polynomial Regression 
cars['Weight_squared'] = cars['Weight']**2
cars['Weight_cubed'] = cars['Weight'] **3
MPG = cars['MPG.city']
Weight_poly = [ [x, y, z] for x,y,z in zip(cars['Weight'].values, cars['Weight_squared'].values,cars['Weight_cubed'])]

ridge = linear_model.Ridge()
ridge.fit(Weight_poly,MPG)

print ridge.coef_[2]
print ridge.coef_[1]
print ridge.coef_[0]


response = ((ridge.coef_[2] * cars['Weight'])**3)+((ridge.coef_[1] * cars['Weight'])**2) + ((ridge.coef_[0] * cars['Weight'])) + ridge.intercept_

print ridge.score(cars[['Weight','Weight_squared','Weight_cubed']].values, MPG)
