#import pandas as pd 
#from pandas import read_csv, DataFrame
#import numpy as np 
#from sklearn import linear_model, feature_selection
#from sklearn.feature_selection.univariate_selection import f_regression

from pandas import read_csv, DataFrame
from numpy import mean
from sklearn import linear_model, feature_selection
from sklearn.feature_selection.univariate_selection import f_regression

# Store data in a consistent place
DATA_DIR = '/users/ansonau/DS_HK_2/data/'

cars = read_csv(DATA_DIR + 'cars93.csv')


#Clean Data
#Get only Numeric Data from cars
car_x = cars._get_numeric_data()
car_x = car_x.fillna(car_x.mean())
#Drop y Values
car_x = cars.drop(['MPG.highway','MPG.city'],1)
#car_x = cars[['EngineSize','Horsepower','RPM','Rev.per.mile','Fuel.tank.capacity','Passengers','Length','Wheelbase','Width','Turn.circle','Weight']]
car_y = cars['MPG.highway']

f,p = f_regression(car_x, car_y.values)  
print [ [x, y] for x,y in zip(f, p)]


# Regression on Weight

#Polynomial Regression 
cars['Weight_squared'] = cars['Weight']**2
cars['Weight_cubed'] = cars['Weight'] **3
cars['Weigh_fourth'] = cars['Weight'] **4
MPG = cars['MPG.city']
Weight_poly = [ [w, x, y, z] for w, x,y,z in zip(cars['Weight'].values, cars['Weight_squared'].values,cars['Weight_cubed'],cars['Weigh_fourth'])]

ridge = linear_model.Ridge()
ridge.fit(Weight_poly,MPG)
#s = 'The value of x is ' + repr(x) + ', and y is ' + repr(y) + '...'
print 'Coef 1: ' + str(ridge.coef_[3])
print 'Coef 2: ' + str(ridge.coef_[2])
print 'Coef 3: ' + str(ridge.coef_[1])
print 'Coef 0: ' + str(ridge.coef_[0])



response = ((ridge.coef_[2] * cars['Weight'])**4)+((ridge.coef_[2] * cars['Weight'])**3)+((ridge.coef_[1] * cars['Weight'])**2) + ((ridge.coef_[0] * cars['Weight'])) + ridge.intercept_

print 'Ridge Score: ' + str(ridge.score(cars[['Weight','Weight_squared','Weight_cubed','Weigh_fourth']].values, MPG))
