import pandas as pd 
import numpy as np 
from sklearn import linear_model, feature_selection
from sklearn.feature_selection.univariate_selection import f_regression

# Store data in a consistent place
DATA_DIR = '/users/ansonau/DS_HK_2/data/'

cars = pd.read_csv(DATA_DIR + 'cars93.csv')

X = cars[['EngineSize','Horsepower','RPM','Rev.per.mile','Fuel.tank.capacity','Passengers','Length','Wheelbase','Width','Turn.circle']]
f,p = f_regression(X, cars['MPG.highway'].values)  
print [ [x, y] for x,y in zip(f, p)]


#,'Rear.seat.room','Luggage.room','Weight'
