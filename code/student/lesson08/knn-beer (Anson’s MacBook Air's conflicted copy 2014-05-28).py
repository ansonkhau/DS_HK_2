import matplotlib
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# Don't show depreciation warnings 
import warnings
warnings.filterwarnings ("ignore", category=DeprecationWarning)

#Set some Pandas Options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

#Set some Pandas Options
pd.set_option('max_columns',30)
pd.set_option('max_rows',20)

#Set some matplotlib Options
matplotlib.rcParams.update({'font.size': 20})

#Store data in consistent place
DATA_DIR = "../../../data/"

from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, feature_selection 

# good function, if rating > 4.3 then it is a good beer

def good (x):
	if x > 4.3:
		return 1
	else:
		return 0

beer = pd.read_csv(DATA_DIR + 'beer.csv', delimiter= '\t')
beer = beer.dropna()
beer['Good'] = beer['WR'].apply(good)



# Divide into different beer types

beer_type = ['Ale', 'Stout', 'IPA', 'Lager']

for i in beer_type:
	beer[i] = beer['Type'].str.contains(i) * 1

#KNN

# Intiial Variables
n_neighbors = range(1,101,2)

# Separate Beer input from Results

x = beer[beer_type].values
y = beer['Good'].values

# Create Test and Data Set using rng in Numpy
n = int(len(y)*.7)
# Stack True and False in Data Set
ind = np.hstack((np.ones(n, dtype=np.bool), np.zeros(len(y)-n,dtype=np.bool)))
np.random.shuffle(ind)

x_train, x_test = x[ind], x[ind == False]
y_train, y_test = y[ind], y[ind == False]

# Loop through each neighbors value from 1 to 101 and append the scores
scores = []

for n in n_neighbors:
	clf = neighbors.KNeighborsClassifier(n)
	clf.fit (x_train,y_train)
	scores.append(clf.score(x_test,y_test))

plt.figure(figsize = (20,8))
plt.plot(n_neighbors, scores, linewidth = 3.0)
plt.show()
scores

# Cross Validation
scores_cv = []
for k in range(5):
	#Random Shuffle 
	x_train, y_test = x[ind], x[ind == False]
	y_train, y_test = y[ind], y[ind == False]
	np.random.shuffle(ind)
	clf = neighbors.KNeighborsClassifier(11, weights = "uniform")
	clf.fit (x_train,y_train)
	scores_cv.append(clf.score(x_test,y_test))

plt.figure(figsize = (20,8))
plt.plot(n_neighbors, scores_cv, linewidth = 3.0)
plt.show()