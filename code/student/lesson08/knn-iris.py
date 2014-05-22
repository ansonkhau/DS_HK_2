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
DATA_DIR = "../data/"

from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, feature_selection

#Various variables we'll need to set initially
n_neighbors = range(1,51,2)
np.random.seed(32)

#Load in the data and separate class labels and input data
iris = datasets.load_iris()
x = iris.data
y = iris.target

#Create the training (and test) set using the rng in numpy
n = int(len(y) * .7)
# Stack True and False (ones and zeros) in dataset 
ind = np.hstack((np.ones(n, dtype=np.bool), np.zeros(len(y)-n,dtype=np.bool)))
np.random.shuffle(ind)
x_train, x_test = x[ind], x[ind == False]
y_train, y_test = y[ind], y[ind == False]

#Loop through each neighbors value from 1 to 51 and append the scores
scores = []
for n in n_neighbors:
	clf = neighbors.KNeighborsClassifier(n)
	clf.fit(x_train,y_train)
	scores.append(clf.score(x_test,y_test))

plt.figure(figsize=(20,8))
plt.plot(n_neighbors, scores, linewidth = 3.0)
plt.show()
scores

# Cross Validation
scores = []
for k in range(5):
	#Random shuffle
	np.random.shuffle(ind)
	x_train, x_test = x[ind], x[ind == False]
	y_train, y_test = y[ind], y[ind == False]
	clf = neighbors.KNeighborsClassifier(11, weights = 'uniform')
	clf.fit(x_train, y_train)
	scores.append(clf.score(x_test,y_test))