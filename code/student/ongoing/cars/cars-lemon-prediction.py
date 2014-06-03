#Load Modules
import pandas as pd 
from sklearn import tree

##Set Default Data Directory
DATA_DIR = '../../../../data/cars/'

#Load in data to create training and test datasets 

l_train = pd.read_csv(DATA_DIR + 'lemon_training.csv')
l_test = pd.read_csv(DATA_DIR + 'lemon_test.csv')
l_train = l_train.dropna(axis = 1)
l_test = l_test.dropna(axis = 1)

#Generate a list of features and removing RefId and IsBadBuy
features = list(l_train.describe().columns)
features.remove('IsBadBuy')
features.remove('RefId')

#Create Training Set

train_X = l_train[features].values
train_y = l_train['IsBadBuy'].values
test_X = l_test[features].values

#Create Classifier and Prediction

clf = tree.DecisionTreeClassifier().fit(train_X,train_y)
clf.score(train_X,train_y)
y_pred = clf.predict(test_X)

#Create a submission
submission = pd.DataFrame({'RefId' : l_test.RefId, 'Prediction' : y_pred})
submission.to_csv('submission.csv')


