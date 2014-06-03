import pandas as pd
import matplotlib.pyplot as plt
from numpy import log, exp, mean
from sklearn import linear_model, feature_selection, metrics

DATA_DIR = '../../../../../data/baseball/'

b2011 = pd.read_csv(DATA_DIR + 'baseball_training_2011.csv')
b2012 = pd.read_csv(DATA_DIR + 'baseball_test_2012.csv')

#Determine the features that highly influence baseball salary using 2011 training set
train_x = b2011[['G','G_batting','AB','R','H','X2B','X3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP','G_old']]
train_y = b2011 ['salary'].values

fp_value = feature_selection.univariate_selection.f_regression(train_x, train_y)
p_value = zip(train_x.columns.values,fp_value[1])
p_value_sorted = sorted(p_value,key=lambda x: x[1])

#Square the top 5 lowest p-values

b2011['RBI_squared'] = b2011 ['RBI'] **2
b2012['RBI_squared'] = b2012 ['RBI'] **2

b2011['HR_squared'] = b2011 ['HR'] **2
b2012['HR_squared'] = b2012 ['HR'] **2

b2011['BB_squared'] = b2011 ['BB'] **2
b2012['BB_squared'] = b2012 ['BB'] **2

b2011['IBB_squared'] = b2011 ['IBB'] **2
b2012['IBB_squared'] = b2012 ['IBB'] **2

b2011['GIDP_squared'] = b2011 ['GIDP'] **4
b2012['GIDP_squared'] = b2012 ['GIDP'] **4

#train_x = b2011[['RBI','RBI_squared', 'HR','HR_squared','BB', 'IBB', 'GIDP', 'R', 'H', 'AB', 'X2B', 'SF', 'G', 'SO']].values
train_x = b2011[['RBI_squared', 'HR_squared','BB_squared', 'IBB_squared', 'GIDP_squared','HR', 'RBI', 'G', 'AB', 'R', 'H', 'X2B', 'SB', 'X3B','CS', 'BB', 'SO', 'IBB', 'HBP', 'SF']].values

lm = linear_model.Ridge()
lm.fit(train_x,train_y)

#Check Performance Rscore
print 'R-Squared:',lm.score(train_x, train_y)

print 'MSE:',metrics.mean_squared_error(lm.predict(train_x), train_y)


#Apply on 2012 Test Data
#test_x = b2012[['RBI','RBI_squared', 'HR','HR_squared','BB', 'IBB', 'GIDP', 'R', 'H', 'AB', 'X2B', 'SF', 'G', 'SO']].values
test_x = b2012[['RBI_squared', 'HR_squared','BB_squared','IBB_squared','GIDP_squared', 'HR', 'RBI','G', 'AB', 'R', 'H', 'X2B', 'SB', 'X3B','CS', 'BB', 'SO', 'IBB', 'HBP', 'SF']].values

b2012_csv = b2012[['playerID','yearID', 'salary']]

# Outputting to a csv file
print "Outputting submission file as 'submission.csv'"
b2012_csv['predicted'] = lm.predict(test_x)
b2012_csv.to_csv('submission.csv')




