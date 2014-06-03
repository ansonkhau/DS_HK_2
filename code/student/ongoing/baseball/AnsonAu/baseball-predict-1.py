# Baseball Prediction with 2011 Training Data

import pandas as pd
import matplotlib.pyplot as plt
from numpy import log, exp, mean
from sklearn import linear_model, feature_selection

DATA_DIR = '../../../../../data/baseball/'

#Read Baseball Data

baseball = pd.read_csv(DATA_DIR + 'baseball_training_2011.csv')

#Get numeric data 

baseball_input = baseball._get_numeric_data()

salary = baseball_input['salary']

#Attempt to clear data with only numeric values not working yet!

#baseball_input = baseball_input.dropna(axis=0)
#baseball_input = baseball_input.drop(['salary'],1)
#baseball_input = baseball_input.fillna(0)

baseball_input = baseball_input[['G','G_batting','AB','R','H','X2B','X3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP','G_old']]

fp_value = feature_selection.univariate_selection.f_regression(baseball_input, salary)
p_value = zip(baseball_input.columns.values,fp_value[1])
p_value_sorted = sorted(p_value,key=lambda x: x[1])

#Since HR has the lowest xPP-Value, look into more details
baseball_input = baseball[baseball['RBI'] > 0]
rbi = [[x] for x in baseball_input['RBI']]
salary = baseball_input['salary'].values
rbi_salary = zip(rbi,salary)

#Scatter Plot rbi against salary
plt.scatter(rbi,salary, c='b', marker = 'o')
plt.show()

#Linear regression of rbi and salary

def SSE(pred, resp):
 	return mean((pred - resp) ** 2)

regr = linear_model.LinearRegression()
regr.fit(rbi,salary)

print "\nRBI | Salary"
print "SSE : %0.4f" % (SSE(regr.predict(rbi), salary)) 
print "R2 : %0.4f" % (regr.score(rbi, salary)) 

plt.scatter(rbi, salary, c='b', marker='o')
plt.plot(rbi, regr.predict(rbi), color='green')
plt.show()

#Terrible fit with simple linear model with score of 0.1018

#Lasso Regression with 3rd order polynomial
baseball_input['RBI_squared'] = baseball_input['RBI'] ** 2
rbi_squared = baseball_input[['RBI','RBI_squared']].values

lasso = linear_model.Lasso()
lasso.fit(rbi_squared,salary)

print "\nRBI | Salary"
print "SSE : %0.4f" % (SSE(lasso.predict(rbi_squared), salary)) 
print "R2 : %0.4f" % (lasso.score(rbi_squared, salary)) 

plt.scatter(rbi, salary, c='b', marker='o')
plt.plot(rbi_squared, lasso.predict(rbi_squared), color='green')
plt.show()

#Terrible Fit with Lasso 3rd order polynomial 0.13

#Try Ridge Fit with Polynomial Regression for RBI
baseball_input['RBI_cubed'] = baseball_input['RBI'] **3
rbi_cubed = baseball_input[['RBI', 'RBI_squared', 'RBI_cubed']].values

ridge = linear_model.Ridge()
ridge.fit(rbi_cubed,salary)

print "\nRBI | Salary"
print "SSE : %0.4f" % (SSE(ridge.predict(rbi_cubed), salary)) 
print "R2 : %0.4f" % (ridge.score(rbi_cubed, salary)) 

plt.scatter(rbi, salary, c='b', marker='o')
plt.plot(rbi_cubed, ridge.predict(rbi_cubed), color='green')
plt.show()


