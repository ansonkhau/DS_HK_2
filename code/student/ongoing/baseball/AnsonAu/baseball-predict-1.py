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
#baseball_input = baseball_input.dropna(axis=0)
#baseball_input = baseball_input.drop(['salary'],1)
#baseball_input = baseball_input.fillna(0)
baseball_input = baseball_input[['G','G_batting','AB','R','H','X2B','X3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP','G_old']]

fp_value = feature_selection.univariate_selection.f_regression(baseball_input, salary)
p_value = zip(baseball_input.columns.values,fp_value[1])
p_value_sorted = sorted(p_value,key=lambda x: x[1])

#Since RBI has the lowest P-Value, look into more details
baseball_input = baseball[baseball['RBI'] > 0]
rbi = [[x] for x in baseball_input['RBI']]
salary = baseball_input['salary'].values
rbi_salary = zip(rbi,salary)

#Scatter Plot rbi against salary
plt.scatter(rbi,salary, c='b', marker = 'o')
plt.show()



