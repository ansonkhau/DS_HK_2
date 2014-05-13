import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nytimes = pd.read_csv('/Users/ansonau/DS_HK_2/data/' + 'nyagg.csv')

from sklearn import linear_model
regr = linear_model.LinearRegression()
print nytimes.describe()
#plt.scatter(nytimes['Age'],nytimes['Ctr'])
#plt.show()
Age = [[x] for x in nytimes['Age'].values]
Ctr = nytimes['Ctr'].values
regr.fit(Age,Ctr)
#Display the Coefficients
print regr.coef_
#Display our SSE:
print np.mean((regr.predict(Age)-Ctr)**2)
#Display regr.score
print regr.score(Age,Ctr)

plt.scatter(Age, Ctr)
plt.plot(Age, regr.predict(Age), color='blue', linewidth=3)
plt.show()